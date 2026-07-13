from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .data.schemas import DEFAULT_DEPTH_SCALE, depth_scale_from_checkpoint
from .model import ConvLSTMCell


MODEL_UNET_SINGLE_FRAME = "unet_single_frame"
MODEL_CNN3D = "cnn3d"
MODEL_CONVGRU = "convgru"
MODEL_CONVLSTM_UNET = "convlstm_unet"
MODEL_TYPES = (MODEL_UNET_SINGLE_FRAME, MODEL_CNN3D, MODEL_CONVGRU, MODEL_CONVLSTM_UNET)


def model_display_name(model_type: str) -> str:
    return {
        MODEL_UNET_SINGLE_FRAME: "U-Net Single Frame",
        MODEL_CNN3D: "3D CNN",
        MODEL_CONVGRU: "ConvGRU",
        MODEL_CONVLSTM_UNET: "Multi-Horizon Conv-LSTM U-Net",
    }.get(model_type, model_type)


def _group_count(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvBlock2d(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )


class BoundedDepthHead(nn.Module):
    def __init__(self, input_channels: int, num_horizons: int, output_max: float) -> None:
        super().__init__()
        self.output_max = float(output_max)
        self.projection = nn.Conv2d(input_channels, num_horizons, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_max * torch.sigmoid(self.projection(x))


class SingleFrameUNet(nn.Module):
    """Strong spatial baseline that only observes the final input frame."""

    def __init__(self, input_channels: int, hidden_channels: int, num_horizons: int, dropout: float, output_max: float) -> None:
        super().__init__()
        self.encoder_full = ConvBlock2d(input_channels, hidden_channels, dropout)
        self.encoder_half = ConvBlock2d(hidden_channels, hidden_channels * 2, dropout)
        self.bottleneck = ConvBlock2d(hidden_channels * 2, hidden_channels * 2, dropout)
        self.decoder = ConvBlock2d(hidden_channels * 3, hidden_channels, dropout)
        self.head = BoundedDepthHead(hidden_channels, num_horizons, output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = self.encoder_full(x[:, -1])
        half = self.encoder_half(F.avg_pool2d(full, 2))
        bottleneck = self.bottleneck(half)
        up = F.interpolate(bottleneck, size=full.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder(torch.cat([up, full], dim=1)))


class CNN3DForecaster(nn.Module):
    """Spatiotemporal convolution baseline with last/mean temporal pooling."""

    def __init__(self, input_channels: int, hidden_channels: int, num_horizons: int, dropout: float, output_max: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.decoder = ConvBlock2d(hidden_channels * 2, hidden_channels, dropout)
        self.head = BoundedDepthHead(hidden_channels, num_horizons, output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x.permute(0, 2, 1, 3, 4))
        pooled = torch.cat([features[:, :, -1], features.mean(dim=2)], dim=1)
        return self.head(self.decoder(pooled))


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        total = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(total, hidden_channels * 2, 3, padding=1, padding_mode="reflect")
        self.candidate = nn.Conv2d(total, hidden_channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        reset, update = torch.chunk(torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))), 2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return update * hidden + (1.0 - update) * candidate


class ConvGRUForecaster(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_horizons: int, dropout: float, output_max: float) -> None:
        super().__init__()
        self.frame_encoder = ConvBlock2d(input_channels, hidden_channels, dropout)
        self.cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.decoder = ConvBlock2d(hidden_channels, hidden_channels, dropout)
        self.head = BoundedDepthHead(hidden_channels, num_horizons, output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = None
        for index in range(x.shape[1]):
            hidden = self.cell(self.frame_encoder(x[:, index]), hidden)
        return self.head(self.decoder(hidden))


class MultiHorizonConvLSTMUNet(nn.Module):
    """Multi-scale Conv-LSTM encoder with a U-Net-style spatial decoder."""

    def __init__(self, input_channels: int, hidden_channels: int, num_horizons: int, dropout: float, output_max: float) -> None:
        super().__init__()
        self.encoder_full = ConvBlock2d(input_channels, hidden_channels, dropout)
        self.encoder_half = ConvBlock2d(hidden_channels, hidden_channels * 2, dropout)
        self.temporal_cell = ConvLSTMCell(hidden_channels * 2, hidden_channels * 2)
        self.decoder = ConvBlock2d(hidden_channels * 3, hidden_channels, dropout)
        self.refinement = ConvBlock2d(hidden_channels, hidden_channels, dropout)
        self.head = BoundedDepthHead(hidden_channels, num_horizons, output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = None
        last_full = None
        for index in range(x.shape[1]):
            full = self.encoder_full(x[:, index])
            half = self.encoder_half(F.avg_pool2d(full, 2))
            state = self.temporal_cell(half, state)
            last_full = full
        hidden, _ = state
        up = F.interpolate(hidden, size=last_full.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder(torch.cat([up, last_full], dim=1))
        return self.head(self.refinement(decoded))


def build_batch4_model(
    model_type: str,
    input_channels: int,
    hidden_channels: int,
    num_horizons: int,
    dropout: float = 0.0,
    output_max: float = DEFAULT_DEPTH_SCALE.max_value,
) -> nn.Module:
    kwargs = {
        "input_channels": input_channels,
        "hidden_channels": hidden_channels,
        "num_horizons": num_horizons,
        "dropout": dropout,
        "output_max": output_max,
    }
    if model_type == MODEL_UNET_SINGLE_FRAME:
        return SingleFrameUNet(**kwargs)
    if model_type == MODEL_CNN3D:
        return CNN3DForecaster(**kwargs)
    if model_type == MODEL_CONVGRU:
        return ConvGRUForecaster(**kwargs)
    if model_type == MODEL_CONVLSTM_UNET:
        return MultiHorizonConvLSTMUNet(**kwargs)
    raise ValueError(f"Unknown Batch 4 model type: {model_type}")


def build_batch4_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    return build_batch4_model(
        model_type=str(checkpoint["model_type"]),
        input_channels=int(checkpoint["input_channels"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        num_horizons=len(checkpoint["lead_times"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        output_max=depth_scale_from_checkpoint(checkpoint).max_value,
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
