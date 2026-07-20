from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model import ConvLSTMCell
from .model_variants import (
    MODEL_CNN_TEMPORAL_TRANSFORMER,
    MODEL_CONVLSTM,
    MODEL_CONVLSTM_ATTENTION,
    TemporalSpatialAttention,
    normalize_model_type,
)


class ExternalConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_max: float = 3.5,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_max = float(output_max)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.cells = nn.ModuleList(
            [ConvLSTMCell(hidden_channels, hidden_channels)]
            + [ConvLSTMCell(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        sequence: list[torch.Tensor] = []
        for time_index in range(x.shape[1]):
            feature = self.encoder(x[:, time_index])
            for layer_index, cell in enumerate(self.cells):
                hidden, memory = cell(feature, states[layer_index])
                states[layer_index] = (hidden, memory)
                feature = hidden
            sequence.append(feature)
        return sequence[-1], sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        feature, _ = self.encode(x)
        raw = self.head(feature)
        if self.use_residual:
            base = x[:, -1, 0:1] * self.output_max
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)


class ExternalConvLSTMAttention(ExternalConvLSTM):
    def __init__(self, *args, attention_dropout: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_channels = self.head[0].in_channels
        self.attention = TemporalSpatialAttention(hidden_channels, dropout=attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        _, sequence = self.encode(x)
        context, _ = self.attention(torch.stack(sequence, dim=1))
        raw = self.head(context)
        if self.use_residual:
            base = x[:, -1, 0:1] * self.output_max
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)


class ExternalCNNTemporalTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.0,
        output_max: float = 3.5,
        max_input_len: int = 64,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_channels % transformer_heads != 0:
            raise ValueError("hidden_channels must be divisible by transformer_heads")
        self.output_max = float(output_max)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        self.max_input_len = int(max_input_len)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position = nn.Parameter(torch.zeros(1, self.max_input_len, hidden_channels))
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        batch, time, _, height, width = x.shape
        if time > self.max_input_len:
            raise ValueError(f"Input length {time} exceeds {self.max_input_len}")
        frames = torch.stack([self.frame_encoder(x[:, index]) for index in range(time)], dim=1)
        tokens = frames.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, -1)
        encoded = self.temporal_encoder(tokens + self.position[:, :time])[:, -1]
        feature = encoded.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        raw = self.head(feature)
        if self.use_residual:
            base = x[:, -1, 0:1] * self.output_max
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)


def build_external_model(
    model_type: str,
    input_channels: int,
    output_channels: int,
    hidden_channels: int = 16,
    num_layers: int = 1,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    transformer_heads: int = 4,
    output_max: float = 3.5,
    max_input_len: int = 64,
    use_residual: bool = True,
    residual_scale: float = 1.0,
) -> nn.Module:
    model_type = normalize_model_type(model_type)
    common = dict(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        output_max=output_max,
        use_residual=use_residual,
        residual_scale=residual_scale,
    )
    if model_type == MODEL_CONVLSTM:
        return ExternalConvLSTM(**common)
    if model_type == MODEL_CONVLSTM_ATTENTION:
        return ExternalConvLSTMAttention(**common, attention_dropout=attention_dropout)
    if model_type == MODEL_CNN_TEMPORAL_TRANSFORMER:
        return ExternalCNNTemporalTransformer(
            **common,
            transformer_heads=transformer_heads,
            max_input_len=max_input_len,
        )
    raise ValueError(f"Unknown external model type: {model_type}")


def count_external_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_external_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    args = checkpoint.get("args", {})
    lead_times = tuple(int(value) for value in checkpoint["lead_times"])
    return build_external_model(
        str(checkpoint["model_type"]),
        input_channels=int(checkpoint["input_channels"]),
        output_channels=len(lead_times),
        hidden_channels=int(checkpoint["hidden_channels"]),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(args.get("dropout", 0.0)),
        attention_dropout=float(args.get("attention_dropout", 0.0)),
        transformer_heads=int(args.get("transformer_heads", 4)),
        output_max=float(checkpoint["depth_scale_m"]),
        max_input_len=int(checkpoint["input_len"]),
        use_residual=bool(checkpoint.get("use_residual", True)),
        residual_scale=float(checkpoint.get("residual_scale", 1.0)),
    )
