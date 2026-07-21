from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .model import ConvLSTMCell
from .model_variants import (
    MODEL_CNN_TEMPORAL_TRANSFORMER,
    MODEL_CONVLSTM,
    MODEL_CONVLSTM_ATTENTION,
    TemporalSpatialAttention,
    normalize_model_type,
)


MODEL_URNN_LITE = "urnn_lite"
MODEL_FNO2D_HISTORY = "fno2d_history"
MODEL_SIMVP_LITE = "simvp_lite"

EXTERNAL_MODEL_TYPES = (
    MODEL_CONVLSTM,
    MODEL_CONVLSTM_ATTENTION,
    MODEL_CNN_TEMPORAL_TRANSFORMER,
    MODEL_URNN_LITE,
    MODEL_FNO2D_HISTORY,
    MODEL_SIMVP_LITE,
)

EXTERNAL_MODEL_LABELS = {
    MODEL_CONVLSTM: "Conv-LSTM",
    MODEL_CONVLSTM_ATTENTION: "Conv-LSTM + Attention",
    MODEL_CNN_TEMPORAL_TRANSFORMER: "CNN-Temporal Transformer",
    MODEL_URNN_LITE: "U-RNN Lite (adapted)",
    MODEL_FNO2D_HISTORY: "FNO2D-History (adapted)",
    MODEL_SIMVP_LITE: "SimVP Lite (adapted)",
}

_EXTERNAL_MODEL_ALIASES = {
    "urnn": MODEL_URNN_LITE,
    "u_rnn": MODEL_URNN_LITE,
    "u-rnn": MODEL_URNN_LITE,
    "urnn_lite": MODEL_URNN_LITE,
    "u_rnn_lite": MODEL_URNN_LITE,
    "fno": MODEL_FNO2D_HISTORY,
    "fno2d": MODEL_FNO2D_HISTORY,
    "fno2d_history": MODEL_FNO2D_HISTORY,
    "simvp": MODEL_SIMVP_LITE,
    "simvp_lite": MODEL_SIMVP_LITE,
}


def normalize_external_model_type(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key in _EXTERNAL_MODEL_ALIASES:
        return _EXTERNAL_MODEL_ALIASES[key]
    try:
        return normalize_model_type(key)
    except ValueError as error:
        raise ValueError(f"Unknown external model type: {value}") from error


def external_model_display_name(value: str) -> str:
    return EXTERNAL_MODEL_LABELS[normalize_external_model_type(value)]


def _initialize_physical_head(head: nn.Conv2d, use_residual: bool, output_max: float) -> None:
    nn.init.zeros_(head.weight)
    if use_residual:
        nn.init.zeros_(head.bias)
    else:
        initial_depth_m = min(1e-3, output_max * 0.01)
        probability = min(max(initial_depth_m / output_max, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(head.bias, torch.logit(torch.tensor(probability)).item())


def _physical_output(
    raw: torch.Tensor,
    x: torch.Tensor,
    output_max: float,
    use_residual: bool,
    residual_scale: float,
) -> torch.Tensor:
    if use_residual:
        base = x[:, -1, 0:1] * output_max
        return torch.clamp(base + residual_scale * torch.tanh(raw), 0.0, output_max)
    return output_max * torch.sigmoid(raw)


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
        _initialize_physical_head(self.head[-1], self.use_residual, self.output_max)

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
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


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
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


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
        _initialize_physical_head(self.head[-1], self.use_residual, self.output_max)
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
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        merged_channels = input_channels + hidden_channels
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(merged_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(merged_channels, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None:
            state = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), dim=1))).chunk(
            2, dim=1
        )
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=1)))
        return (1.0 - update) * state + update * candidate


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


def _conv_block(input_channels: int, output_channels: int, stride: int = 1) -> nn.Sequential:
    groups = _group_count(output_channels)
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, output_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.GroupNorm(groups, output_channels),
        nn.SiLU(inplace=True),
    )


class ExternalURNNLite(nn.Module):
    """Compact multi-scale ConvGRU U-Net adapted from the U-RNN design."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        output_max: float = 3.5,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_max = float(output_max)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        channels = (hidden_channels, hidden_channels * 2, hidden_channels * 4)
        self.encoder1 = _conv_block(input_channels, channels[0])
        self.encoder2 = _conv_block(channels[0], channels[1], stride=2)
        self.encoder3 = _conv_block(channels[1], channels[2], stride=2)
        self.gru1 = ConvGRUCell(channels[0], channels[0])
        self.gru2 = ConvGRUCell(channels[1], channels[1])
        self.gru3 = ConvGRUCell(channels[2], channels[2])
        self.decoder2 = _conv_block(channels[2] + channels[1], channels[1])
        self.decoder1 = _conv_block(channels[1] + channels[0], channels[0])
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], output_channels, 1),
        )
        _initialize_physical_head(self.head[-1], self.use_residual, self.output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        state1 = state2 = state3 = None
        for time_index in range(x.shape[1]):
            feature1 = self.encoder1(x[:, time_index])
            feature2 = self.encoder2(feature1)
            feature3 = self.encoder3(feature2)
            state1 = self.gru1(feature1, state1)
            state2 = self.gru2(feature2, state2)
            state3 = self.gru3(feature3, state3)
        decoded2 = F.interpolate(
            state3, size=state2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded2 = self.decoder2(torch.cat((decoded2, state2), dim=1))
        decoded1 = F.interpolate(
            decoded2, size=state1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decoder1(torch.cat((decoded1, state1), dim=1))
        raw = self.head(decoded1)
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


class SpectralConv2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, modes: int) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.modes = int(modes)
        scale = 1.0 / max(input_channels * output_channels, 1)
        shape = (input_channels, output_channels, modes, modes, 2)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        original_dtype = x.dtype
        with torch.amp.autocast(x.device.type, enabled=False):
            transformed = torch.fft.rfft2(x.float(), norm="ortho")
            output = torch.zeros(
                batch, self.output_channels, height, width // 2 + 1,
                device=x.device, dtype=transformed.dtype,
            )
            modes_y = min(self.modes, height // 2)
            modes_x = min(self.modes, width // 2 + 1)
            output[:, :, :modes_y, :modes_x] = self._multiply(
                transformed[:, :, :modes_y, :modes_x],
                torch.view_as_complex(
                    self.weight_positive[:, :, :modes_y, :modes_x].contiguous()
                ),
            )
            output[:, :, -modes_y:, :modes_x] = self._multiply(
                transformed[:, :, -modes_y:, :modes_x],
                torch.view_as_complex(
                    self.weight_negative[:, :, :modes_y, :modes_x].contiguous()
                ),
            )
            result = torch.fft.irfft2(output, s=(height, width), norm="ortho")
        return result.to(original_dtype)


class ExternalFNO2DHistory(nn.Module):
    """FNO2D baseline that lifts the complete observed history into one field."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        spectral_layers: int = 3,
        modes: int = 8,
        output_max: float = 3.5,
        max_input_len: int = 12,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_max = float(output_max)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        self.max_input_len = int(max_input_len)
        self.lifting = nn.Conv2d(input_channels * self.max_input_len, hidden_channels, 1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(hidden_channels, hidden_channels, modes) for _ in range(spectral_layers)]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(hidden_channels, hidden_channels, 1) for _ in range(spectral_layers)]
        )
        self.norms = nn.ModuleList(
            [
                nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
                for _ in range(spectral_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, output_channels, 1),
        )
        _initialize_physical_head(self.head[-1], self.use_residual, self.output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.max_input_len:
            raise ValueError(f"Expected {self.max_input_len} input frames, got {x.shape[1]}")
        feature = self.lifting(x.flatten(1, 2))
        for spectral, local, norm in zip(self.spectral_layers, self.local_layers, self.norms):
            feature = F.gelu(norm(spectral(feature) + local(feature)))
        raw = self.head(feature)
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


class SimVPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels)
                for kernel in (3, 5, 7)
            ]
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.project = nn.Conv2d(channels, channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = sum(
            (branch(x) for branch in self.branches), start=torch.zeros_like(x)
        ) / len(self.branches)
        candidate, gate = self.project(self.norm(mixed)).chunk(2, dim=1)
        return x + F.gelu(candidate) * torch.sigmoid(gate)


class ExternalSimVPLite(nn.Module):
    """Compact encoder-translator-decoder baseline adapted from SimVP."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        translator_layers: int = 4,
        output_max: float = 3.5,
        max_input_len: int = 12,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_max = float(output_max)
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        self.max_input_len = int(max_input_len)
        latent_channels = hidden_channels * 2
        translator_channels = hidden_channels * 4
        self.frame_encoder = nn.Sequential(
            _conv_block(input_channels, hidden_channels),
            _conv_block(hidden_channels, latent_channels, stride=2),
        )
        self.temporal_lift = nn.Conv2d(
            latent_channels * self.max_input_len, translator_channels, 1
        )
        self.translator = nn.Sequential(
            *[SimVPBlock(translator_channels) for _ in range(translator_layers)]
        )
        self.temporal_project = nn.Conv2d(translator_channels, latent_channels, 1)
        self.last_frame_skip = nn.Conv2d(input_channels, hidden_channels, 1)
        self.decoder = _conv_block(latent_channels + hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
        _initialize_physical_head(self.head[-1], self.use_residual, self.output_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.max_input_len:
            raise ValueError(f"Expected {self.max_input_len} input frames, got {x.shape[1]}")
        encoded = torch.stack(
            [self.frame_encoder(x[:, index]) for index in range(x.shape[1])], dim=1
        )
        batch, time, channels, height, width = encoded.shape
        feature = self.temporal_lift(encoded.reshape(batch, time * channels, height, width))
        feature = self.temporal_project(self.translator(feature))
        feature = F.interpolate(feature, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.decoder(torch.cat((feature, self.last_frame_skip(x[:, -1])), dim=1))
        raw = self.head(feature)
        return _physical_output(raw, x, self.output_max, self.use_residual, self.residual_scale)


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
    fno_modes: int = 8,
    fno_layers: int = 3,
    simvp_blocks: int = 4,
) -> nn.Module:
    model_type = normalize_external_model_type(model_type)
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
    external_common = dict(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        output_max=output_max,
        use_residual=use_residual,
        residual_scale=residual_scale,
    )
    if model_type == MODEL_URNN_LITE:
        return ExternalURNNLite(**external_common)
    if model_type == MODEL_FNO2D_HISTORY:
        return ExternalFNO2DHistory(
            **external_common,
            spectral_layers=fno_layers,
            modes=fno_modes,
            max_input_len=max_input_len,
        )
    if model_type == MODEL_SIMVP_LITE:
        return ExternalSimVPLite(
            **external_common,
            translator_layers=simvp_blocks,
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
        fno_modes=int(args.get("fno_modes", 8)),
        fno_layers=int(args.get("fno_layers", 3)),
        simvp_blocks=int(args.get("simvp_blocks", 4)),
    )
