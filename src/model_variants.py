from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model import ConvLSTMCell, ConvLSTMForecastNet


MODEL_CONVLSTM = "convlstm"
MODEL_CONVLSTM_ATTENTION = "convlstm_attention"
MODEL_CNN_TEMPORAL_TRANSFORMER = "cnn_temporal_transformer"


def normalize_model_type(model_type: str | None) -> str:
    value = (model_type or MODEL_CONVLSTM).strip().lower().replace("-", "_")
    aliases = {
        "conv_lstm": MODEL_CONVLSTM,
        "conv_lstm_attention": MODEL_CONVLSTM_ATTENTION,
        "convlstm_attn": MODEL_CONVLSTM_ATTENTION,
        "conv_lstm_attn": MODEL_CONVLSTM_ATTENTION,
        "cnn_transformer": MODEL_CNN_TEMPORAL_TRANSFORMER,
        "temporal_transformer": MODEL_CNN_TEMPORAL_TRANSFORMER,
    }
    return aliases.get(value, value)


def model_display_name(model_type: str) -> str:
    names = {
        MODEL_CONVLSTM: "Conv-LSTM",
        MODEL_CONVLSTM_ATTENTION: "Conv-LSTM + Attention",
        MODEL_CNN_TEMPORAL_TRANSFORMER: "CNN-Temporal Transformer",
    }
    return names.get(normalize_model_type(model_type), model_type)


class TemporalSpatialAttention(nn.Module):
    """Per-pixel temporal attention over Conv-LSTM hidden states."""

    def __init__(self, hidden_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        mid_channels = max(4, hidden_channels // 2)
        self.score = nn.Sequential(
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 5:
            raise ValueError(f"Expected states [B,T,C,H,W], got {tuple(states.shape)}")
        b, t, c, h, w = states.shape
        flat_states = states.reshape(b * t, c, h, w)
        scores = self.score(flat_states).reshape(b, t, 1, h, w)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * states, dim=1)
        return context, weights


class ConvLSTMWithAttentionForecastNet(nn.Module):
    """Conv-LSTM forecaster with learned temporal attention over all hidden states."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_max: float = 1.0,
        residual_scale: float = 0.35,
        use_residual: bool = False,
        fused_channel: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = int(num_layers)
        self.output_max = float(output_max)
        self.residual_scale = float(residual_scale)
        self.use_residual = bool(use_residual)
        self.fused_channel = int(fused_channel)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.cell = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=kernel_size)
        self.extra_cells = nn.ModuleList(
            ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=kernel_size)
            for _ in range(self.num_layers - 1)
        )
        self.attention = TemporalSpatialAttention(hidden_channels, dropout=attention_dropout)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {tuple(x.shape)}")
        _, t, c, _, _ = x.shape
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.num_layers)]
        hidden_by_time: list[torch.Tensor] = []
        for i in range(t):
            feat = self.encoder(x[:, i])
            h_i, c_i = self.cell(feat, states[0])
            states[0] = (h_i, c_i)
            feat = h_i
            for layer_idx, cell in enumerate(self.extra_cells, start=1):
                h_i, c_i = cell(feat, states[layer_idx])
                states[layer_idx] = (h_i, c_i)
                feat = h_i
            hidden_by_time.append(feat)
        context, _ = self.attention(torch.stack(hidden_by_time, dim=1))
        raw = self.head(context)
        if self.use_residual and 0 <= self.fused_channel < c:
            base = x[:, -1, self.fused_channel : self.fused_channel + 1]
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)


class CNNTemporalTransformerForecastNet(nn.Module):
    """CNN frame encoder followed by a per-pixel temporal Transformer."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_mult: float = 4.0,
        dropout: float = 0.0,
        output_max: float = 1.0,
        residual_scale: float = 0.35,
        use_residual: bool = False,
        fused_channel: int = 4,
        max_input_len: int = 128,
    ) -> None:
        super().__init__()
        if hidden_channels % transformer_heads != 0:
            raise ValueError("hidden_channels must be divisible by transformer_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_max = float(output_max)
        self.residual_scale = float(residual_scale)
        self.use_residual = bool(use_residual)
        self.fused_channel = int(fused_channel)
        self.max_input_len = int(max_input_len)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=max(hidden_channels, int(hidden_channels * transformer_ffn_mult)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_input_len, hidden_channels))
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t > self.max_input_len:
            raise ValueError(f"input length {t} exceeds max_input_len={self.max_input_len}")
        encoded = [self.frame_encoder(x[:, i]) for i in range(t)]
        seq = torch.stack(encoded, dim=1)
        tokens = seq.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, -1)
        tokens = tokens + self.pos_embedding[:, :t, :]
        temporal = self.temporal_encoder(tokens)
        last = temporal[:, -1].reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        raw = self.head(last)
        if self.use_residual and 0 <= self.fused_channel < c:
            base = x[:, -1, self.fused_channel : self.fused_channel + 1]
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)


def build_forecast_model(
    model_type: str,
    input_channels: int,
    hidden_channels: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    output_max: float = 1.0,
    residual_scale: float = 0.35,
    use_residual: bool = False,
    attention_dropout: float = 0.0,
    transformer_heads: int = 4,
    transformer_ffn_mult: float = 4.0,
    max_input_len: int = 128,
) -> nn.Module:
    model_type = normalize_model_type(model_type)
    if model_type == MODEL_CONVLSTM:
        return ConvLSTMForecastNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            output_max=output_max,
            residual_scale=residual_scale,
            use_residual=use_residual,
        )
    if model_type == MODEL_CONVLSTM_ATTENTION:
        return ConvLSTMWithAttentionForecastNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            output_max=output_max,
            residual_scale=residual_scale,
            use_residual=use_residual,
        )
    if model_type == MODEL_CNN_TEMPORAL_TRANSFORMER:
        return CNNTemporalTransformerForecastNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_mult=transformer_ffn_mult,
            dropout=dropout,
            output_max=output_max,
            residual_scale=residual_scale,
            use_residual=use_residual,
            max_input_len=max_input_len,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def checkpoint_model_type(checkpoint: dict[str, Any]) -> str:
    return normalize_model_type(checkpoint.get("model_type", MODEL_CONVLSTM))


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    input_len = int(checkpoint.get("input_len", 12))
    return build_forecast_model(
        model_type=checkpoint_model_type(checkpoint),
        input_channels=int(checkpoint["input_channels"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.0)),
        output_max=float(checkpoint.get("output_max", 1.0)),
        residual_scale=float(checkpoint.get("residual_scale", 0.35)),
        use_residual=bool(checkpoint.get("use_residual", False)),
        attention_dropout=float(checkpoint.get("attention_dropout", checkpoint.get("dropout", 0.0))),
        transformer_heads=int(checkpoint.get("transformer_heads", 4)),
        transformer_ffn_mult=float(checkpoint.get("transformer_ffn_mult", 4.0)),
        max_input_len=max(int(checkpoint.get("max_input_len", 128)), input_len),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
