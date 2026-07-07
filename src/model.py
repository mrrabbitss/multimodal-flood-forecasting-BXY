from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        b, _, h, w = x.shape
        if state is None:
            h_t = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            c_t = torch.zeros_like(h_t)
        else:
            h_t, c_t = state
        combined = torch.cat([x, h_t], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecastNet(nn.Module):
    """Lightweight Conv-LSTM forecaster.

    Input:  [B, T, C, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 24,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        states = [None for _ in range(self.num_layers)]
        for i in range(t):
            feat = self.encoder(x[:, i])
            h_i, c_i = self.cell(feat, states[0])
            states[0] = (h_i, c_i)
            feat = h_i
            for layer_idx, cell in enumerate(self.extra_cells, start=1):
                h_i, c_i = cell(feat, states[layer_idx])
                states[layer_idx] = (h_i, c_i)
                feat = h_i
        h_last, _ = states[-1]
        raw = self.head(h_last)
        if self.use_residual and 0 <= self.fused_channel < c:
            base = x[:, -1, self.fused_channel : self.fused_channel + 1]
            return torch.clamp(base + self.residual_scale * torch.tanh(raw), 0.0, self.output_max)
        return self.output_max * torch.sigmoid(raw)
