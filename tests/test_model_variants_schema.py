from __future__ import annotations

import torch

from src.dataset import CHANNEL_NAMES
from src.model_variants import build_forecast_model


def test_all_model_variants_accept_current_rain_schema() -> None:
    x = torch.rand(1, 3, len(CHANNEL_NAMES), 8, 8)
    fused_channel = CHANNEL_NAMES.index("fused_depth")
    for model_type in ("convlstm", "convlstm_attention", "cnn_temporal_transformer"):
        model = build_forecast_model(
            model_type=model_type,
            input_channels=len(CHANNEL_NAMES),
            hidden_channels=4,
            num_layers=1,
            transformer_heads=2,
            max_input_len=3,
            fused_channel=fused_channel,
        )
        output = model(x)
        assert output.shape == (1, 1, 8, 8)
        assert torch.isfinite(output).all()
