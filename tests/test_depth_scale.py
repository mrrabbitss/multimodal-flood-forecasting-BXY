from __future__ import annotations

import numpy as np
import torch

from src.data.schemas import DEFAULT_DEPTH_SCALE, LEGACY_DEPTH_SCALE, depth_scale_from_checkpoint
from src.generate_synthetic import generate_event
from src.model import ConvLSTMForecastNet


def test_generated_labels_respect_depth_scale() -> None:
    event = generate_event(0, t=12, h=8, w=8, seed=7, depth_scale=DEFAULT_DEPTH_SCALE)
    assert float(event["gt_depth"].min()) >= DEFAULT_DEPTH_SCALE.min_value
    assert float(event["gt_depth"].max()) <= DEFAULT_DEPTH_SCALE.max_value
    assert np.isclose(float(event["depth_max"]), DEFAULT_DEPTH_SCALE.max_value)


def test_sigmoid_and_residual_outputs_respect_depth_scale() -> None:
    x = torch.rand(2, 3, 13, 8, 8)
    x[:, -1, 4] = 1.18
    for use_residual in (False, True):
        model = ConvLSTMForecastNet(
            input_channels=13,
            hidden_channels=4,
            output_max=DEFAULT_DEPTH_SCALE.max_value,
            use_residual=use_residual,
        )
        output = model(x)
        assert output.shape == (2, 1, 8, 8)
        assert float(output.detach().min()) >= DEFAULT_DEPTH_SCALE.min_value
        assert float(output.detach().max()) <= DEFAULT_DEPTH_SCALE.max_value + 1e-6


def test_checkpoint_depth_scale_prefers_saved_metadata_and_supports_legacy() -> None:
    current = depth_scale_from_checkpoint({"depth_scale": DEFAULT_DEPTH_SCALE.to_dict(), "output_max": 1.0})
    legacy_explicit = depth_scale_from_checkpoint({"output_max": 1.0})
    legacy_implicit = depth_scale_from_checkpoint({})
    assert current == DEFAULT_DEPTH_SCALE
    assert legacy_explicit == LEGACY_DEPTH_SCALE
    assert legacy_implicit == LEGACY_DEPTH_SCALE
