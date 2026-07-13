import numpy as np

from src.compare_baselines import linear_extrapolation


def test_linear_extrapolation_uses_recent_fused_trend_and_clips() -> None:
    x = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    x[:, -2, 1] = 0.2
    x[:, -1, 1] = 0.4
    prediction = linear_extrapolation(x, fused_channel=1, lead_time=3, output_max=0.8)
    assert prediction.shape == (1, 1, 2, 2)
    assert np.allclose(prediction, 0.8)

