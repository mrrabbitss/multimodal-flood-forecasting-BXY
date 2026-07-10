from __future__ import annotations

import numpy as np

from src.fuse_dynamic_gate import softmax_reliability
from src.metrics import binary_risk_metrics


def test_csi_is_the_iou_alias_for_binary_flood_masks() -> None:
    pred = np.array([[1, 1], [0, 0]], dtype=np.float32)
    target = np.array([[1, 0], [1, 0]], dtype=np.float32)
    metrics = binary_risk_metrics(pred, target, threshold=0.5)
    assert np.isclose(metrics["csi"], 1.0 / 3.0)
    assert metrics["csi"] == metrics["iou"]
    assert metrics["csi_iou_equivalent"] is True
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1 and metrics["tn"] == 1


def test_unavailable_modalities_receive_exactly_zero_weight() -> None:
    reliability = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)[:, None, None]
    availability = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)[:, None, None]
    weights = softmax_reliability(reliability, availability=availability)
    assert weights[1, 0, 0] == 0.0
    assert weights[3, 0, 0] == 0.0
    assert np.isclose(weights[:, 0, 0].sum(), 1.0)
