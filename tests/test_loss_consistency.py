from __future__ import annotations

import torch

from src.train import combined_depth_mask_loss, weighted_depth_loss
from src.training.losses import LossConfig, build_loss


def test_zero_auxiliary_weights_match_original_depth_loss() -> None:
    pred = torch.tensor([[[[0.1, 0.4], [0.7, 0.2]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0, 0.3], [0.8, 0.1]]]], dtype=torch.float32)
    config = LossConfig(loss_threshold=0.2, class_threshold=0.28)
    expected = weighted_depth_loss(pred, target, high_threshold=0.2)
    actual = build_loss(pred, target, config)
    assert torch.allclose(actual.total, expected)
    assert torch.allclose(actual.total, actual.depth)


def test_auxiliary_loss_is_identical_through_legacy_and_shared_entrypoints() -> None:
    pred = torch.rand(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    config = LossConfig(
        loss_threshold=0.2,
        class_threshold=0.28,
        class_temperature=0.04,
        bce_weight=0.2,
        dice_weight=0.3,
        focal_weight=0.1,
    )
    shared = build_loss(pred, target, config)
    compatible = combined_depth_mask_loss(pred, target, 0.2, 0.28, 0.04, 0.2, 0.3, 0.1)
    assert torch.allclose(shared.total, compatible)
    assert shared.bce.item() > 0
    assert shared.dice.item() >= 0
