import torch

from src.training.losses import LossConfig, build_loss


def test_multi_horizon_temporal_and_edge_losses_are_active() -> None:
    target = torch.zeros(1, 3, 5, 5)
    target[:, 1] = 0.2
    target[:, 2, 2:, 2:] = 0.8
    prediction = target.clone()
    prediction[:, 1] += 0.1
    prediction[:, 2, :2, :2] += 0.2
    config = LossConfig(temporal_weight=0.5, edge_weight=0.5)
    result = build_loss(prediction, target, config)
    assert result.temporal.item() > 0.0
    assert result.edge.item() > 0.0
    assert result.total.item() > result.depth.item()


def test_legacy_single_horizon_defaults_keep_auxiliary_losses_zero() -> None:
    prediction = torch.rand(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    result = build_loss(prediction, target, LossConfig())
    assert result.temporal.item() == 0.0
    assert result.edge.item() == 0.0
