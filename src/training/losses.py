from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    loss_threshold: float = 0.20
    high_risk_weight: float = 2.5
    mse_weight: float = 0.25
    class_threshold: float = 0.28
    class_temperature: float = 0.04
    bce_weight: float = 0.0
    dice_weight: float = 0.0
    focal_weight: float = 0.0
    temporal_weight: float = 0.0
    edge_weight: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "LossConfig":
        value = checkpoint.get("loss_config")
        if isinstance(value, Mapping):
            known = {field: value[field] for field in cls.__dataclass_fields__ if field in value}
            return cls(**known)
        return cls(
            loss_threshold=float(checkpoint.get("loss_threshold", 0.20)),
            class_threshold=float(checkpoint.get("class_threshold", checkpoint.get("threshold", 0.28))),
            class_temperature=float(checkpoint.get("class_temperature", 0.04)),
            bce_weight=float(checkpoint.get("bce_loss_weight", 0.0)),
            dice_weight=float(checkpoint.get("dice_loss_weight", 0.0)),
            focal_weight=float(checkpoint.get("focal_loss_weight", 0.0)),
        )


@dataclass
class LossBreakdown:
    total: torch.Tensor
    depth: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    focal: torch.Tensor
    temporal: torch.Tensor
    edge: torch.Tensor

    def detached_values(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().item()),
            "loss_depth": float(self.depth.detach().item()),
            "loss_bce": float(self.bce.detach().item()),
            "loss_dice": float(self.dice.detach().item()),
            "loss_focal": float(self.focal.detach().item()),
            "loss_temporal": float(self.temporal.detach().item()),
            "loss_edge": float(self.edge.detach().item()),
        }


def weighted_depth_loss(pred: torch.Tensor, target: torch.Tensor, config: LossConfig) -> torch.Tensor:
    weight = 1.0 + config.high_risk_weight * (target >= config.loss_threshold).float()
    mae = torch.mean(weight * torch.abs(pred - target))
    mse = torch.mean(weight * (pred - target) ** 2)
    return mae + config.mse_weight * mse


def mask_loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_bin = (target >= threshold).float()
    logits = (pred - threshold) / max(float(temperature), 1e-6)
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target_bin)

    cross_entropy = F.binary_cross_entropy_with_logits(logits, target_bin, reduction="none")
    pt = torch.where(target_bin > 0.5, probability, 1.0 - probability)
    focal = ((1.0 - pt).pow(2.0) * cross_entropy).mean()

    dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(probability * target_bin, dim=dims)
    denominator = torch.sum(probability + target_bin, dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return bce, dice, focal


def build_loss(pred: torch.Tensor, target: torch.Tensor, config: LossConfig) -> LossBreakdown:
    depth = weighted_depth_loss(pred, target, config)
    bce, dice, focal = mask_loss_terms(
        pred,
        target,
        threshold=config.class_threshold,
        temperature=config.class_temperature,
    )
    zero = pred.new_zeros(())
    temporal = zero
    if config.temporal_weight > 0.0 and pred.shape[1] > 1:
        temporal = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    edge = zero
    if config.edge_weight > 0.0:
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        edge = F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)
    total = (
        depth
        + config.bce_weight * bce
        + config.dice_weight * dice
        + config.focal_weight * focal
        + config.temporal_weight * temporal
        + config.edge_weight * edge
    )
    return LossBreakdown(total=total, depth=depth, bce=bce, dice=dice, focal=focal, temporal=temporal, edge=edge)
