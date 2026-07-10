from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {"mae": mae, "rmse": rmse}


def binary_risk_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.20) -> dict:
    pred_bin = pred >= threshold
    target_bin = target >= threshold

    tp = float(np.logical_and(pred_bin, target_bin).sum())
    fp = float(np.logical_and(pred_bin, np.logical_not(target_bin)).sum())
    fn = float(np.logical_and(np.logical_not(pred_bin), target_bin).sum())
    tn = float(np.logical_and(np.logical_not(pred_bin), np.logical_not(target_bin)).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    csi = iou
    far = fp / (tp + fp + eps)
    pod = recall
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    frequency_bias = (tp + fp) / (tp + fn + eps)
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fp * fn) / (hss_denominator + eps)
    total = tp + fp + fn + tn
    random_hits = (tp + fp) * (tp + fn) / (total + eps)
    ets = (tp - random_hits) / (tp + fp + fn - random_hits + eps)
    flood_extent_error = abs((tp + fp) - (tp + fn)) / (total + eps)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall_pod": float(pod),
        "f1": float(f1),
        "iou": float(iou),
        "csi": float(csi),
        "csi_iou_equivalent": True,
        "far": float(far),
        "accuracy": float(acc),
        "frequency_bias": float(frequency_bias),
        "hss": float(hss),
        "ets": float(ets),
        "flood_extent_error": float(flood_extent_error),
    }


def all_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.20) -> dict:
    m = regression_metrics(pred, target)
    m.update(binary_risk_metrics(pred, target, threshold=threshold))
    m["peak_depth_error"] = float(abs(float(np.max(pred)) - float(np.max(target))))
    return m
