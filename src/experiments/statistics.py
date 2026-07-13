from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


LOWER_IS_BETTER = {"mae", "rmse", "far", "latency_ms", "peak_cuda_mb"}


def summarize_values(values: Sequence[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty value sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "per_seed": [float(value) for value in array],
    }


def summarize_multiseed(
    rows: Iterable[Mapping[str, object]],
    group_key: str = "variant",
    metrics: Sequence[str] = ("mae", "rmse", "csi", "f1", "far", "recall_pod"),
) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    summaries = []
    for group, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: int(row.get("seed", 0)))
        item: dict[str, object] = {group_key: group, "seeds": [int(row["seed"]) for row in ordered]}
        for metric in metrics:
            values = [float(row[metric]) for row in ordered if metric in row]
            if values:
                item[metric] = summarize_values(values)
        summaries.append(item)
    return summaries


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    metric: str,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 2026,
    tie_tolerance: float = 1e-8,
) -> dict[str, object]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.shape != candidate_array.shape or baseline_array.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be one-dimensional arrays with equal shape")
    if baseline_array.size < 2:
        raise ValueError("Paired bootstrap requires at least two paired observations")
    raw_difference = candidate_array - baseline_array
    oriented_difference = -raw_difference if metric in LOWER_IS_BETTER else raw_difference
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, oriented_difference.size, size=(n_resamples, oriented_difference.size))
    bootstrap_means = oriented_difference[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    wins = int(np.sum(oriented_difference > tie_tolerance))
    losses = int(np.sum(oriented_difference < -tie_tolerance))
    ties = int(oriented_difference.size - wins - losses)
    return {
        "metric": metric,
        "difference_definition": "candidate improvement over baseline; positive is better",
        "mean_improvement": float(oriented_difference.mean()),
        "confidence": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_pairs": int(oriented_difference.size),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }

