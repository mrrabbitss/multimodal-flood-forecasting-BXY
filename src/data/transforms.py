from __future__ import annotations

from typing import Mapping

import numpy as np


RAIN_FEATURE_VERSION = "causal_rolling_v1"
RAIN_FEATURE_NAMES: tuple[str, ...] = (
    "rain_current",
    "rain_accum_3",
    "rain_accum_6",
    "rain_accum_12",
    "rain_max_recent_6",
    "rain_trend_3",
)


def causal_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("rolling window must be >= 1")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"rain series must be one-dimensional, got {values.shape}")
    cumulative = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)])
    result = np.empty_like(values, dtype=np.float32)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = cumulative[index + 1] - cumulative[start]
    return result


def causal_rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("rolling window must be >= 1")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"rain series must be one-dimensional, got {values.shape}")
    return np.asarray(
        [values[max(0, index - window + 1) : index + 1].max() for index in range(values.size)],
        dtype=np.float32,
    )


def causal_trend(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("trend window must be >= 2")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"rain series must be one-dimensional, got {values.shape}")
    result = np.zeros_like(values, dtype=np.float32)
    for index in range(1, values.size):
        start = max(0, index - window + 1)
        steps = index - start
        result[index] = (values[index] - values[start]) / max(steps, 1)
    return result


def derive_rain_features(rain: np.ndarray) -> dict[str, np.ndarray]:
    rain = np.asarray(rain, dtype=np.float32)
    if rain.ndim != 1:
        raise ValueError(f"rain series must be one-dimensional, got {rain.shape}")
    if not np.all(np.isfinite(rain)):
        raise ValueError("rain series contains NaN or Inf")
    return {
        "rain_current": rain.copy(),
        "rain_accum_3": causal_rolling_sum(rain, 3),
        "rain_accum_6": causal_rolling_sum(rain, 6),
        "rain_accum_12": causal_rolling_sum(rain, 12),
        "rain_max_recent_6": causal_rolling_max(rain, 6),
        "rain_trend_3": causal_trend(rain, 3),
    }


def rain_features_from_mapping(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "rain" not in data and "rain_current" not in data:
        raise KeyError("Rain channels require 'rain' or materialized 'rain_current' in the event artifact")
    base = np.asarray(data["rain"] if "rain" in data else data["rain_current"], dtype=np.float32)
    derived = derive_rain_features(base)
    for name in RAIN_FEATURE_NAMES:
        if name in data:
            materialized = np.asarray(data[name], dtype=np.float32)
            if materialized.shape != base.shape:
                raise ValueError(f"Rain feature {name!r} has shape {materialized.shape}, expected {base.shape}")
            derived[name] = materialized
    return derived
