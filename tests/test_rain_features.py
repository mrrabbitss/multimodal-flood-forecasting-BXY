from __future__ import annotations

from pathlib import Path

import numpy as np

from src.align_modalities import align_event
from src.data.transforms import RAIN_FEATURE_NAMES, derive_rain_features
from src.data.validation import validate_realtime_causality
from src.dataset import (
    LEGACY_RAIN_ACCUM_CHANNEL_NAMES,
    LEGACY_RAIN_CURRENT_CHANNEL_NAMES,
    inspect_dataset_schema,
    resolve_channel_names,
    validate_channel_availability,
)
from src.fuse_dynamic_gate import fuse_event
from src.generate_synthetic import generate_event


def test_rain_features_use_only_current_and_past_values() -> None:
    rain = np.array([0.0, 0.2, 0.5, 0.1, 0.8], dtype=np.float32)
    features = derive_rain_features(rain)
    assert np.allclose(features["rain_accum_3"], [0.0, 0.2, 0.7, 0.8, 1.4])
    assert np.allclose(features["rain_accum_6"], np.cumsum(rain))
    assert np.allclose(features["rain_max_recent_6"], [0.0, 0.2, 0.5, 0.5, 0.8])

    changed_future = rain.copy()
    changed_future[-1] = 0.0
    changed = derive_rain_features(changed_future)
    for name in RAIN_FEATURE_NAMES:
        assert np.allclose(features[name][:-1], changed[name][:-1]), name


def test_named_ablation_channel_sets_are_stable() -> None:
    assert resolve_channel_names("legacy_rain_current") == LEGACY_RAIN_CURRENT_CHANNEL_NAMES
    assert resolve_channel_names("legacy_rain_accum") == LEGACY_RAIN_ACCUM_CHANNEL_NAMES
    assert len(LEGACY_RAIN_CURRENT_CHANNEL_NAMES) == 14
    assert len(LEGACY_RAIN_ACCUM_CHANNEL_NAMES) == 17


def test_missing_rain_source_has_clear_error() -> None:
    with np.testing.assert_raises_regex(KeyError, "rain or rain_current"):
        validate_channel_availability({"gt_depth": np.zeros((3, 2, 2), dtype=np.float32)}, ("rain_current",))


def test_generated_rain_schema_survives_alignment_and_fusion(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    aligned = tmp_path / "aligned.npz"
    fused_dir = tmp_path / "fused"
    fused_dir.mkdir()
    fused = fused_dir / "event_0000.npz"
    np.savez_compressed(raw, **generate_event(0, t=12, h=8, w=8, seed=31))
    align_event(raw, aligned, "realtime", 3, 0.015, 0.002, 0.1)
    fuse_event(aligned, fused, 0.2, 0.4)

    with np.load(fused) as artifact:
        for name in RAIN_FEATURE_NAMES:
            assert name in artifact.files
        assert str(artifact["rain_feature_version"].item()) == "causal_rolling_v1"
    schema = inspect_dataset_schema(fused_dir, "default")
    assert schema["input_channels"] == 23
    assert schema["rain_features_materialized"] is True


def test_causality_validator_rejects_corrupted_materialized_rain(tmp_path: Path) -> None:
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()
    anchors = np.arange(8, dtype=np.int32)
    rain = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    features = derive_rain_features(rain)
    features["rain_accum_3"][2] += 0.5
    np.savez_compressed(
        aligned_dir / "event_0000.npz",
        mode=np.array("realtime"),
        anchors=anchors,
        rain=rain,
        sat_observation_time=anchors,
        gis_observation_time=np.zeros_like(anchors),
        soc_latest_observation_time=np.full_like(anchors, -1),
        **features,
    )
    report = validate_realtime_causality(aligned_dir=aligned_dir, input_len=3, lead_time=2)
    assert not report["valid"]
    assert any(item["field"] == "rain_accum_3" for item in report["violations"])
