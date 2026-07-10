from __future__ import annotations

from pathlib import Path

import numpy as np

from src.align_modalities import aggregate_social, align_event
from src.generate_synthetic import generate_event


def test_social_mask_distinguishes_zero_depth_report_from_no_report() -> None:
    empty = aggregate_social(
        *(np.array([], dtype=dtype) for dtype in (np.int32, np.int32, np.int32, np.float32, np.float32)),
        anchor=2,
        h=9,
        w=9,
        window=2,
        lambda_soc=0.1,
        mode="realtime",
        radius=2,
        sigma=1.0,
    )
    assert np.count_nonzero(empty["observation_mask"]) == 0

    observed_zero = aggregate_social(
        np.array([2], dtype=np.int32),
        np.array([4], dtype=np.int32),
        np.array([4], dtype=np.int32),
        np.array([0.0], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        anchor=2,
        h=9,
        w=9,
        window=2,
        lambda_soc=0.1,
        mode="realtime",
        radius=2,
        sigma=1.0,
    )
    assert np.count_nonzero(observed_zero["observation_mask"]) == 25
    assert np.all(observed_zero["value_map"] == 0.0)
    assert observed_zero["miss_soc"] == 0


def test_realtime_social_aggregation_excludes_future_reports() -> None:
    result = aggregate_social(
        np.array([3], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([0.5], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        anchor=2,
        h=5,
        w=5,
        window=2,
        lambda_soc=0.1,
        mode="realtime",
    )
    assert result["miss_soc"] == 1
    assert result["latest_observation_time"] == -1


def test_default_alignment_keeps_gis_value_static_and_legacy_decays(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.npz"
    none_path = tmp_path / "none.npz"
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(raw_path, **generate_event(0, t=10, h=8, w=8, seed=11))

    common = dict(
        path=raw_path,
        mode="realtime",
        social_window=3,
        lambda_sat=0.015,
        lambda_gis=0.1,
        lambda_soc=0.1,
    )
    align_event(out_path=none_path, value_decay_mode="none", **common)
    align_event(out_path=legacy_path, value_decay_mode="legacy", **common)

    with np.load(none_path) as aligned:
        assert np.allclose(aligned["gis_risk"][0], aligned["gis_risk"][-1])
        assert aligned["dt_gis"][-1] > aligned["dt_gis"][0]
        assert aligned["gis_observation_time"].max() <= aligned["anchors"].max()
    with np.load(legacy_path) as aligned:
        assert float(aligned["gis_risk"][-1].mean()) < float(aligned["gis_risk"][0].mean())
