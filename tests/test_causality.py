from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.validation import validate_realtime_causality


def _write_aligned(path: Path, sat_times: np.ndarray, mode: str = "realtime") -> None:
    anchors = np.arange(len(sat_times), dtype=np.int32)
    np.savez_compressed(
        path,
        mode=np.array(mode),
        anchors=anchors,
        sat_observation_time=sat_times.astype(np.int32),
        gis_observation_time=np.zeros_like(anchors),
        soc_latest_observation_time=np.full_like(anchors, -1),
    )


def test_valid_realtime_timestamps_pass(tmp_path: Path) -> None:
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    _write_aligned(aligned / "event_0000.npz", np.arange(8, dtype=np.int32))
    report = validate_realtime_causality(aligned_dir=aligned, input_len=3, lead_time=2)
    assert report["valid"]
    assert report["violations"] == []


def test_future_observation_reports_event_anchor_field_and_timestamp(tmp_path: Path) -> None:
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    selected = np.arange(8, dtype=np.int32)
    selected[3] = 4
    _write_aligned(aligned / "event_0007.npz", selected)
    report = validate_realtime_causality(aligned_dir=aligned, input_len=3, lead_time=2)
    assert not report["valid"]
    violation = report["violations"][0]
    assert violation["event"] == "event_0007.npz"
    assert violation["anchor"] == 3
    assert violation["field"] == "sat_observation_time"
    assert violation["observed_timestamp"] == 4


def test_missing_timestamp_audit_fields_fail_strict_validation(tmp_path: Path) -> None:
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.savez_compressed(
        aligned / "event_0001.npz",
        mode=np.array("realtime"),
        anchors=np.arange(8, dtype=np.int32),
    )
    report = validate_realtime_causality(aligned_dir=aligned, input_len=3, lead_time=2)
    assert not report["valid"]
    assert {item["field"] for item in report["violations"]} == {
        "sat_observation_time",
        "gis_observation_time",
        "soc_latest_observation_time",
    }
