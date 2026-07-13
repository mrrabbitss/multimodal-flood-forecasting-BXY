from __future__ import annotations

from pathlib import Path

import numpy as np

from src.align_modalities import align_event
from src.dataset import (
    BATCH1_CHANNEL_NAMES,
    CHANNEL_NAMES,
    LEGACY_CHANNEL_NAMES,
    FloodSequenceDataset,
    channel_names_for_data,
    channel_names_from_checkpoint,
    inspect_dataset_schema,
    resolve_channel_names,
    validate_checkpoint_data_schema,
)
from src.fuse_dynamic_gate import fuse_event
from src.generate_synthetic import generate_event


def _make_fused_event(root: Path) -> Path:
    raw = root / "raw.npz"
    aligned = root / "aligned.npz"
    fused_dir = root / "fused"
    fused_dir.mkdir()
    fused = fused_dir / "event_0000.npz"
    np.savez_compressed(raw, **generate_event(0, t=8, h=8, w=8, seed=17))
    align_event(raw, aligned, "realtime", 3, 0.015, 0.002, 0.1)
    fuse_event(aligned, fused, 0.2, 0.4)
    return fused_dir


def test_default_and_legacy_channel_schemas_are_both_loadable(tmp_path: Path) -> None:
    fused_dir = _make_fused_event(tmp_path)
    current = FloodSequenceDataset(fused_dir, [0], input_len=2, lead_time=1)
    legacy = FloodSequenceDataset(fused_dir, [0], input_len=2, lead_time=1, channel_names="legacy")
    x_current, _ = current[0]
    x_legacy, _ = legacy[0]
    assert x_current.shape[1] == len(CHANNEL_NAMES) == 23
    assert x_legacy.shape[1] == len(LEGACY_CHANNEL_NAMES) == 13
    assert channel_names_for_data(fused_dir) == CHANNEL_NAMES


def test_arbitrary_channel_order_and_checkpoint_compatibility(tmp_path: Path) -> None:
    fused_dir = _make_fused_event(tmp_path)
    names = ("q_soc", "meteo", "soc_observation_mask", "fused_depth")
    dataset = FloodSequenceDataset(fused_dir, [0], input_len=2, lead_time=1, channel_names=names)
    x, _ = dataset[0]
    assert x.shape[1] == len(names)
    assert channel_names_from_checkpoint({"input_channels": 13}) == LEGACY_CHANNEL_NAMES
    assert channel_names_from_checkpoint({"input_channels": 19}) == BATCH1_CHANNEL_NAMES
    assert channel_names_from_checkpoint({"input_channels": len(names), "channel_names": list(names)}) == names


def test_checkpoint_data_schema_validates_channel_order_and_rain_version(tmp_path: Path) -> None:
    fused_dir = _make_fused_event(tmp_path)
    names = ("meteo", "rain_current", "rain_accum_6")
    schema = inspect_dataset_schema(fused_dir, names)
    checkpoint = {"input_channels": len(names), "channel_names": list(names), "data_schema": schema}
    current = validate_checkpoint_data_schema(checkpoint, fused_dir)
    assert current["checkpoint_schema_compatibility"] == "validated"

    incompatible = {**checkpoint, "data_schema": {**schema, "rain_feature_version": "future_v9"}}
    with np.testing.assert_raises_regex(ValueError, "rain feature version"):
        validate_checkpoint_data_schema(incompatible, fused_dir)


def test_batch3_modality_channel_sets_are_named_and_nonempty() -> None:
    full = resolve_channel_names("full")
    no_satellite = resolve_channel_names("without_satellite")
    no_social = resolve_channel_names("without_social")
    no_metadata = resolve_channel_names("without_meta")
    assert "satellite" in full and "satellite" not in no_satellite
    assert "social" in full and "social" not in no_social
    assert "miss_sat" not in no_metadata and "fused_depth" in no_metadata
    assert len(no_satellite) < len(full)
