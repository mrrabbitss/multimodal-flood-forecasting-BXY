from pathlib import Path

import numpy as np

from src.experiments.splits import build_event_split_manifest, save_event_split_manifest


def make_event(path: Path) -> None:
    np.savez_compressed(path, gt_depth=np.zeros((20, 4, 4), dtype=np.float32))


def test_event_split_manifest_is_disjoint_and_reproducible(tmp_path: Path) -> None:
    for index in range(10):
        make_event(tmp_path / f"event_{index:03d}.npz")
    first = build_event_split_manifest(tmp_path, seed=44, shuffle=True)
    second = build_event_split_manifest(tmp_path, seed=44, shuffle=True)
    assert first == second
    partitions = first["partitions"]
    ids = [{row["event_id"] for row in partitions[name]} for name in ("train", "val", "test")]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert sum(map(len, ids)) == 10
    assert first["leakage_check"]["window_overlap_across_partitions"] is False


def test_split_manifest_is_saved_under_artifacts(tmp_path: Path) -> None:
    fused = tmp_path / "fused"
    fused.mkdir()
    for index in range(6):
        make_event(fused / f"event_{index:03d}.npz")
    _, path = save_event_split_manifest(fused, tmp_path / "run", seed=7)
    assert path == tmp_path / "run" / "artifacts" / "splits" / "split_seed_7.json"
    assert path.exists()

