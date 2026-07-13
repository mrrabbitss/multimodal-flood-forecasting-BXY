from pathlib import Path

import numpy as np

from src.align_modalities import align_event
from src.batch4_dataset import MultiHorizonFloodDataset, normalize_lead_times
from src.fuse_dynamic_gate import fuse_event
from src.generate_synthetic import generate_event


def make_fused_event(root: Path) -> Path:
    raw = root / "raw.npz"
    aligned = root / "aligned.npz"
    fused_dir = root / "fused"
    fused_dir.mkdir()
    fused = fused_dir / "event_0000.npz"
    np.savez_compressed(raw, **generate_event(0, t=32, h=8, w=8, seed=17))
    align_event(raw, aligned, "realtime", 3, 0.015, 0.002, 0.1)
    fuse_event(aligned, fused, 0.2, 0.4)
    return fused_dir


def test_multi_horizon_dataset_returns_aligned_targets(tmp_path: Path) -> None:
    fused_dir = make_fused_event(tmp_path)
    dataset = MultiHorizonFloodDataset(
        fused_dir, [0], input_len=4, lead_times=(1, 3, 6), channel_names="legacy_rain_accum"
    )
    x, target = dataset[0]
    assert x.shape == (4, 17, 8, 8)
    assert target.shape == (3, 8, 8)
    with np.load(fused_dir / "event_0000.npz") as artifact:
        assert np.allclose(target.numpy()[0], artifact["gt_depth"][4])
        assert np.allclose(target.numpy()[2], artifact["gt_depth"][9])


def test_lead_times_are_sorted_and_validated() -> None:
    assert normalize_lead_times((24, 1, 6)) == (1, 6, 24)
    for invalid in ((), (0, 1), (3, 3)):
        try:
            normalize_lead_times(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid lead times to fail: {invalid}")
