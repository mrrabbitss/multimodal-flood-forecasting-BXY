from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dataset import FloodSequenceDataset
from ..utils import list_npz_files, save_json


def build_event_split_manifest(
    fused_dir: str | Path,
    seed: int,
    shuffle: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    files = [path for path in list_npz_files(fused_dir) if path.name.startswith("event_")]
    train, val, test = FloodSequenceDataset.split_indices(
        len(files), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed, shuffle=shuffle
    )
    partitions = {"train": train, "val": val, "test": test}
    index_sets = {name: set(indices) for name, indices in partitions.items()}
    if index_sets["train"] & index_sets["val"] or index_sets["train"] & index_sets["test"] or index_sets["val"] & index_sets["test"]:
        raise ValueError("Event split partitions overlap")
    return {
        "split_unit": "event",
        "split_seed": int(seed),
        "shuffle": bool(shuffle),
        "num_events": len(files),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "partitions": {
            name: [
                {"event_index": int(index), "event_id": files[index].stem, "file_name": files[index].name}
                for index in indices
            ]
            for name, indices in partitions.items()
        },
        "leakage_check": {
            "event_disjoint": True,
            "window_overlap_across_partitions": False,
        },
    }


def save_event_split_manifest(
    fused_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    shuffle: bool = True,
) -> tuple[dict[str, Any], Path]:
    manifest = build_event_split_manifest(fused_dir, seed=seed, shuffle=shuffle)
    path = Path(output_dir) / "artifacts" / "splits" / f"split_seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, path)
    return manifest, path

