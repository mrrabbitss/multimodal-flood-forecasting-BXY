from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .batch4_dataset import MultiHorizonFloodDataset
from .batch4_engine import benchmark_model, configure_determinism, evaluate_batch4_loader
from .batch4_models import build_batch4_model_from_checkpoint
from .dataset import FloodSequenceDataset, channel_names_from_checkpoint, validate_checkpoint_data_schema
from .training.losses import LossConfig
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Batch 4 checkpoint.")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--benchmark_batches", type=int, default=20)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    seed = int(checkpoint.get("seed", 42))
    set_seed(seed)
    configure_determinism(True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    channel_names = channel_names_from_checkpoint(checkpoint)
    data_schema = validate_checkpoint_data_schema(checkpoint, args.fused_dir)
    files = [path for path in list_npz_files(args.fused_dir) if path.name.startswith("event_")]
    _, _, test_indices = FloodSequenceDataset.split_indices(
        len(files), seed=int(checkpoint["split_seed"]), shuffle=bool(checkpoint.get("shuffle_split", True))
    )
    lead_times = tuple(int(value) for value in checkpoint["lead_times"])
    dataset = MultiHorizonFloodDataset(
        args.fused_dir, test_indices, int(checkpoint["input_len"]), lead_times, channel_names
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_batch4_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"])
    efficiency = benchmark_model(model, loader, device, args.warmup_batches, args.benchmark_batches)
    result = evaluate_batch4_loader(
        model,
        loader,
        device,
        lead_times,
        float(checkpoint["threshold"]),
        LossConfig.from_checkpoint(checkpoint),
        include_per_event=True,
    )
    result["aggregate"].update(
        {
            "model_type": checkpoint["model_type"],
            "model_label": checkpoint["model_label"],
            "parameter_count": int(checkpoint["parameter_count"]),
            "seed": seed,
            "split_seed": int(checkpoint["split_seed"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "lead_times": list(lead_times),
            "channel_names": list(channel_names),
            "data_schema": data_schema,
            **efficiency,
        }
    )
    output_dir = ensure_dir(Path(args.output_dir) / "metrics")
    save_json(result, output_dir / "batch4_eval_metrics.json")
    write_csv(result["per_horizon"], output_dir / "per_horizon_metrics.csv")
    write_csv(result["per_event_horizon"], output_dir / "per_event_horizon_metrics.csv")
    print(result["aggregate"])


if __name__ == "__main__":
    main()
