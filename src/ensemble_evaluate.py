from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodSequenceDataset
from .metrics import all_metrics
from .model import ConvLSTMForecastNet
from .utils import ensure_dir, list_npz_files, save_json


def parse_checkpoint_paths(value: str) -> list[Path]:
    paths = [Path(x.strip().strip('"')) for x in value.split(",") if x.strip()]
    if not paths:
        raise ValueError("Provide at least one checkpoint path.")
    return paths


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> ConvLSTMForecastNet:
    model = ConvLSTMForecastNet(
        input_channels=int(checkpoint["input_channels"]),
        hidden_channels=int(checkpoint.get("hidden_channels", 24)),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.0)),
        output_max=float(checkpoint.get("output_max", 1.0)),
        residual_scale=float(checkpoint.get("residual_scale", 0.35)),
        use_residual=bool(checkpoint.get("use_residual", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Average predictions from multiple Conv-LSTM checkpoints on one test split.")
    parser.add_argument("--fused_dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    print(f"Using device: {device}")

    checkpoint_paths = parse_checkpoint_paths(args.checkpoints)
    checkpoints = [torch.load(path, map_location=device) for path in checkpoint_paths]
    base = checkpoints[0]
    input_len = int(base.get("input_len", 12))
    lead_time = int(base.get("lead_time", 6))
    split_seed = int(base.get("split_seed", args.seed))
    shuffle_split = bool(base.get("shuffle_split", True))
    threshold = float(args.threshold if args.threshold is not None else base.get("threshold", 0.30))

    for path, checkpoint in zip(checkpoint_paths, checkpoints):
        if int(checkpoint.get("input_len", input_len)) != input_len or int(checkpoint.get("lead_time", lead_time)) != lead_time:
            raise ValueError(f"Checkpoint {path} uses a different input_len/lead_time.")
        if int(checkpoint.get("split_seed", split_seed)) != split_seed or bool(checkpoint.get("shuffle_split", shuffle_split)) != shuffle_split:
            raise ValueError(f"Checkpoint {path} uses a different split; ensemble would not be comparable.")

    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    train_idx, val_idx, test_idx = FloodSequenceDataset.split_indices(
        len(files),
        seed=split_seed,
        shuffle=shuffle_split,
    )
    test_ds = FloodSequenceDataset(args.fused_dir, test_idx, input_len, lead_time)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    models = [build_model_from_checkpoint(checkpoint, device) for checkpoint in checkpoints]
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Ensembling {len(models)} models"):
            x = x.to(device, non_blocking=True)
            batch_preds = [model(x) for model in models]
            pred = torch.stack(batch_preds, dim=0).mean(dim=0)
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.numpy())

    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = all_metrics(pred_np, target_np, threshold=threshold)
    metrics.update(
        {
            "threshold": threshold,
            "n_models": len(models),
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "train_events": train_idx,
            "val_events": val_idx,
            "test_events": test_idx,
            "split_seed": split_seed,
            "shuffle_split": shuffle_split,
        }
    )

    metric_dir = ensure_dir(Path(args.output_dir) / "metrics")
    save_json(metrics, metric_dir / "ensemble_metrics.json")
    print(f"Ensemble metrics: {metrics}")


if __name__ == "__main__":
    main()
