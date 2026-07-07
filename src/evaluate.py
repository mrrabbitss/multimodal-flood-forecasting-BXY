from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FloodSequenceDataset
from .metrics import all_metrics
from .model import ConvLSTMForecastNet
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    ckpt = torch.load(args.checkpoint, map_location=device)
    input_len = int(ckpt["input_len"])
    lead_time = int(ckpt["lead_time"])
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("threshold", 0.30))
    split_seed = ckpt.get("split_seed", args.seed)
    shuffle_split = bool(ckpt.get("shuffle_split", False))

    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    _, _, test_idx = FloodSequenceDataset.split_indices(len(files), seed=split_seed, shuffle=shuffle_split)
    test_ds = FloodSequenceDataset(args.fused_dir, test_idx, input_len=input_len, lead_time=lead_time)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvLSTMForecastNet(
        input_channels=ckpt["input_channels"],
        hidden_channels=ckpt["hidden_channels"],
        num_layers=int(ckpt.get("num_layers", 1)),
        dropout=float(ckpt.get("dropout", 0.0)),
        output_max=float(ckpt.get("output_max", 1.0)),
        residual_scale=float(ckpt.get("residual_scale", 0.35)),
        use_residual=bool(ckpt.get("use_residual", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = all_metrics(pred_np, target_np, threshold=threshold)
    metrics["num_test_samples"] = int(len(test_ds))
    metrics["test_events"] = test_idx
    metrics["best_epoch"] = int(ckpt.get("best_epoch", 0))
    metrics["split_seed"] = int(split_seed)
    metrics["shuffle_split"] = bool(shuffle_split)
    metrics["threshold"] = float(threshold)

    out_dir = ensure_dir(Path(args.output_dir) / "metrics")
    save_json(metrics, out_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
