from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .batch4_engine import benchmark_model
from .data.schemas import depth_scale_from_checkpoint, make_risk_threshold
from .dataset import FloodSequenceDataset, channel_names_from_checkpoint, validate_checkpoint_data_schema
from .metrics import all_metrics
from .model import ConvLSTMForecastNet
from .training.losses import LossConfig, build_loss
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def write_per_event_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = ["event_id", "num_samples", "mae", "rmse", "csi", "f1", "far", "recall_pod", "peak_depth_error"]
    extra = sorted(key for key in rows[0] if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--per_event", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--benchmark_batches", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    ckpt = torch.load(args.checkpoint, map_location=device)
    input_len = int(ckpt["input_len"])
    lead_time = int(ckpt["lead_time"])
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("threshold", 0.30))
    depth_scale = depth_scale_from_checkpoint(ckpt)
    channel_names = channel_names_from_checkpoint(ckpt)
    data_schema = validate_checkpoint_data_schema(ckpt, args.fused_dir)
    loss_config = LossConfig.from_checkpoint(ckpt)
    split_seed = ckpt.get("split_seed", args.seed)
    shuffle_split = bool(ckpt.get("shuffle_split", False))

    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    _, _, test_idx = FloodSequenceDataset.split_indices(len(files), seed=split_seed, shuffle=shuffle_split)
    test_ds = FloodSequenceDataset(
        args.fused_dir,
        test_idx,
        input_len=input_len,
        lead_time=lead_time,
        channel_names=channel_names,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvLSTMForecastNet(
        input_channels=ckpt["input_channels"],
        hidden_channels=ckpt["hidden_channels"],
        num_layers=int(ckpt.get("num_layers", 1)),
        dropout=float(ckpt.get("dropout", 0.0)),
        output_max=depth_scale.max_value,
        residual_scale=float(ckpt.get("residual_scale", 0.35)),
        use_residual=bool(ckpt.get("use_residual", False)),
        fused_channel=int(ckpt.get("fused_channel", channel_names.index("fused_depth") if "fused_depth" in channel_names else -1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    efficiency = benchmark_model(
        model,
        loader,
        device,
        warmup_batches=args.warmup_batches,
        benchmark_batches=args.benchmark_batches,
    )

    preds, targets = [], []
    event_predictions: dict[str, list[np.ndarray]] = {}
    event_targets: dict[str, list[np.ndarray]] = {}
    loss_values: dict[str, list[float]] = {}
    sample_cursor = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            target_batch = y.numpy()
            targets.append(target_batch)
            if args.per_event:
                batch_samples = test_ds.samples[sample_cursor : sample_cursor + pred.shape[0]]
                for local_index, sample in enumerate(batch_samples):
                    event_id = sample.file_path.stem
                    event_predictions.setdefault(event_id, []).append(pred[local_index : local_index + 1])
                    event_targets.setdefault(event_id, []).append(target_batch[local_index : local_index + 1])
                sample_cursor += pred.shape[0]
            breakdown = build_loss(torch.from_numpy(pred), y, loss_config)
            for key, value in breakdown.detached_values().items():
                loss_values.setdefault(key, []).append(value)
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = all_metrics(pred_np, target_np, threshold=threshold)
    for key, values in loss_values.items():
        metrics[key] = float(np.mean(values))
    metrics["loss"] = metrics["loss_total"]
    metrics["num_test_samples"] = int(len(test_ds))
    metrics["test_events"] = test_idx
    metrics["best_epoch"] = int(ckpt.get("best_epoch", 0))
    metrics["parameter_count"] = int(
        ckpt.get("parameter_count", sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    )
    metrics["split_seed"] = int(split_seed)
    metrics["shuffle_split"] = bool(shuffle_split)
    metrics["threshold"] = float(threshold)
    metrics["risk_threshold"] = make_risk_threshold(threshold, depth_scale).to_dict()
    metrics["depth_scale"] = depth_scale.to_dict()
    metrics["channel_names"] = list(channel_names)
    metrics["data_schema"] = data_schema
    metrics["loss_config"] = loss_config.to_dict()
    metrics.update(efficiency)
    metrics["benchmark"] = {
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "benchmark_batches": args.benchmark_batches,
    }

    out_dir = ensure_dir(Path(args.output_dir) / "metrics")
    per_event_rows = []
    if args.per_event:
        for event_id in sorted(event_predictions):
            event_pred = np.concatenate(event_predictions[event_id], axis=0)
            event_target = np.concatenate(event_targets[event_id], axis=0)
            row = all_metrics(event_pred, event_target, threshold=threshold)
            row["event_id"] = event_id
            row["num_samples"] = int(event_pred.shape[0])
            per_event_rows.append(row)
        write_per_event_csv(per_event_rows, out_dir / "per_event_metrics.csv")
        save_json({"rows": per_event_rows, "threshold": threshold}, out_dir / "per_event_metrics.json")
        metrics["per_event_metrics_file"] = str(out_dir / "per_event_metrics.csv")
    save_json(metrics, out_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
