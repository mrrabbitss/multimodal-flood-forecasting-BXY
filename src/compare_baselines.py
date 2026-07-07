from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FloodSequenceDataset
from .metrics import all_metrics
from .model import ConvLSTMForecastNet
from .utils import ensure_dir, list_npz_files, save_json, set_seed


BASELINE_CHANNELS = {
    "persistence_meteo": 0,
    "persistence_sat_proxy": 1,
    "persistence_soc": 3,
    "persistence_fused": 4,
    "persistence_risk_score": 5,
}


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def evaluate_arrays(name: str, pred: np.ndarray, target: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        metrics = all_metrics(pred, target, threshold=threshold)
        metrics["model"] = name
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    preferred = [
        "model",
        "threshold",
        "mae",
        "rmse",
        "precision",
        "recall_pod",
        "f1",
        "iou",
        "csi",
        "far",
        "accuracy",
    ]
    fieldnames = preferred + sorted(k for k in rows[0] if k not in preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Conv-LSTM checkpoint against simple depth baselines.")
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--shuffle_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thresholds", type=str, default="0.25,0.28,0.30")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    ckpt = None
    input_len = args.input_len
    lead_time = args.lead_time
    split_seed = args.seed if args.split_seed is None else args.split_seed
    shuffle_split = args.shuffle_split
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        input_len = int(ckpt.get("input_len", input_len))
        lead_time = int(ckpt.get("lead_time", lead_time))
        split_seed = int(ckpt.get("split_seed", split_seed))
        shuffle_split = bool(ckpt.get("shuffle_split", shuffle_split))

    thresholds = parse_float_list(args.thresholds)
    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    _, _, test_idx = FloodSequenceDataset.split_indices(len(files), seed=split_seed, shuffle=shuffle_split)
    test_ds = FloodSequenceDataset(args.fused_dir, test_idx, input_len=input_len, lead_time=lead_time)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = None
    if ckpt is not None:
        model = ConvLSTMForecastNet(
            input_channels=int(ckpt["input_channels"]),
            hidden_channels=int(ckpt["hidden_channels"]),
            num_layers=int(ckpt.get("num_layers", 1)),
            dropout=float(ckpt.get("dropout", 0.0)),
            output_max=float(ckpt.get("output_max", 1.0)),
            residual_scale=float(ckpt.get("residual_scale", 0.35)),
            use_residual=bool(ckpt.get("use_residual", False)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

    targets = []
    preds_by_name: dict[str, list[np.ndarray]] = {"zero_depth": []}
    if model is not None:
        preds_by_name["convlstm"] = []
    for name in BASELINE_CHANNELS:
        preds_by_name[name] = []

    with torch.no_grad():
        for x, y in loader:
            targets.append(y.numpy())
            x_np = x.numpy()
            preds_by_name["zero_depth"].append(np.zeros_like(y.numpy(), dtype=np.float32))
            for name, channel in BASELINE_CHANNELS.items():
                if channel < x_np.shape[2]:
                    preds_by_name[name].append(x_np[:, -1, channel : channel + 1])
            if model is not None:
                preds_by_name["convlstm"].append(model(x.to(device)).cpu().numpy())

    target_np = np.concatenate(targets, axis=0)
    rows = []
    for name, parts in preds_by_name.items():
        if not parts:
            continue
        pred_np = np.concatenate(parts, axis=0)
        rows.extend(evaluate_arrays(name, pred_np, target_np, thresholds))

    rows.sort(key=lambda row: (row["threshold"], -row["csi"], row["model"]))
    out_dir = ensure_dir(Path(args.output_dir) / "metrics")
    save_json(
        {
            "rows": rows,
            "num_test_samples": int(len(test_ds)),
            "test_events": test_idx,
            "split_seed": int(split_seed),
            "shuffle_split": bool(shuffle_split),
        },
        out_dir / "baseline_comparison.json",
    )
    write_csv(rows, out_dir / "baseline_comparison.csv")

    for row in rows:
        print(
            f"{row['model']:>22s} thr={row['threshold']:.2f} "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} CSI={row['csi']:.4f} "
            f"F1={row['f1']:.4f} FAR={row['far']:.4f}"
        )


if __name__ == "__main__":
    main()
