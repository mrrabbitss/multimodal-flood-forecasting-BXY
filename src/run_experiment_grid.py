from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from .utils import ensure_dir, load_json, save_json


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    preferred = [
        "run_name",
        "seed",
        "split_seed",
        "hidden",
        "num_layers",
        "dropout",
        "threshold",
        "loss_threshold",
        "auto_threshold",
        "threshold_metric",
        "bce_loss_weight",
        "dice_loss_weight",
        "focal_loss_weight",
        "checkpoint_metric",
        "mae",
        "rmse",
        "precision",
        "recall_pod",
        "f1",
        "csi",
        "far",
        "accuracy",
        "loss",
        "best_epoch",
        "runtime_sec",
    ]
    fieldnames = preferred + sorted(k for k in rows[0] if k not in preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Conv-LSTM capacity and seed grid on an existing fused dataset.")
    parser.add_argument("--fused_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="runs/grid")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--hidden_values", type=str, default="24,32")
    parser.add_argument("--num_layers_values", type=str, default="1")
    parser.add_argument("--dropout_values", type=str, default="0.0")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--input_channels", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--auto_threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold_candidates", type=str, default="0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36")
    parser.add_argument("--threshold_metric", type=str, default="csi", choices=["csi", "f1"])
    parser.add_argument("--class_threshold", type=float, default=None)
    parser.add_argument("--class_temperature", type=float, default=0.04)
    parser.add_argument("--bce_loss_weight", type=float, default=0.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.0)
    parser.add_argument("--focal_loss_weight", type=float, default=0.0)
    parser.add_argument("--checkpoint_metric", type=str, default="loss", choices=["loss", "mae", "rmse", "csi", "f1"])
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--shuffle_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    hidden_values = parse_int_list(args.hidden_values)
    num_layers_values = parse_int_list(args.num_layers_values)
    dropout_values = parse_float_list(args.dropout_values)

    output_root = ensure_dir(args.output_root)
    rows: list[dict] = []
    for hidden in hidden_values:
        for num_layers in num_layers_values:
            for dropout in dropout_values:
                for seed in seeds:
                    run_name = f"h{hidden}_l{num_layers}_d{dropout:g}_seed{seed}"
                    out_dir = output_root / run_name / "outputs"
                    cmd = [
                        sys.executable,
                        "-m",
                        "src.train",
                        "--fused_dir",
                        args.fused_dir,
                        "--output_dir",
                        str(out_dir),
                        "--input_len",
                        str(args.input_len),
                        "--lead_time",
                        str(args.lead_time),
                        "--input_channels",
                        args.input_channels,
                        "--epochs",
                        str(args.epochs),
                        "--batch_size",
                        str(args.batch_size),
                        "--hidden",
                        str(hidden),
                        "--num_layers",
                        str(num_layers),
                        "--dropout",
                        str(dropout),
                        "--seed",
                        str(seed),
                        "--split_seed",
                        str(seed),
                        "--threshold",
                        str(args.threshold),
                        "--loss_threshold",
                        str(args.loss_threshold),
                        "--threshold_candidates",
                        args.threshold_candidates,
                        "--threshold_metric",
                        args.threshold_metric,
                        "--class_temperature",
                        str(args.class_temperature),
                        "--bce_loss_weight",
                        str(args.bce_loss_weight),
                        "--dice_loss_weight",
                        str(args.dice_loss_weight),
                        "--focal_loss_weight",
                        str(args.focal_loss_weight),
                        "--checkpoint_metric",
                        args.checkpoint_metric,
                        "--early_stop_patience",
                        str(args.early_stop_patience),
                        "--device",
                        args.device,
                    ]
                    if args.auto_threshold:
                        cmd.append("--auto_threshold")
                    if args.class_threshold is not None:
                        cmd.extend(["--class_threshold", str(args.class_threshold)])
                    if not args.shuffle_split:
                        cmd.append("--no-shuffle_split")

                    print("\n>>> " + " ".join(cmd), flush=True)
                    if not args.dry_run:
                        subprocess.run(cmd, check=True)
                        metrics_path = out_dir / "metrics" / "test_metrics.json"
                        metrics = load_json(metrics_path)
                        metrics.update(
                            {
                                "run_name": run_name,
                                "seed": seed,
                                "hidden": hidden,
                                "num_layers": num_layers,
                                "dropout": dropout,
                            }
                        )
                        rows.append(metrics)
                        save_json({"rows": rows}, output_root / "grid_results.json")
                        write_csv(rows, output_root / "grid_results.csv")

    if rows:
        best = max(rows, key=lambda row: (row.get("csi", 0.0), row.get("f1", 0.0), -row.get("mae", 9e9)))
        print(f"\nBest by CSI: {best['run_name']} csi={best['csi']:.4f} f1={best['f1']:.4f} mae={best['mae']:.4f}")
        save_json({"rows": rows, "best_by_csi": best}, output_root / "grid_results.json")
        write_csv(rows, output_root / "grid_results.csv")


if __name__ == "__main__":
    main()
