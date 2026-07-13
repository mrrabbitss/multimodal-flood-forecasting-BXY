from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir, load_json, save_json


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Lead times must be positive integers")
    return values


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["lead_time", "mae", "rmse", "csi", "far", "recall_pod", "f1", "threshold", "num_test_samples", "best_epoch"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_lead_times(rows: list[dict], path: Path) -> None:
    leads = [row["lead_time"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(leads, [row["mae"] for row in rows], marker="o", label="MAE", color="#386CB0")
    axes[0].plot(leads, [row["rmse"] for row in rows], marker="s", label="RMSE", color="#F28E2B")
    axes[0].set_title("Regression error by lead time")
    axes[0].set_ylabel("Lower is better")
    axes[0].legend()
    axes[1].plot(leads, [row["csi"] for row in rows], marker="o", label="CSI", color="#4E9A51")
    axes[1].plot(leads, [row["far"] for row in rows], marker="s", label="FAR", color="#D95F5F")
    axes[1].plot(leads, [row["recall_pod"] for row in rows], marker="^", label="POD", color="#7A5195")
    axes[1].set_title("Risk skill by lead time")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Forecast lead (time steps)")
        axis.set_xticks(leads)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Conv-LSTM at multiple forecast lead times.")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--output_root", default="runs/lead_time")
    parser.add_argument("--lead_times", default="1,3,6,12,24")
    parser.add_argument("--input_channels", default="full")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument(
        "--evaluation_threshold",
        type=float,
        default=None,
        help="Optional pre-registered common threshold used for every test lead.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--auto_threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    rows = []
    for lead_time in parse_int_list(args.lead_times):
        output_dir = output_root / f"lead_{lead_time}" / "outputs"
        checkpoint = output_dir / "checkpoints" / "best.pt"
        if not args.skip_training:
            command = [
                sys.executable, "-m", "src.train",
                "--fused_dir", args.fused_dir,
                "--output_dir", str(output_dir),
                "--input_channels", args.input_channels,
                "--input_len", str(args.input_len),
                "--lead_time", str(lead_time),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--hidden", str(args.hidden),
                "--seed", str(args.seed),
                "--split_seed", str(args.split_seed),
                "--threshold", str(args.threshold),
                "--device", args.device,
                "--no-progress",
            ]
            command.append("--auto_threshold" if args.auto_threshold else "--no-auto_threshold")
            print("\n>>> " + " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        elif not checkpoint.exists() and not args.dry_run:
            raise FileNotFoundError(checkpoint)
        evaluate = [
            sys.executable, "-m", "src.evaluate",
            "--fused_dir", args.fused_dir,
            "--checkpoint", str(checkpoint),
            "--output_dir", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
        ]
        if args.evaluation_threshold is not None:
            evaluate.extend(["--threshold", str(args.evaluation_threshold)])
        print(">>> " + " ".join(evaluate), flush=True)
        if not args.dry_run:
            subprocess.run(evaluate, check=True)
            metrics = load_json(output_dir / "metrics" / "eval_metrics.json")
            rows.append({"lead_time": lead_time, **metrics})

    if args.dry_run:
        return
    write_csv(rows, output_root / "lead_time_metrics.csv")
    save_json(
        {
            "rows": rows,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "channel_set": args.input_channels,
            "threshold_policy": (
                f"pre-registered common evaluation threshold {args.evaluation_threshold}"
                if args.evaluation_threshold is not None
                else ("selected on validation" if args.auto_threshold else f"fixed at {args.threshold}")
            ),
        },
        output_root / "lead_time_metrics.json",
    )
    plot_lead_times(rows, output_root / "lead_time_curves.png")
    print(f"Lead-time results: {output_root / 'lead_time_metrics.json'}")


if __name__ == "__main__":
    main()
