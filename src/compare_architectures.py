from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate_architecture import evaluate_checkpoint
from .model_variants import model_display_name, normalize_model_type
from .utils import ensure_dir, load_json, save_json


DEFAULT_BASELINE_CHECKPOINT = "runs/large60_grid_h24_h32_l1/h32_l1_d0_seed44/outputs/checkpoints/best.pt"


def parse_model_types(value: str) -> list[str]:
    return [normalize_model_type(x.strip()) for x in value.split(",") if x.strip()]


def checkpoint_defaults(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    ckpt = torch.load(p, map_location="cpu")
    return {
        "input_len": int(ckpt.get("input_len", 12)),
        "lead_time": int(ckpt.get("lead_time", 6)),
        "split_seed": int(ckpt.get("split_seed", 42)),
        "shuffle_split": bool(ckpt.get("shuffle_split", True)),
        "threshold": float(ckpt.get("threshold", 0.30)),
    }


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    preferred = [
        "model_label",
        "model_type",
        "mae",
        "rmse",
        "csi",
        "f1",
        "precision",
        "recall_pod",
        "far",
        "accuracy",
        "latency_ms_per_sample",
        "latency_ms_per_batch",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "parameter_count",
        "threshold",
        "best_epoch",
        "training_runtime_sec",
        "evaluation_runtime_sec",
        "checkpoint",
        "output_dir",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = preferred + sorted(k for k in all_keys if k not in preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_bars(ax, values: list[float], fmt: str) -> None:
    for idx, value in enumerate(values):
        ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_metric_bars(rows: list[dict], fig_dir: Path) -> None:
    labels = [str(row["model_label"]) for row in rows]
    x = np.arange(len(labels))
    mae = [float(row["mae"]) for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    csi = [float(row["csi"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.36
    axes[0].bar(x - width / 2, mae, width=width, label="MAE", color="#4E79A7")
    axes[0].bar(x + width / 2, rmse, width=width, label="RMSE", color="#F28E2B")
    axes[0].set_ylabel("Lower is better")
    axes[0].set_title("Regression Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, csi, width=width, label="CSI", color="#59A14F")
    axes[1].bar(x + width / 2, f1, width=width, label="F1", color="#E15759")
    axes[1].set_ylabel("Higher is better")
    axes[1].set_title("Risk Mask Skill")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].set_ylim(0.0, max(1.0, max(csi + f1) * 1.08))
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_dir / "architecture_metrics.png", dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict], fig_dir: Path) -> None:
    labels = [str(row["model_label"]) for row in rows]
    x = np.arange(len(labels))
    latency = [float(row.get("latency_ms_per_sample", 0.0)) for row in rows]
    memory = [float(row.get("peak_memory_allocated_mb", 0.0)) for row in rows]
    params = [float(row.get("parameter_count", 0.0)) / 1_000_000.0 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(x, latency, color="#4E79A7")
    axes[0].set_title("Inference Latency")
    axes[0].set_ylabel("ms / sample")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bars(axes[0], latency, "{:.2f}")

    axes[1].bar(x, memory, color="#B07AA1")
    axes[1].set_title("Peak CUDA Memory")
    axes[1].set_ylabel("MB")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    annotate_bars(axes[1], memory, "{:.0f}")

    axes[2].bar(x, params, color="#9C755F")
    axes[2].set_title("Trainable Parameters")
    axes[2].set_ylabel("Millions")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=12, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    annotate_bars(axes[2], params, "{:.2f}")

    fig.tight_layout()
    fig.savefig(fig_dir / "architecture_efficiency.png", dpi=180)
    plt.close(fig)


def find_history_path(row: dict) -> Path | None:
    output_dir = row.get("output_dir")
    if output_dir:
        p = Path(output_dir) / "metrics" / "train_history.json"
        if p.exists():
            return p
    checkpoint = row.get("checkpoint")
    if checkpoint:
        p = Path(checkpoint).parent.parent / "metrics" / "train_history.json"
        if p.exists():
            return p
    return None


def attach_training_metadata(row: dict, output_dir: Path | None = None) -> None:
    candidates = []
    if output_dir is not None:
        candidates.append(output_dir / "metrics" / "test_metrics.json")
    checkpoint = row.get("checkpoint")
    if checkpoint:
        candidates.append(Path(checkpoint).parent.parent / "metrics" / "test_metrics.json")
    for path in candidates:
        if not path.exists():
            continue
        metrics = load_json(path)
        if "runtime_sec" in metrics:
            row["training_runtime_sec"] = float(metrics["runtime_sec"])
        if "best_epoch" in metrics:
            row["best_epoch"] = int(metrics["best_epoch"])
        return


def plot_training_curves(rows: list[dict], fig_dir: Path) -> None:
    histories = []
    for row in rows:
        path = find_history_path(row)
        if path is None:
            continue
        history = load_json(path)
        if "val_loss" in history:
            histories.append((str(row["model_label"]), history))
    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, history in histories:
        epochs = np.arange(1, len(history.get("val_loss", [])) + 1)
        if len(epochs) == 0:
            continue
        axes[0].plot(epochs, history.get("train_loss", []), marker="o", linewidth=1.5, label=f"{label} train")
        axes[0].plot(epochs, history.get("val_loss", []), marker="s", linewidth=1.5, label=f"{label} val")
        if "val_csi" in history:
            axes[1].plot(epochs, history.get("val_csi", []), marker="o", linewidth=1.5, label=label)

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Validation CSI")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CSI")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "architecture_training_curves.png", dpi=180)
    plt.close(fig)


def train_variant(args: argparse.Namespace, model_type: str, output_dir: Path, split_seed: int, shuffle_split: bool) -> Path:
    layers = args.transformer_layers if model_type == "cnn_temporal_transformer" else args.convlstm_layers
    cmd = [
        sys.executable,
        "-m",
        "src.train_architecture",
        "--model_type",
        model_type,
        "--fused_dir",
        args.fused_dir,
        "--output_dir",
        str(output_dir),
        "--input_len",
        str(args.input_len),
        "--lead_time",
        str(args.lead_time),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--num_layers",
        str(layers),
        "--dropout",
        str(args.dropout),
        "--attention_dropout",
        str(args.attention_dropout),
        "--transformer_heads",
        str(args.transformer_heads),
        "--transformer_ffn_mult",
        str(args.transformer_ffn_mult),
        "--seed",
        str(args.seed),
        "--split_seed",
        str(split_seed),
        "--threshold",
        str(args.threshold),
        "--loss_threshold",
        str(args.loss_threshold),
        "--checkpoint_metric",
        args.checkpoint_metric,
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--device",
        args.device,
    ]
    if args.use_residual:
        cmd.append("--use_residual")
    if not args.progress:
        cmd.append("--no-progress")
    if not shuffle_split:
        cmd.append("--no-shuffle_split")
    run_command(cmd, dry_run=args.dry_run)
    return output_dir / "checkpoints" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Conv-LSTM, Conv-LSTM + Attention, and CNN-Temporal Transformer.")
    parser.add_argument("--fused_dir", type=str, default="runs/large60_h24_l1_seed42/data/fused")
    parser.add_argument("--baseline_checkpoint", type=str, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--output_root", type=str, default="runs/architecture_comparison")
    parser.add_argument("--model_types", type=str, default="convlstm_attention,cnn_temporal_transformer")
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--lead_time", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--convlstm_layers", type=int, default=1)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ffn_mult", type=float, default=4.0)
    parser.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--checkpoint_metric", type=str, default="loss", choices=["loss", "mae", "rmse", "csi", "f1"])
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--shuffle_split", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--benchmark_batches", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    fig_dir = ensure_dir(output_root / "figures")
    defaults = checkpoint_defaults(args.baseline_checkpoint)
    args.input_len = args.input_len if args.input_len is not None else int(defaults.get("input_len", 12))
    args.lead_time = args.lead_time if args.lead_time is not None else int(defaults.get("lead_time", 6))
    split_seed = args.split_seed if args.split_seed is not None else int(defaults.get("split_seed", args.seed))
    shuffle_split = args.shuffle_split if args.shuffle_split is not None else bool(defaults.get("shuffle_split", True))

    rows: list[dict] = []
    baseline_path = Path(args.baseline_checkpoint)
    if baseline_path.exists():
        baseline_output = output_root / "conv_lstm_existing" / "outputs"
        if not args.dry_run:
            baseline_metrics = evaluate_checkpoint(
                fused_dir=args.fused_dir,
                checkpoint_path=baseline_path,
                output_dir=baseline_output,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device_value=args.device,
                seed=args.seed,
                threshold=args.threshold,
                warmup_batches=args.warmup_batches,
                benchmark_batches=args.benchmark_batches,
                amp=args.amp,
            )
            attach_training_metadata(baseline_metrics)
            baseline_metrics["output_dir"] = str(baseline_output)
            rows.append(baseline_metrics)
        else:
            print(f"Would evaluate baseline checkpoint: {baseline_path}")
    else:
        print(f"Baseline checkpoint not found, skipping: {baseline_path}")

    for model_type in parse_model_types(args.model_types):
        model_output = output_root / model_type / "outputs"
        checkpoint_path = model_output / "checkpoints" / "best.pt"
        if args.force_retrain or not checkpoint_path.exists():
            if args.skip_training:
                raise FileNotFoundError(f"Missing checkpoint for {model_type}: {checkpoint_path}")
            checkpoint_path = train_variant(args, model_type, model_output, split_seed, shuffle_split)
        else:
            print(f"Reusing existing checkpoint: {checkpoint_path}")

        if not args.dry_run:
            metrics = evaluate_checkpoint(
                fused_dir=args.fused_dir,
                checkpoint_path=checkpoint_path,
                output_dir=model_output,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device_value=args.device,
                seed=args.seed,
                threshold=args.threshold,
                warmup_batches=args.warmup_batches,
                benchmark_batches=args.benchmark_batches,
                amp=args.amp,
            )
            attach_training_metadata(metrics, model_output)
            metrics["output_dir"] = str(model_output)
            rows.append(metrics)

    if not args.dry_run and rows:
        model_order = {
            "convlstm": 0,
            "convlstm_attention": 1,
            "cnn_temporal_transformer": 2,
        }
        rows.sort(key=lambda row: model_order.get(str(row.get("model_type")), 99))
        save_json(
            {
                "rows": rows,
                "fused_dir": args.fused_dir,
                "threshold": float(args.threshold),
                "split_seed": int(split_seed),
                "shuffle_split": bool(shuffle_split),
            },
            output_root / "architecture_comparison.json",
        )
        write_csv(rows, output_root / "architecture_comparison.csv")
        plot_metric_bars(rows, fig_dir)
        plot_efficiency(rows, fig_dir)
        plot_training_curves(rows, fig_dir)

        best = max(rows, key=lambda row: (float(row.get("csi", 0.0)), float(row.get("f1", 0.0)), -float(row.get("mae", 9e9))))
        print(
            f"\nBest by CSI: {best['model_label']} "
            f"CSI={best['csi']:.4f} MAE={best['mae']:.4f} "
            f"latency={best.get('latency_ms_per_sample', 0.0):.2f} ms/sample"
        )
        print(f"Comparison CSV: {output_root / 'architecture_comparison.csv'}")
        print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
