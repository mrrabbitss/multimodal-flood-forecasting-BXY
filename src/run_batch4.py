from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .batch4_engine import parse_lead_times
from .batch4_models import MODEL_CONVLSTM_UNET, MODEL_TYPES, model_display_name
from .experiments.statistics import paired_bootstrap, summarize_values
from .utils import ensure_dir, load_json, save_json


DEFAULT_SEEDS = "42,44,52,77,2026"


def parse_names(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("Model names must be unique and non-empty")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return names


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and non-empty")
    return seeds


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, group in grouped.items():
        summary = {name: value for name, value in zip(group_keys, key)}
        summary["seeds"] = sorted({int(row["seed"]) for row in group})
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            statistics = summarize_values(values)
            for statistic in ("mean", "std", "min", "max"):
                summary[f"{metric}_{statistic}"] = statistics[statistic]
        output.append(summary)
    return output


def event_seed_means(rows: list[dict], model_type: str, lead_time: int, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["model_type"] == model_type and int(row["lead_time"]) == lead_time:
            grouped[str(row["event_id"])].append(float(row[metric]))
    return {event_id: float(np.mean(values)) for event_id, values in grouped.items()}


def plot_model_summary(rows: list[dict], path: Path) -> None:
    labels = [model_display_name(str(row["model_type"])) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, [row["mae_mean"] for row in rows], yerr=[row["mae_std"] for row in rows], capsize=4, color="#386CB0")
    axes[0].set_title("Five-seed aggregate MAE")
    axes[0].set_ylabel("Lower is better")
    axes[1].bar(x, [row["csi_mean"] for row in rows], yerr=[row["csi_std"] for row in rows], capsize=4, color="#4E9A51")
    axes[1].set_title("Five-seed aggregate CSI")
    axes[1].set_ylabel("Higher is better")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon_curves(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#386CB0", "#F28E2B", "#7A5195", "#4E9A51"]
    for color, model_type in zip(colors, MODEL_TYPES):
        model_rows = sorted((row for row in rows if row["model_type"] == model_type), key=lambda row: int(row["lead_time"]))
        if not model_rows:
            continue
        leads = np.asarray([row["lead_time"] for row in model_rows])
        label = model_display_name(model_type)
        for axis, metric in ((axes[0], "mae"), (axes[1], "csi")):
            means = np.asarray([row[f"{metric}_mean"] for row in model_rows])
            stds = np.asarray([row[f"{metric}_std"] for row in model_rows])
            axis.plot(leads, means, marker="o", label=label, color=color)
            axis.fill_between(leads, means - stds, means + stds, alpha=0.12, color=color)
    axes[0].set_title("MAE by forecast horizon")
    axes[0].set_ylabel("Lower is better")
    axes[1].set_title("CSI by forecast horizon")
    axes[1].set_ylabel("Higher is better")
    for axis in axes:
        axis.set_xlabel("Lead time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(
            row["latency_ms_per_sample_mean"], row["csi_mean"],
            s=max(50.0, row["parameter_count_mean"] / 450.0), alpha=0.8,
            label=model_display_name(str(row["model_type"])),
        )
    axis.set_xlabel("Latency (ms/sample, lower is better)")
    axis.set_ylabel("Aggregate CSI (higher is better)")
    axis.set_title("Accuracy-efficiency tradeoff; marker area follows parameter count")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Batch 4 five-seed architecture benchmark.")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--output_root", default="runs/batch4_multihorizon/experiments")
    parser.add_argument("--models", default=",".join(MODEL_TYPES))
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--input_channels", default="full")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_times", default="1,3,6,12,24")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    models = parse_names(args.models, MODEL_TYPES)
    seeds = parse_seeds(args.seeds)
    lead_times = parse_lead_times(args.lead_times)
    output_root = ensure_dir(args.output_root)
    aggregate_rows: list[dict] = []
    horizon_rows: list[dict] = []
    event_rows: list[dict] = []
    for model_type in models:
        for seed in seeds:
            output_dir = output_root / model_type / f"seed_{seed}" / "outputs"
            checkpoint = output_dir / "checkpoints" / "best.pt"
            if not args.skip_training:
                train = [
                    sys.executable, "-m", "src.train_batch4",
                    "--model_type", model_type,
                    "--fused_dir", args.fused_dir,
                    "--output_dir", str(output_dir),
                    "--input_channels", args.input_channels,
                    "--input_len", str(args.input_len),
                    "--lead_times", args.lead_times,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--hidden", str(args.hidden),
                    "--threshold", str(args.threshold),
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--device", args.device,
                    "--no-progress",
                ]
                print("\n>>> " + " ".join(train), flush=True)
                if not args.dry_run:
                    subprocess.run(train, check=True)
            elif not checkpoint.exists() and not args.dry_run:
                raise FileNotFoundError(checkpoint)
            evaluate = [
                sys.executable, "-m", "src.evaluate_batch4",
                "--fused_dir", args.fused_dir,
                "--checkpoint", str(checkpoint),
                "--output_dir", str(output_dir),
                "--batch_size", str(args.batch_size),
                "--device", args.device,
            ]
            print(">>> " + " ".join(evaluate), flush=True)
            if args.dry_run:
                continue
            subprocess.run(evaluate, check=True)
            result = load_json(output_dir / "metrics" / "batch4_eval_metrics.json")
            aggregate_rows.append(dict(result["aggregate"]))
            horizon_rows.extend({**row, "model_type": model_type, "seed": seed} for row in result["per_horizon"])
            event_rows.extend({**row, "model_type": model_type, "seed": seed} for row in result["per_event_horizon"])
    if args.dry_run:
        return

    aggregate_summary = summarize_rows(
        aggregate_rows,
        ("model_type",),
        ("mae", "rmse", "csi", "f1", "far", "latency_ms_per_sample", "peak_cuda_memory_mb", "parameter_count"),
    )
    horizon_summary = summarize_rows(horizon_rows, ("model_type", "lead_time"), ("mae", "rmse", "csi", "f1", "far"))
    bootstrap_rows = []
    if MODEL_CONVLSTM_UNET in models:
        for baseline in [model for model in models if model != MODEL_CONVLSTM_UNET]:
            for lead_time in lead_times:
                for metric in ("mae", "csi"):
                    reference = event_seed_means(event_rows, baseline, lead_time, metric)
                    candidate = event_seed_means(event_rows, MODEL_CONVLSTM_UNET, lead_time, metric)
                    event_ids = sorted(set(reference) & set(candidate))
                    statistics = paired_bootstrap(
                        [reference[event_id] for event_id in event_ids],
                        [candidate[event_id] for event_id in event_ids],
                        metric=metric,
                    )
                    bootstrap_rows.append(
                        {"baseline": baseline, "candidate": MODEL_CONVLSTM_UNET, "lead_time": lead_time, **statistics}
                    )

    write_csv(aggregate_rows, output_root / "per_seed_aggregate.csv")
    write_csv(horizon_rows, output_root / "per_seed_horizon.csv")
    write_csv(event_rows, output_root / "per_event_horizon.csv")
    write_csv(aggregate_summary, output_root / "model_summary.csv")
    write_csv(horizon_summary, output_root / "horizon_summary.csv")
    write_csv(bootstrap_rows, output_root / "paired_bootstrap.csv")
    save_json(
        {
            "experiment_batch": 4,
            "models": list(models),
            "seeds": list(seeds),
            "split_seed": args.split_seed,
            "lead_times": list(lead_times),
            "input_channels": args.input_channels,
            "epochs": args.epochs,
            "aggregate_summary": aggregate_summary,
            "horizon_summary": horizon_summary,
            "paired_bootstrap": bootstrap_rows,
        },
        output_root / "batch4_summary.json",
    )
    figure_dir = ensure_dir(output_root / "figures")
    plot_model_summary(aggregate_summary, figure_dir / "batch4_model_summary.png")
    plot_horizon_curves(horizon_summary, figure_dir / "batch4_horizon_curves.png")
    plot_efficiency(aggregate_summary, figure_dir / "batch4_efficiency.png")
    print(f"Batch 4 summary: {output_root / 'batch4_summary.json'}")


if __name__ == "__main__":
    main()
