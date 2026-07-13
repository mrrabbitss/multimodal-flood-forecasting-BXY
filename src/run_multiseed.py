from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiments.statistics import paired_bootstrap, summarize_multiseed
from .run_input_ablation import DEFAULT_VARIANTS, parse_variants
from .utils import ensure_dir, load_json, save_json


def parse_int_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty list without duplicates")
    return seeds


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summaries(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        row: dict[str, object] = {"variant": summary["variant"], "seeds": ",".join(map(str, summary["seeds"]))}
        for metric, values in summary.items():
            if isinstance(values, dict) and "mean" in values:
                for statistic in ("mean", "std", "min", "max"):
                    row[f"{metric}_{statistic}"] = values[statistic]
        rows.append(row)
    return rows


def aggregate_event_means(per_seed_rows: list[dict], variant: str, metric: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in per_seed_rows:
        if row["variant"] == variant:
            values[str(row["event_id"])].append(float(row[metric]))
    return {event_id: float(np.mean(event_values)) for event_id, event_values in values.items()}


def plot_multiseed(summaries: list[dict], path: Path) -> None:
    labels = [str(row["variant"]) for row in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, color, title in (
        (axes[0], "mae", "#386CB0", "MAE across training seeds"),
        (axes[1], "csi", "#4E9A51", "CSI across training seeds"),
    ):
        means = [row[metric]["mean"] for row in summaries]
        errors = [row[metric]["std"] for row in summaries]
        axis.bar(x, means, yerr=errors, capsize=4, color=color, alpha=0.88)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Lower is better" if metric == "mae" else "Higher is better")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bootstrap_intervals(rows: list[dict], path: Path) -> None:
    labels = [f"{row['candidate']} - {row['baseline']} ({str(row['metric']).upper()})" for row in rows]
    means = np.asarray([row["mean_improvement"] for row in rows], dtype=float)
    lower = np.asarray([row["ci_lower"] for row in rows], dtype=float)
    upper = np.asarray([row["ci_upper"] for row in rows], dtype=float)
    y = np.arange(len(rows))
    colors = ["#4E9A51" if low > 0 else "#777777" for low in lower]
    fig, axis = plt.subplots(figsize=(9.2, 4.5))
    for index in range(len(rows)):
        axis.errorbar(
            means[index], y[index],
            xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
            fmt="o", capsize=4, color=colors[index], markersize=6,
        )
    axis.axvline(0.0, color="#222222", linewidth=1)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Candidate improvement over baseline (positive is better)")
    axis.set_title("Paired event bootstrap 95% confidence intervals")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired multi-seed Conv-LSTM input experiment.")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--output_root", default="runs/multiseed")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", default="42,44,52,77,2026")
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment_kind", choices=["smoke", "controlled", "formal"], default="formal")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    seeds = parse_int_list(args.seeds)
    output_root = ensure_dir(args.output_root)
    aggregate_rows: list[dict] = []
    per_event_rows: list[dict] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        command = [
            sys.executable, "-m", "src.run_input_ablation",
            "--fused_dir", args.fused_dir,
            "--output_root", str(seed_root),
            "--variants", args.variants,
            "--input_len", str(args.input_len),
            "--lead_time", str(args.lead_time),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--hidden", str(args.hidden),
            "--seed", str(seed),
            "--split_seed", str(args.split_seed),
            "--threshold", str(args.threshold),
            "--device", args.device,
            "--experiment_kind", args.experiment_kind,
        ]
        if args.skip_training:
            command.append("--skip_training")
        print("\n>>> " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            result = load_json(seed_root / "input_ablation.json")
            for row in result["rows"]:
                aggregate_rows.append({**row, "seed": seed})
            for row in result["per_event_differences"]:
                per_event_rows.append({**row, "seed": seed})

    if args.dry_run:
        return
    summaries = summarize_multiseed(aggregate_rows)
    baseline_name = variants[0][0]
    bootstrap_rows = []
    for candidate_name, _ in variants[1:]:
        for metric in ("mae", "csi"):
            baseline = aggregate_event_means(per_event_rows, baseline_name, metric)
            candidate = aggregate_event_means(per_event_rows, candidate_name, metric)
            event_ids = sorted(set(baseline) & set(candidate))
            statistics = paired_bootstrap(
                [baseline[event_id] for event_id in event_ids],
                [candidate[event_id] for event_id in event_ids],
                metric=metric,
            )
            bootstrap_rows.append({"baseline": baseline_name, "candidate": candidate_name, **statistics})

    flat_rows = flatten_summaries(summaries)
    write_csv(aggregate_rows, output_root / "per_seed_metrics.csv")
    write_csv(flat_rows, output_root / "multiseed_summary.csv")
    write_csv(bootstrap_rows, output_root / "paired_bootstrap.csv")
    save_json(
        {
            "experiment_kind": args.experiment_kind,
            "seeds": seeds,
            "split_seed": args.split_seed,
            "paired_design": True,
            "threshold_policy": "fixed test threshold supplied to every run",
            "summaries": summaries,
            "paired_bootstrap": bootstrap_rows,
        },
        output_root / "multiseed_summary.json",
    )
    figure_dir = ensure_dir(output_root / "figures")
    plot_multiseed(summaries, figure_dir / "multiseed_error_bars.png")
    plot_bootstrap_intervals(bootstrap_rows, figure_dir / "paired_bootstrap_ci.png")
    print(f"Multi-seed summary: {output_root / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
