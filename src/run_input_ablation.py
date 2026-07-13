from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import CHANNEL_SETS, resolve_channel_names
from .utils import ensure_dir, load_json, save_json


DEFAULT_VARIANTS = "A_legacy13=legacy;B_rain_current=legacy_rain_current;C_rain_accum=legacy_rain_accum"


def parse_variants(value: str) -> list[tuple[str, str]]:
    variants = []
    for item in value.split(";"):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid variant {item!r}; expected name=channel_set")
        name, channel_set = (part.strip() for part in item.split("=", 1))
        if not name or not channel_set:
            raise ValueError(f"Invalid variant {item!r}; name and channel_set are required")
        resolve_channel_names(channel_set)
        variants.append((name, channel_set))
    if len(variants) < 2:
        raise ValueError("At least two input variants are required")
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("Input variant names must be unique")
    return variants


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = set().union(*(row.keys() for row in rows))
    preferred = [
        "variant",
        "channel_set",
        "input_channels",
        "parameter_count",
        "best_epoch",
        "event_id",
        "mae",
        "rmse",
        "csi",
        "f1",
        "far",
        "mae_delta_vs_A",
        "csi_delta_vs_A",
        "csi_outcome_vs_A",
        "mae_delta_vs_baseline",
        "csi_delta_vs_baseline",
        "csi_outcome_vs_baseline",
    ]
    fieldnames = [name for name in preferred if name in fields] + sorted(fields - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_per_event_differences(per_event: dict[str, list[dict]], baseline_name: str) -> list[dict]:
    baseline = {row["event_id"]: row for row in per_event[baseline_name]}
    rows = []
    for variant, variant_rows in per_event.items():
        for row in variant_rows:
            event_id = row["event_id"]
            if event_id not in baseline:
                raise ValueError(f"Event {event_id!r} is absent from baseline variant {baseline_name!r}")
            base = baseline[event_id]
            csi_delta = float(row["csi"]) - float(base["csi"])
            outcome = "win" if csi_delta > 1e-8 else ("loss" if csi_delta < -1e-8 else "tie")
            rows.append(
                {
                    "variant": variant,
                    "event_id": event_id,
                    "mae": float(row["mae"]),
                    "csi": float(row["csi"]),
                    "mae_delta_vs_A": float(row["mae"]) - float(base["mae"]),
                    "csi_delta_vs_A": csi_delta,
                    "csi_outcome_vs_A": outcome,
                    "mae_delta_vs_baseline": float(row["mae"]) - float(base["mae"]),
                    "csi_delta_vs_baseline": csi_delta,
                    "csi_outcome_vs_baseline": outcome,
                }
            )
    return rows


def plot_aggregate(rows: list[dict], path: Path) -> None:
    labels = [row["variant"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    axes[0].bar(x - width / 2, [row["mae"] for row in rows], width, label="MAE", color="#386CB0")
    axes[0].bar(x + width / 2, [row["rmse"] for row in rows], width, label="RMSE", color="#F28E2B")
    axes[0].set_title("Regression Error")
    axes[0].set_ylabel("Lower is better")
    axes[0].legend()
    axes[1].bar(x - width / 2, [row["csi"] for row in rows], width, label="CSI", color="#4E9A51")
    axes[1].bar(x + width / 2, [row["f1"] for row in rows], width, label="F1", color="#D95F5F")
    axes[1].set_title("Flood-mask Skill")
    axes[1].set_ylabel("Higher is better")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_event_deltas(rows: list[dict], baseline_name: str, path: Path) -> None:
    variants = [name for name in dict.fromkeys(row["variant"] for row in rows) if name != baseline_name]
    if not variants:
        return
    mae_values = [[row["mae_delta_vs_A"] for row in rows if row["variant"] == name] for name in variants]
    csi_values = [[row["csi_delta_vs_A"] for row in rows if row["variant"] == name] for name in variants]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].boxplot(mae_values, showmeans=True)
    axes[0].set_xticks(np.arange(1, len(variants) + 1))
    axes[0].set_xticklabels(variants)
    axes[0].axhline(0.0, color="#444444", linewidth=1)
    axes[0].set_title(f"Per-event MAE delta vs {baseline_name}")
    axes[0].set_ylabel("Negative is better")
    axes[1].boxplot(csi_values, showmeans=True)
    axes[1].set_xticks(np.arange(1, len(variants) + 1))
    axes[1].set_xticklabels(variants)
    axes[1].axhline(0.0, color="#444444", linewidth=1)
    axes[1].set_title(f"Per-event CSI delta vs {baseline_name}")
    axes[1].set_ylabel("Positive is better")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    rows: list[dict],
    difference_rows: list[dict],
    path: Path,
    experiment_kind: str,
    title: str = "Rain Input Ablation",
    baseline_name: str = "A",
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Experiment kind: **{experiment_kind}**. All variants use the same event split, seed, budget, and fixed test threshold.",
        "",
        "| Variant | Channels | Parameters | Best epoch | MAE | RMSE | CSI | F1 | FAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['input_channels']} | {row['parameter_count']} | {row['best_epoch']} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | "
            f"{row['csi']:.6f} | {row['f1']:.6f} | {row['far']:.6f} |"
        )
    lines.extend(["", f"## Per-event CSI outcomes versus {baseline_name}", ""])
    for variant in [row["variant"] for row in rows[1:]]:
        outcomes = [row["csi_outcome_vs_baseline"] for row in difference_rows if row["variant"] == variant]
        lines.append(
            f"- {variant}: wins={outcomes.count('win')}, ties={outcomes.count('tie')}, losses={outcomes.count('loss')}"
        )
    lines.extend(
        [
            "",
            "These results apply only to this synthetic experiment configuration. No improvement is assumed before running the experiment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy inputs with causal rainfall channel additions.")
    parser.add_argument("--fused_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="runs/input_ablation")
    parser.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment_kind", choices=["smoke", "controlled", "formal"], default="controlled")
    parser.add_argument("--report_title", default="Rain Input Ablation")
    parser.add_argument("--report_file", default="RAIN_INPUT_ABLATION.md")
    parser.add_argument("--aggregate_figure", default="rain_input_ablation.png")
    parser.add_argument("--event_figure", default="rain_per_event_deltas.png")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    output_root = ensure_dir(args.output_root)
    figure_dir = ensure_dir(output_root / "figures")
    aggregate_rows = []
    per_event: dict[str, list[dict]] = {}

    for variant, channel_set in variants:
        output_dir = output_root / variant / "outputs"
        checkpoint = output_dir / "checkpoints" / "best.pt"
        if args.force_retrain or not checkpoint.exists():
            if args.skip_training:
                raise FileNotFoundError(f"Missing checkpoint for {variant}: {checkpoint}")
            train_command = [
                sys.executable,
                "-m",
                "src.train",
                "--fused_dir",
                args.fused_dir,
                "--output_dir",
                str(output_dir),
                "--input_channels",
                channel_set,
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
                str(args.num_layers),
                "--dropout",
                str(args.dropout),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
                "--split_seed",
                str(split_seed),
                "--threshold",
                str(args.threshold),
                "--loss_threshold",
                str(args.loss_threshold),
                "--early_stop_patience",
                str(args.early_stop_patience),
                "--device",
                args.device,
                "--no-progress",
            ]
            run_command(train_command, args.dry_run)
        elif not args.dry_run:
            print(f"Reusing checkpoint: {checkpoint}")

        evaluate_command = [
            sys.executable,
            "-m",
            "src.evaluate",
            "--fused_dir",
            args.fused_dir,
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--batch_size",
            str(args.batch_size),
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
        ]
        run_command(evaluate_command, args.dry_run)
        if args.dry_run:
            continue

        metrics = load_json(output_dir / "metrics" / "eval_metrics.json")
        aggregate_rows.append(
            {
                "variant": variant,
                "channel_set": channel_set,
                "input_channels": len(resolve_channel_names(channel_set)),
                "parameter_count": int(metrics["parameter_count"]),
                "best_epoch": int(metrics["best_epoch"]),
                "channel_names": ",".join(resolve_channel_names(channel_set)),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "csi": float(metrics["csi"]),
                "f1": float(metrics["f1"]),
                "far": float(metrics["far"]),
                "threshold": float(metrics["threshold"]),
                "checkpoint": str(checkpoint),
            }
        )
        per_event[variant] = load_json(output_dir / "metrics" / "per_event_metrics.json")["rows"]

    if not aggregate_rows:
        return
    baseline_name = variants[0][0]
    difference_rows = build_per_event_differences(per_event, baseline_name)
    write_csv(aggregate_rows, output_root / "input_ablation.csv")
    write_csv(difference_rows, output_root / "per_event_differences.csv")
    save_json(
        {
            "experiment_kind": args.experiment_kind,
            "fused_dir": args.fused_dir,
            "seed": args.seed,
            "split_seed": split_seed,
            "threshold": args.threshold,
            "rows": aggregate_rows,
            "per_event_differences": difference_rows,
            "available_channel_sets": {name: list(channels) for name, channels in CHANNEL_SETS.items()},
        },
        output_root / "input_ablation.json",
    )
    plot_aggregate(aggregate_rows, figure_dir / args.aggregate_figure)
    plot_event_deltas(difference_rows, baseline_name, figure_dir / args.event_figure)
    write_report(
        aggregate_rows,
        difference_rows,
        output_root / args.report_file,
        args.experiment_kind,
        args.report_title,
        baseline_name,
    )
    best = max(aggregate_rows, key=lambda row: (row["csi"], -row["mae"]))
    print(f"\nBest by CSI: {best['variant']} CSI={best['csi']:.4f} MAE={best['mae']:.4f}")
    print(f"Report: {output_root / args.report_file}")


if __name__ == "__main__":
    main()
