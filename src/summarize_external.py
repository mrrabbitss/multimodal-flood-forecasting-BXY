from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .model_variants import model_display_name
from .utils import ensure_dir, load_json, save_json


MODEL_ORDER = ("convlstm", "convlstm_attention", "cnn_temporal_transformer")
MODEL_COLORS = {
    "convlstm": "#2F6B9A",
    "convlstm_attention": "#D97732",
    "cnn_temporal_transformer": "#4C956C",
}
PERSISTENCE_COLOR = "#555555"
DATASET_LABELS = {"larno_ukea": "LarNO UKEA", "urbanflood24": "UrbanFlood24"}


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])
    extra = sorted(set().union(*(row.keys() for row in rows)) - set(fields))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)


def _safe_reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference if abs(reference) > 1e-12 else 0.0


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = [float(value) for value in values]
    return mean(items), pstdev(items) if len(items) > 1 else 0.0


def load_external_runs(input_root: str | Path) -> list[dict]:
    root = Path(input_root)
    paths = sorted(root.rglob("test_metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No test_metrics.json files found under {root}")
    runs = []
    seen = set()
    for path in paths:
        result = load_json(path)
        if result.get("schema_version") != "external_physical_v1":
            continue
        key = (
            result.get("dataset"),
            result.get("location"),
            result.get("model_type"),
            int(result.get("seed", -1)),
            int(result.get("split_seed", -1)),
        )
        if key in seen:
            raise ValueError(f"Duplicate external benchmark run {key}: {path}")
        seen.add(key)
        result["_metrics_path"] = path.relative_to(root).as_posix()
        for split in ("train", "validation", "test"):
            manifest_path = path.parent / f"{split}_manifest.json"
            if manifest_path.exists():
                manifest = load_json(manifest_path)
                result[f"{split}_samples"] = int(manifest.get("num_samples", 0))
                result[f"{split}_events_count"] = int(manifest.get("num_events", 0))
        runs.append(result)
    if not runs:
        raise ValueError(f"No external_physical_v1 metrics found under {root}")
    validate_external_runs(runs)
    return runs


def validate_external_runs(runs: Sequence[dict]) -> None:
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        required = {
            "dataset",
            "location",
            "model_type",
            "seed",
            "split_seed",
            "per_horizon",
            "persistence_per_horizon",
            "test_events",
        }
        missing = sorted(required - set(run))
        if missing:
            raise ValueError(f"Missing fields {missing} in {run.get('_metrics_path', '<memory>')}")
        if len(run["per_horizon"]) != len(run["persistence_per_horizon"]):
            raise ValueError("Model and persistence horizon counts differ")
        by_group[(str(run["dataset"]), str(run["location"]))].append(run)

    for group, group_runs in by_group.items():
        reference = group_runs[0]
        signature = (
            int(reference["split_seed"]),
            tuple(reference["test_events"]),
            tuple(row["lead_minutes"] for row in reference["per_horizon"]),
            tuple(float(value) for value in reference.get("thresholds_m", [])),
            float(reference.get("primary_threshold_m", 0.10)),
        )
        for run in group_runs[1:]:
            candidate = (
                int(run["split_seed"]),
                tuple(run["test_events"]),
                tuple(row["lead_minutes"] for row in run["per_horizon"]),
                tuple(float(value) for value in run.get("thresholds_m", [])),
                float(run.get("primary_threshold_m", 0.10)),
            )
            if candidate != signature:
                raise ValueError(f"Inconsistent evaluation protocol for dataset/location {group}")


def flatten_external_runs(runs: Sequence[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows: list[dict] = []
    horizon_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for run in runs:
        common = {
            "dataset": str(run["dataset"]),
            "location": str(run["location"]),
            "model_type": str(run["model_type"]),
            "model_label": str(run.get("model_label", model_display_name(str(run["model_type"])))),
            "seed": int(run["seed"]),
            "split_seed": int(run["split_seed"]),
        }
        per_horizon = run["per_horizon"]
        persistence = run["persistence_per_horizon"]
        for model_row, baseline_row in zip(per_horizon, persistence):
            if int(model_row["lead_minutes"]) != int(baseline_row["lead_minutes"]):
                raise ValueError("Model and persistence lead times differ")
            row = {
                **common,
                "lead_steps": int(model_row["lead_steps"]),
                "lead_minutes": int(model_row["lead_minutes"]),
                "mae_cm": float(model_row["mae_cm"]),
                "rmse_cm": float(model_row["rmse_cm"]),
                "wet_mae_cm": 100.0 * float(model_row["wet_mae_m"]),
                "peak_depth_mae_cm": 100.0 * float(model_row["peak_depth_mae_m"]),
                "csi": float(model_row["csi"]),
                "pod": float(model_row["pod"]),
                "far": float(model_row["far"]),
                "persistence_mae_cm": float(baseline_row["mae_cm"]),
                "persistence_rmse_cm": float(baseline_row["rmse_cm"]),
                "persistence_csi": float(baseline_row["csi"]),
                "persistence_pod": float(baseline_row["pod"]),
                "persistence_far": float(baseline_row["far"]),
            }
            row["mae_reduction_pct"] = _safe_reduction(row["persistence_mae_cm"], row["mae_cm"])
            row["rmse_reduction_pct"] = _safe_reduction(row["persistence_rmse_cm"], row["rmse_cm"])
            row["csi_gain"] = row["csi"] - row["persistence_csi"]
            horizon_rows.append(row)

            for threshold, values in model_row.get("threshold_metrics", {}).items():
                baseline_values = baseline_row.get("threshold_metrics", {}).get(threshold, {})
                threshold_rows.append(
                    {
                        **common,
                        "lead_minutes": int(model_row["lead_minutes"]),
                        "threshold_m": float(threshold),
                        "csi": float(values["csi"]),
                        "pod": float(values["pod"]),
                        "far": float(values["far"]),
                        "persistence_csi": float(baseline_values.get("csi", np.nan)),
                        "persistence_pod": float(baseline_values.get("pod", np.nan)),
                        "persistence_far": float(baseline_values.get("far", np.nan)),
                    }
                )

        model_mae = mean(float(row["mae_cm"]) for row in per_horizon)
        baseline_mae = mean(float(row["mae_cm"]) for row in persistence)
        model_rmse = mean(float(row["rmse_cm"]) for row in per_horizon)
        baseline_rmse = mean(float(row["rmse_cm"]) for row in persistence)
        model_csi = mean(float(row["csi"]) for row in per_horizon)
        baseline_csi = mean(float(row["csi"]) for row in persistence)
        run_rows.append(
            {
                **common,
                "best_epoch": int(run.get("best_epoch", 0)),
                "mean_mae_cm": model_mae,
                "mean_rmse_cm": model_rmse,
                "mean_csi": model_csi,
                "mean_pod": mean(float(row["pod"]) for row in per_horizon),
                "mean_far": mean(float(row["far"]) for row in per_horizon),
                "persistence_mean_mae_cm": baseline_mae,
                "persistence_mean_rmse_cm": baseline_rmse,
                "persistence_mean_csi": baseline_csi,
                "mae_reduction_pct": _safe_reduction(baseline_mae, model_mae),
                "rmse_reduction_pct": _safe_reduction(baseline_rmse, model_rmse),
                "csi_gain": model_csi - baseline_csi,
                "parameter_count": int(run.get("parameter_count", 0)),
                "latency_ms_per_sample": float(run.get("latency_ms_per_sample", 0.0)),
                "peak_cuda_memory_mb": float(run.get("peak_cuda_memory_mb", 0.0)),
                "runtime_sec": float(run.get("runtime_sec", 0.0)),
                "train_samples": int(run.get("train_samples", 0)),
                "validation_samples": int(run.get("validation_samples", 0)),
                "test_samples": int(run.get("test_samples", run.get("samples", 0))),
                "metrics_path": str(run.get("_metrics_path", "")),
            }
        )
    return run_rows, horizon_rows, threshold_rows


def summarize_models(run_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["location"], row["model_type"])].append(row)
    summaries = []
    metrics = (
        "mean_mae_cm",
        "mean_rmse_cm",
        "mean_csi",
        "mean_pod",
        "mean_far",
        "mae_reduction_pct",
        "rmse_reduction_pct",
        "csi_gain",
        "latency_ms_per_sample",
        "peak_cuda_memory_mb",
        "runtime_sec",
    )
    for (dataset, location, model_type), rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": model_display_name(model_type),
            "seed_count": len(rows),
            "seeds": ",".join(str(value) for value in sorted(int(row["seed"]) for row in rows)),
            "parameter_count": int(rows[0]["parameter_count"]),
            "train_samples": int(rows[0]["train_samples"]),
            "validation_samples": int(rows[0]["validation_samples"]),
            "test_samples": int(rows[0]["test_samples"]),
        }
        for metric in metrics:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = _mean_std(row[metric] for row in rows)
        summaries.append(summary)
    return summaries


def summarize_datasets(run_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["model_type"])].append(row)
    output = []
    metrics = (
        "mean_mae_cm",
        "mean_rmse_cm",
        "mean_csi",
        "mean_pod",
        "mean_far",
        "mae_reduction_pct",
        "rmse_reduction_pct",
        "csi_gain",
        "latency_ms_per_sample",
        "peak_cuda_memory_mb",
    )
    for (dataset, model_type), rows in sorted(grouped.items()):
        record = {
            "dataset": dataset,
            "model_type": model_type,
            "model_label": model_display_name(model_type),
            "run_count": len(rows),
            "location_count": len({row["location"] for row in rows}),
            "locations": ",".join(sorted({str(row["location"]) for row in rows})),
            "seed_count": len({int(row["seed"]) for row in rows}),
            "seeds": ",".join(str(value) for value in sorted({int(row["seed"]) for row in rows})),
            "parameter_count": int(rows[0]["parameter_count"]),
        }
        for metric in metrics:
            record[f"{metric}_mean"], record[f"{metric}_std"] = _mean_std(row[metric] for row in rows)
        output.append(record)
    return output


def summarize_horizons(horizon_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in horizon_rows:
        grouped[(row["dataset"], row["location"], row["model_type"], int(row["lead_minutes"]))].append(row)
    summaries = []
    metrics = (
        "mae_cm",
        "rmse_cm",
        "csi",
        "pod",
        "far",
        "mae_reduction_pct",
        "rmse_reduction_pct",
        "csi_gain",
        "persistence_mae_cm",
        "persistence_rmse_cm",
        "persistence_csi",
    )
    for (dataset, location, model_type, lead_minutes), rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": model_display_name(model_type),
            "lead_minutes": lead_minutes,
            "seed_count": len(rows),
        }
        for metric in metrics:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = _mean_std(row[metric] for row in rows)
        summaries.append(summary)
    return summaries


def summarize_thresholds(threshold_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int, float], list[dict]] = defaultdict(list)
    for row in threshold_rows:
        key = (
            row["dataset"],
            row["location"],
            row["model_type"],
            int(row["lead_minutes"]),
            float(row["threshold_m"]),
        )
        grouped[key].append(row)
    output = []
    for (dataset, location, model_type, lead_minutes, threshold), rows in sorted(grouped.items()):
        record = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": model_display_name(model_type),
            "lead_minutes": lead_minutes,
            "threshold_m": threshold,
            "seed_count": len(rows),
        }
        for metric in ("csi", "pod", "far", "persistence_csi", "persistence_pod", "persistence_far"):
            record[f"{metric}_mean"], record[f"{metric}_std"] = _mean_std(row[metric] for row in rows)
        output.append(record)
    return output


def _ordered_models(rows: Sequence[dict]) -> list[str]:
    available = {str(row["model_type"]) for row in rows}
    return [model for model in MODEL_ORDER if model in available] + sorted(available - set(MODEL_ORDER))


def _group_rows(rows: Sequence[dict]) -> dict[tuple[str, str], list[dict]]:
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        output[(str(row["dataset"]), str(row["location"]))].append(row)
    return output


def _slug(dataset: str, location: str) -> str:
    return f"{dataset}_{location}".replace("/", "_").replace(" ", "_")


def _group_label(dataset: str, location: str) -> str:
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    if dataset == "larno_ukea" and location == "ukea":
        return dataset_label
    location_label = "UKEA" if location == "ukea" else location.replace("location", "Location ")
    return f"{dataset_label} / {location_label}"


def _plot_overview(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    lookup = {row["model_type"]: row for row in rows}
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specifications = (
        ("mean_mae_cm", "Mean MAE (cm)", "lower"),
        ("mean_rmse_cm", "Mean RMSE (cm)", "lower"),
        ("mean_csi", "Mean CSI at 0.10 m", "higher"),
    )
    for axis, (metric, label, direction) in zip(axes, specifications):
        values = [lookup[model][f"{metric}_mean"] for model in models]
        errors = [lookup[model][f"{metric}_std"] for model in models]
        axis.bar(x, values, yerr=errors, capsize=4, color=[MODEL_COLORS.get(model, "#777777") for model in models])
        axis.set_title(f"{label}\n({direction} is better)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, [model_display_name(model) for model in models], rotation=16, ha="right")
        if metric == "mean_csi":
            axis.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_horizons(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = (("mae_cm", "MAE (cm)"), ("rmse_cm", "RMSE (cm)"), ("csi", "CSI at 0.10 m"))
    for model in models:
        model_rows = sorted((row for row in rows if row["model_type"] == model), key=lambda row: row["lead_minutes"])
        leads = np.asarray([row["lead_minutes"] for row in model_rows])
        for axis, (metric, ylabel) in zip(axes, specs):
            values = np.asarray([row[f"{metric}_mean"] for row in model_rows])
            deviations = np.asarray([row[f"{metric}_std"] for row in model_rows])
            axis.plot(leads, values, marker="o", color=MODEL_COLORS.get(model), label=model_display_name(model))
            axis.fill_between(leads, values - deviations, values + deviations, color=MODEL_COLORS.get(model), alpha=0.12)
    reference = sorted((row for row in rows if row["model_type"] == models[0]), key=lambda row: row["lead_minutes"])
    leads = [row["lead_minutes"] for row in reference]
    for axis, (metric, ylabel) in zip(axes, specs):
        axis.plot(leads, [row[f"persistence_{metric}_mean"] for row in reference], marker="s", linestyle="--", color=PERSISTENCE_COLOR, label="Persistence")
        axis.set_xlabel("Forecast lead time (min)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_skill_gain(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    leads = sorted({int(row["lead_minutes"]) for row in rows})
    x = np.arange(len(leads))
    width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for index, model in enumerate(models):
        model_rows = {int(row["lead_minutes"]): row for row in rows if row["model_type"] == model}
        offset = (index - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, [model_rows[lead]["mae_reduction_pct_mean"] for lead in leads], width, color=MODEL_COLORS.get(model), label=model_display_name(model))
        axes[1].bar(x + offset, [100.0 * model_rows[lead]["csi_gain_mean"] for lead in leads], width, color=MODEL_COLORS.get(model), label=model_display_name(model))
    axes[0].set_ylabel("MAE reduction vs persistence (%)")
    axes[1].set_ylabel("CSI gain vs persistence (percentage points)")
    for axis in axes:
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_xlabel("Forecast lead time (min)")
        axis.set_xticks(x, leads)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_thresholds(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for model in models:
        model_rows = [row for row in rows if row["model_type"] == model]
        thresholds = sorted({float(row["threshold_m"]) for row in model_rows})
        csi = [mean(row["csi_mean"] for row in model_rows if float(row["threshold_m"]) == threshold) for threshold in thresholds]
        pod = [mean(row["pod_mean"] for row in model_rows if float(row["threshold_m"]) == threshold) for threshold in thresholds]
        axes[0].plot(thresholds, csi, marker="o", color=MODEL_COLORS.get(model), label=model_display_name(model))
        axes[1].plot(thresholds, pod, marker="o", color=MODEL_COLORS.get(model), label=model_display_name(model))
    reference = [row for row in rows if row["model_type"] == models[0]]
    thresholds = sorted({float(row["threshold_m"]) for row in reference})
    axes[0].plot(thresholds, [mean(row["persistence_csi_mean"] for row in reference if float(row["threshold_m"]) == value) for value in thresholds], marker="s", linestyle="--", color=PERSISTENCE_COLOR, label="Persistence")
    axes[1].plot(thresholds, [mean(row["persistence_pod_mean"] for row in reference if float(row["threshold_m"]) == value) for value in thresholds], marker="s", linestyle="--", color=PERSISTENCE_COLOR, label="Persistence")
    axes[0].set_ylabel("CSI averaged over lead times")
    axes[1].set_ylabel("POD averaged over lead times")
    for axis in axes:
        axis.set_xlabel("Flood threshold (m)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_efficiency(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    lookup = {row["model_type"]: row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model in models:
        row = lookup[model]
        axes[0].scatter(row["latency_ms_per_sample_mean"], row["mean_csi_mean"], s=max(70, row["peak_cuda_memory_mb_mean"] * 1.4), color=MODEL_COLORS.get(model), alpha=0.82, label=model_display_name(model))
        axes[0].annotate(model_display_name(model), (row["latency_ms_per_sample_mean"], row["mean_csi_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Latency (ms/sample; lower is better)")
    axes[0].set_ylabel("Mean CSI (higher is better)")
    axes[0].grid(alpha=0.25)
    x = np.arange(len(models))
    memory = [lookup[model]["peak_cuda_memory_mb_mean"] for model in models]
    axes[1].bar(x, memory, color=[MODEL_COLORS.get(model) for model in models])
    axes[1].set_xticks(x, [model_display_name(model) for model in models], rotation=16, ha="right")
    axes[1].set_ylabel("Peak CUDA memory (MB)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Marker area at left also follows VRAM")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    leads = sorted({int(row["lead_minutes"]) for row in rows})
    lookup = {(row["model_type"], int(row["lead_minutes"])): row for row in rows}
    matrix = np.asarray([[lookup[(model, lead)]["mae_reduction_pct_mean"] for lead in leads] for model in models])
    fig, axis = plt.subplots(figsize=(8, 1.5 + 0.7 * len(models)))
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.1f}%", ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(leads)), [f"{lead} min" for lead in leads])
    axis.set_yticks(range(len(models)), [model_display_name(model) for model in models])
    axis.set_title(f"{title}\nMAE reduction relative to persistence")
    fig.colorbar(image, ax=axis, label="MAE reduction (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_dataset(rows: Sequence[dict], path: Path) -> None:
    groups = sorted({(str(row["dataset"]), str(row["location"])) for row in rows})
    models = _ordered_models(rows)
    lookup = {(row["dataset"], row["location"], row["model_type"]): row for row in rows}
    x = np.arange(len(groups))
    width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        values = [lookup[(*group, model)] for group in groups]
        axes[0].bar(x + offset, [row["mae_reduction_pct_mean"] for row in values], width, color=MODEL_COLORS.get(model), label=model_display_name(model))
        axes[1].bar(x + offset, [100.0 * row["csi_gain_mean"] for row in values], width, color=MODEL_COLORS.get(model), label=model_display_name(model))
    labels = [_group_label(*group) for group in groups]
    axes[0].set_ylabel("Mean MAE reduction vs persistence (%)")
    axes[1].set_ylabel("Mean CSI gain vs persistence (percentage points)")
    for axis in axes:
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=14, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Cross-dataset generalization relative to persistence")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_pm(value: float, deviation: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} +/- {deviation:.{digits}f}"


def _write_report(
    model_rows: Sequence[dict],
    horizon_rows: Sequence[dict],
    path: Path,
    figure_paths: Sequence[Path],
    configs: Sequence[dict],
) -> None:
    config_lookup = {(str(config["dataset"]), str(config["location"])): config for config in configs}
    lines = [
        "# External Physical-Data Benchmark",
        "",
        "This report evaluates copied model variants on physical flood-simulation datasets. Existing Conv-LSTM source files and checkpoints are not modified.",
        "",
        "## Protocol",
        "",
        "- Common resolution: 8 m / 5 min",
        "- Forecast horizons: 5, 15, 30, and 60 minutes",
        "- Event-disjoint train/validation/test splits",
        "- Physical depth output in metres; MAE/RMSE reported in centimetres",
        "- Flood skill at 0.10 m, plus threshold sensitivity at 0.05/0.10/0.20/0.30 m",
        "- Persistence is evaluated on exactly the same test pixels",
        "",
    ]
    for (dataset, location), rows in _group_rows(model_rows).items():
        lines.extend([f"## {_group_label(dataset, location)}", ""])
        minimum_seeds = min(int(row["seed_count"]) for row in rows)
        if minimum_seeds < 3:
            lines.extend([
                "> Status: pilot evidence only. Fewer than three seeds are available for this dataset/location; do not treat its model ranking as statistically stable.",
                "",
            ])
        config = config_lookup.get((dataset, location), {}).get("configuration", {})
        reference = rows[0]
        lines.extend([
            f"- Run budget: {minimum_seeds} seeds, {config.get('epochs', 'unknown')} epochs, batch size {config.get('batch_size', 'unknown')}",
            f"- Samples per run: {reference['train_samples']} train / {reference['validation_samples']} validation / {reference['test_samples']} test",
            f"- Sampling caps per event: train={config.get('max_train_samples_per_event', 'unknown')}, evaluation={config.get('max_eval_samples_per_event', 'unknown')} (`0` means all available)",
            f"- Model width / learning rate: {config.get('hidden', 'unknown')} hidden channels / {config.get('lr', 'unknown')}",
            "",
        ])
        lines.extend(["| Model | Seeds | MAE (cm) | RMSE (cm) | CSI | MAE gain vs persistence | CSI gain | Latency (ms/sample) | VRAM (MB) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in sorted(rows, key=lambda item: _ordered_models(rows).index(item["model_type"])):
            lines.append(
                f"| {row['model_label']} | {row['seed_count']} | "
                f"{_format_pm(row['mean_mae_cm_mean'], row['mean_mae_cm_std'])} | "
                f"{_format_pm(row['mean_rmse_cm_mean'], row['mean_rmse_cm_std'])} | "
                f"{_format_pm(row['mean_csi_mean'], row['mean_csi_std'], 4)} | "
                f"{row['mae_reduction_pct_mean']:.1f}% | {100.0 * row['csi_gain_mean']:.1f} pp | "
                f"{row['latency_ms_per_sample_mean']:.2f} | {row['peak_cuda_memory_mb_mean']:.1f} |"
            )
        group_horizons = [row for row in horizon_rows if row["dataset"] == dataset and row["location"] == location]
        best_mae = min(rows, key=lambda row: row["mean_mae_cm_mean"])
        best_csi = max(rows, key=lambda row: row["mean_csi_mean"])
        long_rows = [row for row in group_horizons if int(row["lead_minutes"]) == max(int(value["lead_minutes"]) for value in group_horizons)]
        best_long = max(long_rows, key=lambda row: row["csi_mean"])
        lines.extend([
            "",
            f"- Lowest average MAE: **{best_mae['model_label']}** ({best_mae['mean_mae_cm_mean']:.3f} cm).",
            f"- Highest average CSI: **{best_csi['model_label']}** ({best_csi['mean_csi_mean']:.4f}).",
            f"- Best longest-horizon CSI: **{best_long['model_label']}** ({best_long['csi_mean']:.4f} at {best_long['lead_minutes']} min).",
            "",
        ])
    lines.extend(["## Figures", ""])
    for figure in figure_paths:
        lines.append(f"![{figure.stem}](figures/{figure.name})")
        lines.append("")
    lines.extend([
        "## Interpretation Rules",
        "",
        "Positive MAE reduction and positive CSI gain mean the learned model beats persistence. Model-to-model claims should rely on multi-seed means and deviations, not a single run. Latency and VRAM are device-specific and are intended for relative comparison on the recorded GPU.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_external_results(input_root: str | Path, output_dir: str | Path) -> dict:
    runs = load_external_runs(input_root)
    run_rows, horizon_rows, threshold_rows = flatten_external_runs(runs)
    model_rows = summarize_models(run_rows)
    dataset_rows = summarize_datasets(run_rows)
    horizon_summary = summarize_horizons(horizon_rows)
    threshold_summary = summarize_thresholds(threshold_rows)
    output = ensure_dir(output_dir)
    figures = ensure_dir(output / "figures")
    configs = [load_json(path) for path in sorted(Path(input_root).rglob("benchmark_config.json"))]
    _write_csv(run_rows, output / "external_per_run.csv")
    _write_csv(horizon_rows, output / "external_per_horizon.csv")
    _write_csv(threshold_rows, output / "external_per_threshold.csv")
    _write_csv(model_rows, output / "external_model_summary.csv")
    _write_csv(dataset_rows, output / "external_dataset_summary.csv")
    _write_csv(horizon_summary, output / "external_horizon_summary.csv")
    _write_csv(threshold_summary, output / "external_threshold_summary.csv")

    figure_paths = []
    model_groups = _group_rows(model_rows)
    horizon_groups = _group_rows(horizon_summary)
    threshold_groups = _group_rows(threshold_summary)
    for group, group_model_rows in model_groups.items():
        dataset, location = group
        slug = _slug(dataset, location)
        title = f"{_group_label(dataset, location)}: external physical benchmark"
        group_horizons = horizon_groups[group]
        group_thresholds = threshold_groups[group]
        plotters = (
            (_plot_overview, group_model_rows, figures / f"{slug}_model_overview.png"),
            (_plot_horizons, group_horizons, figures / f"{slug}_horizon_curves.png"),
            (_plot_skill_gain, group_horizons, figures / f"{slug}_skill_gain.png"),
            (_plot_thresholds, group_thresholds, figures / f"{slug}_threshold_sensitivity.png"),
            (_plot_efficiency, group_model_rows, figures / f"{slug}_efficiency.png"),
            (_plot_heatmap, group_horizons, figures / f"{slug}_skill_heatmap.png"),
        )
        for plotter, rows, path in plotters:
            plotter(rows, path, title)
            figure_paths.append(path)

    cross_dataset_path = figures / "cross_dataset_generalization.png"
    _plot_cross_dataset(model_rows, cross_dataset_path)
    figure_paths.insert(0, cross_dataset_path)

    summary = {
        "schema_version": "external_physical_summary_v1",
        "input_root": Path(input_root).as_posix(),
        "run_count": len(runs),
        "datasets": sorted({str(run["dataset"]) for run in runs}),
        "models": _ordered_models(run_rows),
        "seeds": sorted({int(run["seed"]) for run in runs}),
        "model_summary": model_rows,
        "dataset_summary": dataset_rows,
        "horizon_summary": horizon_summary,
        "threshold_summary": threshold_summary,
        "benchmark_configs": configs,
        "figures": [path.relative_to(output).as_posix() for path in figure_paths],
    }
    save_json(summary, output / "external_benchmark_summary.json")
    _write_report(model_rows, horizon_summary, output / "EXTERNAL_PHYSICAL_BENCHMARK.md", figure_paths, configs)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize external physical-data benchmark runs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = summarize_external_results(args.input_root, args.output_dir)
    print(f"Summarized {summary['run_count']} runs: {Path(args.output_dir) / 'external_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
