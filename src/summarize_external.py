from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .external_models import (
    EXTERNAL_MODEL_TYPES,
    external_model_display_name,
)
from .utils import ensure_dir, load_json, save_json


MODEL_ORDER = EXTERNAL_MODEL_TYPES
MODEL_COLORS = {
    "convlstm": "#2F6B9A",
    "convlstm_attention": "#D97732",
    "cnn_temporal_transformer": "#4C956C",
    "urnn_lite": "#8A5A9E",
    "fno2d_history": "#C84B31",
    "simvp_lite": "#2A9D8F",
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


def _public_benchmark_config(config: dict) -> dict:
    output = dict(config)
    configuration = dict(config.get("configuration", {}))
    for key in ("larno_root", "urban_root"):
        value = str(configuration.get(key, ""))
        if value and Path(value).is_absolute():
            configuration[key] = f"<local-{key.replace('_', '-')}>"
    for key in ("output_root", "summary_dir"):
        value = str(configuration.get(key, ""))
        if value:
            configuration[key] = Path(value).as_posix()
    output["configuration"] = configuration
    return output


def _safe_reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference if abs(reference) > 1e-12 else 0.0


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = [float(value) for value in values if np.isfinite(float(value))]
    if not items:
        return float("nan"), float("nan")
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
        if result.get("schema_version") not in {"external_physical_v1", "external_physical_v2"}:
            continue
        key = (
            result.get("dataset"),
            result.get("location"),
            result.get("model_type"),
            int(result.get("seed", -1)),
            int(result.get("split_seed", -1)),
            result.get("protocol_details", {}).get("prediction_mode", "residual"),
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
                if split == "test":
                    result["_test_manifest_signature"] = (
                        tuple(manifest.get("channel_names", [])),
                        int(manifest.get("input_len", 0)),
                        tuple(int(value) for value in manifest.get("lead_times", [])),
                        int(manifest.get("time_step_minutes", 0)),
                        int(manifest.get("patch_size", 0)),
                        int(manifest.get("patch_stride", 0)),
                        float(manifest.get("depth_scale_m", 0.0)),
                        float(manifest.get("rain_scale_mm_5min", 0.0)),
                        int(manifest.get("num_samples", 0)),
                        manifest.get("sampling_seed"),
                    )
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
            str(reference.get("protocol_details", {}).get("rain_forcing", "past_only")),
            str(reference.get("protocol_details", {}).get("prediction_mode", "residual")),
            int(reference.get("test_samples", reference.get("samples", 0))),
            tuple(reference.get("_test_manifest_signature", ())),
        )
        for run in group_runs[1:]:
            candidate = (
                int(run["split_seed"]),
                tuple(run["test_events"]),
                tuple(row["lead_minutes"] for row in run["per_horizon"]),
                tuple(float(value) for value in run.get("thresholds_m", [])),
                float(run.get("primary_threshold_m", 0.10)),
                str(run.get("protocol_details", {}).get("rain_forcing", "past_only")),
                str(run.get("protocol_details", {}).get("prediction_mode", "residual")),
                int(run.get("test_samples", run.get("samples", 0))),
                tuple(run.get("_test_manifest_signature", ())),
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
            "model_label": str(run.get("model_label", external_model_display_name(str(run["model_type"])))),
            "seed": int(run["seed"]),
            "split_seed": int(run["split_seed"]),
            "rain_forcing": str(run.get("protocol_details", {}).get("rain_forcing", "past_only")),
            "prediction_mode": str(
                run.get("protocol_details", {}).get("prediction_mode", "residual")
            ),
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
                "wet_rmse_cm": 100.0 * float(model_row.get("wet_rmse_m", np.nan)),
                "dry_prediction_mean_cm": 100.0
                * float(model_row.get("dry_prediction_mean_m", np.nan)),
                "peak_depth_mae_cm": 100.0 * float(model_row["peak_depth_mae_m"]),
                "csi": float(model_row["csi"]),
                "pod": float(model_row["pod"]),
                "far": float(model_row["far"]),
                "f1": float(model_row.get("f1", np.nan)),
                "dry_false_positive_rate": float(
                    model_row.get("dry_false_positive_rate", np.nan)
                ),
                "boundary_f1": float(model_row.get("boundary_f1", np.nan)),
                "persistence_mae_cm": float(baseline_row["mae_cm"]),
                "persistence_rmse_cm": float(baseline_row["rmse_cm"]),
                "persistence_wet_mae_cm": 100.0
                * float(baseline_row.get("wet_mae_m", np.nan)),
                "persistence_wet_rmse_cm": 100.0
                * float(baseline_row.get("wet_rmse_m", np.nan)),
                "persistence_dry_prediction_mean_cm": 100.0
                * float(baseline_row.get("dry_prediction_mean_m", np.nan)),
                "persistence_csi": float(baseline_row["csi"]),
                "persistence_pod": float(baseline_row["pod"]),
                "persistence_far": float(baseline_row["far"]),
                "persistence_f1": float(baseline_row.get("f1", np.nan)),
                "persistence_dry_false_positive_rate": float(
                    baseline_row.get("dry_false_positive_rate", np.nan)
                ),
                "persistence_boundary_f1": float(baseline_row.get("boundary_f1", np.nan)),
            }
            for bin_label, bin_values in model_row.get("depth_bin_metrics", {}).items():
                slug = bin_label.replace(".", "p").replace("-", "_").replace("+", "plus")
                row[f"depth_bin_{slug}_mae_cm"] = 100.0 * float(bin_values["mae_m"])
                row[f"depth_bin_{slug}_rmse_cm"] = 100.0 * float(bin_values["rmse_m"])
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
                        "f1": float(values.get("f1", np.nan)),
                        "false_positive_rate": float(values.get("false_positive_rate", np.nan)),
                        "persistence_csi": float(baseline_values.get("csi", np.nan)),
                        "persistence_pod": float(baseline_values.get("pod", np.nan)),
                        "persistence_far": float(baseline_values.get("far", np.nan)),
                        "persistence_f1": float(baseline_values.get("f1", np.nan)),
                        "persistence_false_positive_rate": float(
                            baseline_values.get("false_positive_rate", np.nan)
                        ),
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
                "mean_wet_mae_cm": mean(100.0 * float(row["wet_mae_m"]) for row in per_horizon),
                "mean_wet_rmse_cm": mean(
                    100.0 * float(row.get("wet_rmse_m", np.nan)) for row in per_horizon
                ),
                "mean_dry_prediction_cm": mean(
                    100.0 * float(row.get("dry_prediction_mean_m", np.nan)) for row in per_horizon
                ),
                "mean_boundary_f1": mean(
                    float(row.get("boundary_f1", np.nan)) for row in per_horizon
                ),
                "peak_time_mae_min": float(run.get("peak_time_mae_min", np.nan)),
                "persistence_peak_time_mae_min": float(
                    run.get("persistence_peak_time_mae_min", np.nan)
                ),
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
                "training_time_sec": float(run.get("training_time_sec", np.nan)),
                "epochs_ran": int(run.get("epochs_ran", run.get("best_epoch", 0))),
                "train_samples": int(run.get("train_samples", 0)),
                "validation_samples": int(run.get("validation_samples", 0)),
                "test_samples": int(run.get("test_samples", run.get("samples", 0))),
                "metrics_path": str(run.get("_metrics_path", "")),
            }
        )
    return run_rows, horizon_rows, threshold_rows


def flatten_external_events(runs: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        for event in run.get("per_event", []):
            model_horizons = event["per_horizon"]
            persistence_horizons = event["persistence_per_horizon"]
            model_mae = mean(float(row["mae_cm"]) for row in model_horizons)
            persistence_mae = mean(float(row["mae_cm"]) for row in persistence_horizons)
            model_csi = mean(float(row["csi"]) for row in model_horizons)
            persistence_csi = mean(float(row["csi"]) for row in persistence_horizons)
            rows.append(
                {
                    "dataset": str(run["dataset"]),
                    "location": str(run["location"]),
                    "model_type": str(run["model_type"]),
                    "model_label": str(run.get("model_label", run["model_type"])),
                    "seed": int(run["seed"]),
                    "event_id": str(event["event_id"]),
                    "samples": int(event["samples"]),
                    "mean_mae_cm": model_mae,
                    "mean_rmse_cm": mean(float(row["rmse_cm"]) for row in model_horizons),
                    "mean_csi": model_csi,
                    "mean_wet_rmse_cm": mean(
                        100.0 * float(row.get("wet_rmse_m", np.nan)) for row in model_horizons
                    ),
                    "mean_boundary_f1": mean(
                        float(row.get("boundary_f1", np.nan)) for row in model_horizons
                    ),
                    "peak_time_mae_min": float(event.get("peak_time_mae_min", np.nan)),
                    "persistence_mean_mae_cm": persistence_mae,
                    "persistence_mean_csi": persistence_csi,
                    "persistence_peak_time_mae_min": float(
                        event.get("persistence_peak_time_mae_min", np.nan)
                    ),
                    "mae_reduction_pct": _safe_reduction(persistence_mae, model_mae),
                    "csi_gain": model_csi - persistence_csi,
                }
            )
    return rows


def flatten_external_depth_bins(runs: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        common = {
            "dataset": str(run["dataset"]),
            "location": str(run["location"]),
            "model_type": str(run["model_type"]),
            "model_label": str(run.get("model_label", run["model_type"])),
            "seed": int(run["seed"]),
        }
        for model_horizon, persistence_horizon in zip(
            run["per_horizon"], run["persistence_per_horizon"]
        ):
            persistence_bins = persistence_horizon.get("depth_bin_metrics", {})
            for depth_bin, values in model_horizon.get("depth_bin_metrics", {}).items():
                baseline = persistence_bins.get(depth_bin, {})
                rows.append(
                    {
                        **common,
                        "lead_minutes": int(model_horizon["lead_minutes"]),
                        "depth_bin_m": depth_bin,
                        "pixels": int(values["pixels"]),
                        "mae_cm": 100.0 * float(values["mae_m"]),
                        "rmse_cm": 100.0 * float(values["rmse_m"]),
                        "persistence_mae_cm": 100.0 * float(baseline.get("mae_m", np.nan)),
                        "persistence_rmse_cm": 100.0 * float(
                            baseline.get("rmse_m", np.nan)
                        ),
                    }
                )
    return rows


def summarize_depth_bins(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["dataset"], row["location"], row["model_type"], row["depth_bin_m"])
        ].append(row)
    output: list[dict] = []
    for (dataset, location, model_type, depth_bin), values in sorted(grouped.items()):
        record = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": external_model_display_name(model_type),
            "depth_bin_m": depth_bin,
            "seed_count": len({int(row["seed"]) for row in values}),
        }
        for metric in ("mae_cm", "rmse_cm", "persistence_mae_cm", "persistence_rmse_cm"):
            record[f"{metric}_mean"], record[f"{metric}_std"] = _mean_std(
                row[metric] for row in values
            )
        output.append(record)
    return output


def summarize_event_robustness(event_rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    by_event: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in event_rows:
        by_event[(row["dataset"], row["location"], row["model_type"], row["event_id"])].append(row)
    event_means: list[dict] = []
    metrics = (
        "mean_mae_cm",
        "mean_rmse_cm",
        "mean_csi",
        "mean_wet_rmse_cm",
        "mean_boundary_f1",
        "peak_time_mae_min",
        "mae_reduction_pct",
        "csi_gain",
    )
    for (dataset, location, model_type, event_id), rows in sorted(by_event.items()):
        record = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": external_model_display_name(model_type),
            "event_id": event_id,
            "seed_count": len(rows),
        }
        for metric in metrics:
            record[metric] = _mean_std(row[metric] for row in rows)[0]
        event_means.append(record)

    by_model: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in event_means:
        by_model[(row["dataset"], row["location"], row["model_type"])].append(row)
    summaries: list[dict] = []
    for (dataset, location, model_type), rows in sorted(by_model.items()):
        record = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": external_model_display_name(model_type),
            "event_count": len(rows),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                record[f"{metric}_median"] = float(np.median(values))
                record[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                record[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            else:
                record[f"{metric}_median"] = float("nan")
                record[f"{metric}_q25"] = float("nan")
                record[f"{metric}_q75"] = float("nan")
        summaries.append(record)
    return event_means, summaries


def _paired_permutation_pvalue(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences)]
    count = int(differences.size)
    if count == 0:
        return float("nan")
    observed = abs(float(differences.mean()))
    if count <= 16:
        masks = np.arange(1 << count, dtype=np.uint32)[:, None]
        bits = (masks >> np.arange(count, dtype=np.uint32)) & 1
        signs = 1.0 - 2.0 * bits
    else:
        rng = np.random.default_rng(44)
        signs = rng.choice((-1.0, 1.0), size=(100_000, count))
    permuted = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def _bootstrap_mean_interval(differences: np.ndarray, seed: int) -> tuple[float, float]:
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(20_000, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    return float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))


def compare_models_by_event(event_mean_rows: Sequence[dict]) -> list[dict]:
    grouped = _group_rows(event_mean_rows)
    output: list[dict] = []
    for (dataset, location), rows in grouped.items():
        models = _ordered_models(rows)
        lookup = {(row["model_type"], row["event_id"]): row for row in rows}
        event_ids = sorted({str(row["event_id"]) for row in rows})
        for comparison_index, (model_a, model_b) in enumerate(combinations(models, 2)):
            common_events = [
                event_id
                for event_id in event_ids
                if (model_a, event_id) in lookup and (model_b, event_id) in lookup
            ]
            mae_difference = np.asarray(
                [
                    lookup[(model_a, event_id)]["mean_mae_cm"]
                    - lookup[(model_b, event_id)]["mean_mae_cm"]
                    for event_id in common_events
                ],
                dtype=np.float64,
            )
            csi_difference = np.asarray(
                [
                    lookup[(model_a, event_id)]["mean_csi"]
                    - lookup[(model_b, event_id)]["mean_csi"]
                    for event_id in common_events
                ],
                dtype=np.float64,
            )
            mae_low, mae_high = _bootstrap_mean_interval(mae_difference, 1000 + comparison_index)
            csi_low, csi_high = _bootstrap_mean_interval(csi_difference, 2000 + comparison_index)
            output.append(
                {
                    "dataset": dataset,
                    "location": location,
                    "model_a": model_a,
                    "model_a_label": external_model_display_name(model_a),
                    "model_b": model_b,
                    "model_b_label": external_model_display_name(model_b),
                    "event_count": len(common_events),
                    "mae_delta_a_minus_b_cm": float(mae_difference.mean()),
                    "mae_delta_ci95_low_cm": mae_low,
                    "mae_delta_ci95_high_cm": mae_high,
                    "mae_a_win_rate": float(np.mean(mae_difference < 0.0)),
                    "mae_permutation_p": _paired_permutation_pvalue(mae_difference),
                    "csi_delta_a_minus_b": float(csi_difference.mean()),
                    "csi_delta_ci95_low": csi_low,
                    "csi_delta_ci95_high": csi_high,
                    "csi_a_win_rate": float(np.mean(csi_difference > 0.0)),
                    "csi_permutation_p": _paired_permutation_pvalue(csi_difference),
                }
            )
    return output


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
        "mean_wet_mae_cm",
        "mean_wet_rmse_cm",
        "mean_dry_prediction_cm",
        "mean_boundary_f1",
        "peak_time_mae_min",
        "persistence_peak_time_mae_min",
        "mae_reduction_pct",
        "rmse_reduction_pct",
        "csi_gain",
        "latency_ms_per_sample",
        "peak_cuda_memory_mb",
        "runtime_sec",
        "training_time_sec",
        "epochs_ran",
    )
    for (dataset, location, model_type), rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": external_model_display_name(model_type),
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
        "mean_wet_mae_cm",
        "mean_wet_rmse_cm",
        "mean_dry_prediction_cm",
        "mean_boundary_f1",
        "peak_time_mae_min",
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
            "model_label": external_model_display_name(model_type),
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
        "f1",
        "wet_mae_cm",
        "wet_rmse_cm",
        "dry_prediction_mean_cm",
        "dry_false_positive_rate",
        "boundary_f1",
        "mae_reduction_pct",
        "rmse_reduction_pct",
        "csi_gain",
        "persistence_mae_cm",
        "persistence_rmse_cm",
        "persistence_csi",
        "persistence_wet_mae_cm",
        "persistence_wet_rmse_cm",
        "persistence_dry_prediction_mean_cm",
        "persistence_f1",
        "persistence_dry_false_positive_rate",
        "persistence_boundary_f1",
    )
    for (dataset, location, model_type, lead_minutes), rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "location": location,
            "model_type": model_type,
            "model_label": external_model_display_name(model_type),
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
            "model_label": external_model_display_name(model_type),
            "lead_minutes": lead_minutes,
            "threshold_m": threshold,
            "seed_count": len(rows),
        }
        for metric in (
            "csi",
            "pod",
            "far",
            "f1",
            "false_positive_rate",
            "persistence_csi",
            "persistence_pod",
            "persistence_far",
            "persistence_f1",
            "persistence_false_positive_rate",
        ):
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
        axis.set_xticks(x, [external_model_display_name(model) for model in models], rotation=16, ha="right")
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
            axis.plot(leads, values, marker="o", color=MODEL_COLORS.get(model), label=external_model_display_name(model))
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
        axes[0].bar(x + offset, [model_rows[lead]["mae_reduction_pct_mean"] for lead in leads], width, color=MODEL_COLORS.get(model), label=external_model_display_name(model))
        axes[1].bar(x + offset, [100.0 * model_rows[lead]["csi_gain_mean"] for lead in leads], width, color=MODEL_COLORS.get(model), label=external_model_display_name(model))
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
        axes[0].plot(thresholds, csi, marker="o", color=MODEL_COLORS.get(model), label=external_model_display_name(model))
        axes[1].plot(thresholds, pod, marker="o", color=MODEL_COLORS.get(model), label=external_model_display_name(model))
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
        axes[0].scatter(row["latency_ms_per_sample_mean"], row["mean_csi_mean"], s=max(70, row["peak_cuda_memory_mb_mean"] * 1.4), color=MODEL_COLORS.get(model), alpha=0.82, label=external_model_display_name(model))
        axes[0].annotate(external_model_display_name(model), (row["latency_ms_per_sample_mean"], row["mean_csi_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Latency (ms/sample; lower is better)")
    axes[0].set_ylabel("Mean CSI (higher is better)")
    axes[0].grid(alpha=0.25)
    x = np.arange(len(models))
    memory = [lookup[model]["peak_cuda_memory_mb_mean"] for model in models]
    axes[1].bar(x, memory, color=[MODEL_COLORS.get(model) for model in models])
    axes[1].set_xticks(x, [external_model_display_name(model) for model in models], rotation=16, ha="right")
    axes[1].set_ylabel("Peak CUDA memory (MB)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Marker area at left also follows VRAM")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_physical_diagnostics(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    lookup = {row["model_type"]: row for row in rows}
    x = np.arange(len(models))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specifications = (
        ("mean_wet_rmse_cm", "Wet-area RMSE (cm)", "lower is better"),
        ("mean_dry_prediction_cm", "Predicted depth on dry cells (cm)", "lower is better"),
        ("mean_boundary_f1", "Flood-boundary F1", "higher is better"),
    )
    for axis, (metric, label, direction) in zip(axes, specifications):
        values = [lookup[model][f"{metric}_mean"] for model in models]
        errors = [lookup[model][f"{metric}_std"] for model in models]
        axis.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=[MODEL_COLORS.get(model, "#777777") for model in models],
        )
        axis.set_title(f"{label}\n({direction})")
        axis.set_xticks(
            x,
            [external_model_display_name(model) for model in models],
            rotation=18,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_ylim(0, 1)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_event_robustness(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    labels = [external_model_display_name(model) for model in models]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    metric_specs = (
        ("mae_reduction_pct", "Per-event MAE reduction vs persistence (%)"),
        ("csi_gain", "Per-event CSI gain vs persistence"),
    )
    for axis, (metric, ylabel) in zip(axes, metric_specs):
        values = [
            [float(row[metric]) for row in rows if row["model_type"] == model]
            for model in models
        ]
        boxes = axis.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, model in zip(boxes["boxes"], models):
            patch.set_facecolor(MODEL_COLORS.get(model, "#777777"))
            patch.set_alpha(0.72)
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"{title}: event-level robustness")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_depth_bins(rows: Sequence[dict], path: Path, title: str) -> None:
    models = _ordered_models(rows)
    preferred_order = ["0.00-0.10", "0.10-0.30", "0.30-0.50", "0.50+"]
    available = {str(row["depth_bin_m"]) for row in rows}
    bins = [value for value in preferred_order if value in available]
    bins.extend(sorted(available - set(bins)))
    lookup = {(row["model_type"], row["depth_bin_m"]): row for row in rows}
    x = np.arange(len(bins))
    width = 0.78 / max(len(models), 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        for axis, metric in zip(axes, ("mae_cm", "rmse_cm")):
            axis.bar(
                x + offset,
                [lookup[(model, depth_bin)][f"{metric}_mean"] for depth_bin in bins],
                width,
                color=MODEL_COLORS.get(model),
                label=external_model_display_name(model),
            )
    reference = [lookup[(models[0], depth_bin)] for depth_bin in bins]
    for axis, metric, ylabel in zip(
        axes,
        ("mae_cm", "rmse_cm"),
        ("MAE (cm)", "RMSE (cm)"),
    ):
        axis.plot(
            x,
            [row[f"persistence_{metric}_mean"] for row in reference],
            color=PERSISTENCE_COLOR,
            marker="s",
            linestyle="--",
            label="Persistence",
        )
        axis.set_xticks(x, [f"{depth_bin} m" for depth_bin in bins])
        axis.set_xlabel("Ground-truth depth bin")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle(f"{title}: depth-stratified errors")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    axis.set_yticks(range(len(models)), [external_model_display_name(model) for model in models])
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
        values = [lookup.get((*group, model)) for group in groups]
        axes[0].bar(
            x + offset,
            [row["mae_reduction_pct_mean"] if row else np.nan for row in values],
            width,
            color=MODEL_COLORS.get(model),
            label=external_model_display_name(model),
        )
        axes[1].bar(
            x + offset,
            [100.0 * row["csi_gain_mean"] if row else np.nan for row in values],
            width,
            color=MODEL_COLORS.get(model),
            label=external_model_display_name(model),
        )
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
    event_summary_rows: Sequence[dict],
    pairwise_rows: Sequence[dict],
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
        "- Past-only rainfall forcing: no future rainfall is exposed to any model",
        "- Physical depth output in metres; MAE/RMSE reported in centimetres",
        "- Flood skill at 0.10 m, plus threshold sensitivity at 0.05/0.10/0.20/0.30 m",
        "- Wet-area RMSE, dry-cell predicted depth, one-pixel-tolerant boundary F1, and per-event robustness",
        "- Persistence is evaluated on exactly the same test pixels",
        "- U-RNN Lite, FNO2D-History, and SimVP Lite are explicitly labelled as adapted baselines, not exact paper reproductions",
        "- Latency and peak CUDA allocation are inference-only measurements on the recorded GPU",
        "",
    ]
    for (dataset, location), rows in _group_rows(model_rows).items():
        lines.extend([f"## {_group_label(dataset, location)}", ""])
        event_group = [
            row
            for row in event_summary_rows
            if row["dataset"] == dataset and row["location"] == location
        ]
        event_lookup = {row["model_type"]: row for row in event_group}
        minimum_seeds = min(int(row["seed_count"]) for row in rows)
        if minimum_seeds < 3:
            lines.extend([
                "> Status: pilot evidence only. Fewer than three seeds are available for this dataset/location; do not treat its model ranking as statistically stable.",
                "",
            ])
        config = config_lookup.get((dataset, location), {}).get("configuration", {})
        reference = rows[0]
        model_lrs = config.get("model_lr_map", {})
        if model_lrs:
            learning_rate_text = ", ".join(
                f"{external_model_display_name(model)}={float(value):.1e}"
                for model, value in model_lrs.items()
            )
        else:
            learning_rate_text = str(config.get("lr", "unknown"))
        lines.extend([
            f"- Run budget: {minimum_seeds} seeds, {config.get('epochs', 'unknown')} epochs, batch size {config.get('batch_size', 'unknown')}",
            f"- Samples per run: {reference['train_samples']} train / {reference['validation_samples']} validation / {reference['test_samples']} test",
            f"- Sampling caps per event: train={config.get('max_train_samples_per_event', 'unknown')}, evaluation={config.get('max_eval_samples_per_event', 'unknown')} (`0` means all available)",
            f"- Model width: {config.get('hidden', 'unknown')} hidden channels",
            f"- Validation-selected learning rates: {learning_rate_text}",
            "",
        ])
        lines.extend(["| Model | Seeds | MAE (cm) | RMSE (cm) | CSI | MAE gain vs persistence | CSI gain | Latency (ms/sample) | Peak inference CUDA (MB) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in sorted(rows, key=lambda item: _ordered_models(rows).index(item["model_type"])):
            lines.append(
                f"| {row['model_label']} | {row['seed_count']} | "
                f"{_format_pm(row['mean_mae_cm_mean'], row['mean_mae_cm_std'])} | "
                f"{_format_pm(row['mean_rmse_cm_mean'], row['mean_rmse_cm_std'])} | "
                f"{_format_pm(row['mean_csi_mean'], row['mean_csi_std'], 4)} | "
                f"{row['mae_reduction_pct_mean']:.1f}% | {100.0 * row['csi_gain_mean']:.1f} pp | "
                f"{row['latency_ms_per_sample_mean']:.2f} | {row['peak_cuda_memory_mb_mean']:.1f} |"
            )
        lines.extend(
            [
                "",
                "| Model | Trainable parameters | Epochs run | Training time (s) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda item: _ordered_models(rows).index(item["model_type"])):
            lines.append(
                f"| {row['model_label']} | {row['parameter_count']:,} | "
                f"{row['epochs_ran_mean']:.1f} +/- {row['epochs_ran_std']:.1f} | "
                f"{row['training_time_sec_mean']:.2f} +/- {row['training_time_sec_std']:.2f} |"
            )
        lines.extend(
            [
                "",
                "| Model | Wet RMSE (cm) | Dry-cell depth (cm) | Boundary F1 | Peak-time error (min) | Event MAE gain median [IQR] |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda item: _ordered_models(rows).index(item["model_type"])):
            event = event_lookup.get(row["model_type"], {})
            event_interval = (
                f"{event.get('mae_reduction_pct_median', float('nan')):.1f}% "
                f"[{event.get('mae_reduction_pct_q25', float('nan')):.1f}, "
                f"{event.get('mae_reduction_pct_q75', float('nan')):.1f}]"
            )
            lines.append(
                f"| {row['model_label']} | {row['mean_wet_rmse_cm_mean']:.3f} | "
                f"{row['mean_dry_prediction_cm_mean']:.3f} | "
                f"{row['mean_boundary_f1_mean']:.4f} | "
                f"{row['peak_time_mae_min_mean']:.2f} | {event_interval} |"
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
        ])
        if event_group:
            best_event = max(event_group, key=lambda row: row["mae_reduction_pct_median"])
            lines.append(
                f"- Strongest median event-level MAE gain: **{best_event['model_label']}** "
                f"({best_event['mae_reduction_pct_median']:.1f}%, IQR "
                f"{best_event['mae_reduction_pct_q25']:.1f}% to "
                f"{best_event['mae_reduction_pct_q75']:.1f}%)."
            )
        if best_mae["model_type"] != best_csi["model_type"]:
            pair = next(
                (
                    row
                    for row in pairwise_rows
                    if row["dataset"] == dataset
                    and row["location"] == location
                    and {row["model_a"], row["model_b"]}
                    == {best_mae["model_type"], best_csi["model_type"]}
                ),
                None,
            )
            if pair:
                direction = 1.0 if pair["model_a"] == best_csi["model_type"] else -1.0
                mae_delta = direction * pair["mae_delta_a_minus_b_cm"]
                csi_delta = direction * pair["csi_delta_a_minus_b"]
                if direction > 0:
                    mae_interval = (
                        pair["mae_delta_ci95_low_cm"],
                        pair["mae_delta_ci95_high_cm"],
                    )
                    csi_interval = (
                        pair["csi_delta_ci95_low"],
                        pair["csi_delta_ci95_high"],
                    )
                else:
                    mae_interval = (
                        -pair["mae_delta_ci95_high_cm"],
                        -pair["mae_delta_ci95_low_cm"],
                    )
                    csi_interval = (
                        -pair["csi_delta_ci95_high"],
                        -pair["csi_delta_ci95_low"],
                    )
                lines.append(
                    f"- Paired across events, {best_csi['model_label']} minus "
                    f"{best_mae['model_label']}: MAE delta {mae_delta:+.3f} cm "
                    f"(95% bootstrap CI {mae_interval[0]:+.3f} to {mae_interval[1]:+.3f}); "
                    f"CSI delta {csi_delta:+.4f} "
                    f"(95% CI {csi_interval[0]:+.4f} to {csi_interval[1]:+.4f})."
                )
        lines.append("")
    lines.extend(["## Figures", ""])
    for figure in figure_paths:
        lines.append(f"![{figure.stem}](figures/{figure.name})")
        lines.append("")
    lines.extend([
        "## Interpretation Rules",
        "",
        "Positive MAE reduction and positive CSI gain mean the learned model beats persistence. Model-to-model claims should rely on multi-seed means, deviations, and event-level IQR rather than a single run. Latency and VRAM are device-specific and are intended for relative comparison on the recorded GPU.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_external_results(input_root: str | Path, output_dir: str | Path) -> dict:
    runs = load_external_runs(input_root)
    run_rows, horizon_rows, threshold_rows = flatten_external_runs(runs)
    event_rows = flatten_external_events(runs)
    depth_bin_rows = flatten_external_depth_bins(runs)
    event_mean_rows, event_summary_rows = summarize_event_robustness(event_rows)
    pairwise_rows = compare_models_by_event(event_mean_rows)
    depth_bin_summary = summarize_depth_bins(depth_bin_rows)
    model_rows = summarize_models(run_rows)
    dataset_rows = summarize_datasets(run_rows)
    horizon_summary = summarize_horizons(horizon_rows)
    threshold_summary = summarize_thresholds(threshold_rows)
    output = ensure_dir(output_dir)
    figures = ensure_dir(output / "figures")
    configs = [
        _public_benchmark_config(load_json(path))
        for path in sorted(Path(input_root).rglob("benchmark_config.json"))
    ]
    _write_csv(run_rows, output / "external_per_run.csv")
    _write_csv(horizon_rows, output / "external_per_horizon.csv")
    _write_csv(threshold_rows, output / "external_per_threshold.csv")
    _write_csv(event_rows, output / "external_per_event_seed.csv")
    _write_csv(event_mean_rows, output / "external_per_event.csv")
    _write_csv(event_summary_rows, output / "external_event_summary.csv")
    _write_csv(pairwise_rows, output / "external_pairwise_event_comparison.csv")
    _write_csv(depth_bin_rows, output / "external_per_depth_bin.csv")
    _write_csv(depth_bin_summary, output / "external_depth_bin_summary.csv")
    _write_csv(model_rows, output / "external_model_summary.csv")
    _write_csv(dataset_rows, output / "external_dataset_summary.csv")
    _write_csv(horizon_summary, output / "external_horizon_summary.csv")
    _write_csv(threshold_summary, output / "external_threshold_summary.csv")

    figure_paths = []
    model_groups = _group_rows(model_rows)
    horizon_groups = _group_rows(horizon_summary)
    threshold_groups = _group_rows(threshold_summary)
    event_groups = _group_rows(event_mean_rows)
    depth_bin_groups = _group_rows(depth_bin_summary)
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
            (
                _plot_physical_diagnostics,
                group_model_rows,
                figures / f"{slug}_physical_diagnostics.png",
            ),
            (_plot_heatmap, group_horizons, figures / f"{slug}_skill_heatmap.png"),
        )
        for plotter, rows, path in plotters:
            plotter(rows, path, title)
            figure_paths.append(path)
        if group in event_groups:
            event_path = figures / f"{slug}_event_robustness.png"
            _plot_event_robustness(event_groups[group], event_path, title)
            figure_paths.append(event_path)
        if group in depth_bin_groups:
            depth_bin_path = figures / f"{slug}_depth_stratified_errors.png"
            _plot_depth_bins(depth_bin_groups[group], depth_bin_path, title)
            figure_paths.append(depth_bin_path)

    if len(model_groups) > 1:
        cross_dataset_path = figures / "cross_dataset_generalization.png"
        _plot_cross_dataset(model_rows, cross_dataset_path)
        figure_paths.insert(0, cross_dataset_path)
    optional_patterns = ("*_spatial_forecast.png", "*_spatial_error.png", "*_horizon_error_matrix.png")
    for optional_pattern in optional_patterns:
        figure_paths.extend(sorted(figures.glob(optional_pattern)))

    summary = {
        "schema_version": "external_physical_summary_v2",
        "input_root": Path(input_root).as_posix(),
        "run_count": len(runs),
        "datasets": sorted({str(run["dataset"]) for run in runs}),
        "models": _ordered_models(run_rows),
        "seeds": sorted({int(run["seed"]) for run in runs}),
        "model_summary": model_rows,
        "dataset_summary": dataset_rows,
        "horizon_summary": horizon_summary,
        "threshold_summary": threshold_summary,
        "event_summary": event_summary_rows,
        "pairwise_event_comparison": pairwise_rows,
        "depth_bin_summary": depth_bin_summary,
        "benchmark_configs": configs,
        "figures": [path.relative_to(output).as_posix() for path in figure_paths],
    }
    save_json(summary, output / "external_benchmark_summary.json")
    _write_report(
        model_rows,
        horizon_summary,
        event_summary_rows,
        pairwise_rows,
        output / "EXTERNAL_PHYSICAL_BENCHMARK.md",
        figure_paths,
        configs,
    )
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
