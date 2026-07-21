from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .external_data import (
    EXTERNAL_CHANNEL_NAMES,
    ExternalFloodDataset,
    discover_larno_ukea,
    discover_urbanflood24,
    split_train_validation,
)
from .external_models import (
    EXTERNAL_MODEL_TYPES,
    build_external_model,
    count_external_parameters,
    external_model_display_name,
    normalize_external_model_type,
)
from .utils import ensure_dir, save_json, set_seed


def parse_int_list(value: str | Sequence[int]) -> tuple[int, ...]:
    values = [int(item.strip()) for item in value.split(",")] if isinstance(value, str) else [int(item) for item in value]
    values = sorted(set(values))
    if not values or values[0] < 1:
        raise ValueError("Expected unique positive integers")
    return tuple(values)


def parse_float_list(value: str | Sequence[float]) -> tuple[float, ...]:
    values = [float(item.strip()) for item in value.split(",")] if isinstance(value, str) else [float(item) for item in value]
    values = sorted(set(values))
    if not values or values[0] <= 0:
        raise ValueError("Expected unique positive thresholds")
    return tuple(values)


def masked_physical_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.10,
    wet_weight: float = 4.0,
    mse_weight: float = 0.25,
    bce_weight: float = 0.02,
    dice_weight: float = 0.05,
    temperature: float = 0.03,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = valid_mask.expand_as(target)
    weights = 1.0 + wet_weight * (target >= threshold).float()
    weighted_mask = mask * weights
    weighted_count = weighted_mask.sum().clamp_min(1.0)
    absolute = (weighted_mask * torch.abs(prediction - target)).sum() / weighted_count
    squared = (weighted_mask * (prediction - target).square()).sum() / weighted_count

    target_binary = (target >= threshold).float()
    logits = (prediction - threshold) / max(temperature, 1e-6)
    bce = (F.binary_cross_entropy_with_logits(logits, target_binary, reduction="none") * mask).sum()
    bce = bce / mask.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * mask
    dimensions = (-2, -1)
    intersection = (probability * target_binary).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + (target_binary * mask).sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    total = absolute + mse_weight * squared + bce_weight * bce + dice_weight * dice
    return total, {
        "loss": float(total.detach().item()),
        "loss_mae": float(absolute.detach().item()),
        "loss_mse": float(squared.detach().item()),
        "loss_bce": float(bce.detach().item()),
        "loss_dice": float(dice.detach().item()),
    }


DEPTH_BINS = (
    ("0.00-0.10", 0.00, 0.10),
    ("0.10-0.30", 0.10, 0.30),
    ("0.30-0.50", 0.30, 0.50),
    ("0.50+", 0.50, None),
)


def _inner_boundary(binary: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(binary, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_binary = np.roll(binary, (dy, dx), axis=(0, 1))
        shifted_valid = np.roll(valid, (dy, dx), axis=(0, 1))
        if dy == -1:
            shifted_valid[-1, :] = False
        elif dy == 1:
            shifted_valid[0, :] = False
        elif dx == -1:
            shifted_valid[:, -1] = False
        else:
            shifted_valid[:, 0] = False
        boundary |= binary & valid & shifted_valid & ~shifted_binary
    return boundary


def _dilate(binary: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    output = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output |= padded[dy : dy + height, dx : dx + width]
    return output


class PhysicalMetricAccumulator:
    def __init__(self, horizons: int, thresholds: Sequence[float], primary_threshold: float) -> None:
        self.horizons = int(horizons)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.primary_threshold = float(primary_threshold)
        self.count = np.zeros(self.horizons, dtype=np.int64)
        self.absolute = np.zeros(self.horizons, dtype=np.float64)
        self.squared = np.zeros(self.horizons, dtype=np.float64)
        self.wet_count = np.zeros(self.horizons, dtype=np.int64)
        self.wet_absolute = np.zeros(self.horizons, dtype=np.float64)
        self.wet_squared = np.zeros(self.horizons, dtype=np.float64)
        self.dry_count = np.zeros(self.horizons, dtype=np.int64)
        self.dry_prediction = np.zeros(self.horizons, dtype=np.float64)
        self.peak_count = np.zeros(self.horizons, dtype=np.int64)
        self.peak_absolute = np.zeros(self.horizons, dtype=np.float64)
        self.depth_bins = {
            label: {
                "count": np.zeros(self.horizons, dtype=np.int64),
                "absolute": np.zeros(self.horizons, dtype=np.float64),
                "squared": np.zeros(self.horizons, dtype=np.float64),
            }
            for label, _, _ in DEPTH_BINS
        }
        self.boundary = np.zeros((self.horizons, 4), dtype=np.int64)
        self.confusion = {
            threshold: np.zeros((self.horizons, 4), dtype=np.int64) for threshold in self.thresholds
        }

    def update(self, prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        prediction_np = prediction.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        for horizon in range(self.horizons):
            valid = np.broadcast_to(mask_np[:, 0], target_np[:, horizon].shape)
            pred = prediction_np[:, horizon][valid]
            truth = target_np[:, horizon][valid]
            difference = pred - truth
            self.count[horizon] += truth.size
            self.absolute[horizon] += np.abs(difference).sum(dtype=np.float64)
            self.squared[horizon] += np.square(difference).sum(dtype=np.float64)
            wet = truth >= self.primary_threshold
            self.wet_count[horizon] += int(wet.sum())
            self.wet_absolute[horizon] += np.abs(difference[wet]).sum(dtype=np.float64)
            self.wet_squared[horizon] += np.square(difference[wet]).sum(dtype=np.float64)
            dry = ~wet
            self.dry_count[horizon] += int(dry.sum())
            self.dry_prediction[horizon] += pred[dry].sum(dtype=np.float64)

            for label, lower, upper in DEPTH_BINS:
                selected = truth >= lower
                if upper is not None:
                    selected &= truth < upper
                bin_difference = difference[selected]
                self.depth_bins[label]["count"][horizon] += int(selected.sum())
                self.depth_bins[label]["absolute"][horizon] += np.abs(bin_difference).sum(dtype=np.float64)
                self.depth_bins[label]["squared"][horizon] += np.square(bin_difference).sum(dtype=np.float64)

            for sample_index in range(target_np.shape[0]):
                sample_valid = mask_np[sample_index, 0]
                if sample_valid.any():
                    pred_peak = float(prediction_np[sample_index, horizon][sample_valid].max())
                    target_peak = float(target_np[sample_index, horizon][sample_valid].max())
                    self.peak_absolute[horizon] += abs(pred_peak - target_peak)
                    self.peak_count[horizon] += 1

                    prediction_extent = (prediction_np[sample_index, horizon] >= self.primary_threshold) & sample_valid
                    target_extent = (target_np[sample_index, horizon] >= self.primary_threshold) & sample_valid
                    prediction_boundary = _inner_boundary(prediction_extent, sample_valid)
                    target_boundary = _inner_boundary(target_extent, sample_valid)
                    matched_prediction = int((prediction_boundary & _dilate(target_boundary)).sum())
                    matched_target = int((target_boundary & _dilate(prediction_boundary)).sum())
                    self.boundary[horizon] += (
                        matched_prediction,
                        int(prediction_boundary.sum()),
                        matched_target,
                        int(target_boundary.sum()),
                    )

            for threshold in self.thresholds:
                pred_binary = pred >= threshold
                truth_binary = truth >= threshold
                tp = np.logical_and(pred_binary, truth_binary).sum()
                fp = np.logical_and(pred_binary, ~truth_binary).sum()
                fn = np.logical_and(~pred_binary, truth_binary).sum()
                tn = np.logical_and(~pred_binary, ~truth_binary).sum()
                self.confusion[threshold][horizon] += (tp, fp, fn, tn)

    def finalize(self, lead_times: Sequence[int], time_step_minutes: int = 5) -> list[dict]:
        rows = []
        for horizon, lead in enumerate(lead_times):
            count = max(int(self.count[horizon]), 1)
            wet_count = max(int(self.wet_count[horizon]), 1)
            dry_count = max(int(self.dry_count[horizon]), 1)
            matched_prediction, prediction_boundary, matched_target, target_boundary = (
                int(value) for value in self.boundary[horizon]
            )
            boundary_precision = matched_prediction / max(prediction_boundary, 1)
            boundary_recall = matched_target / max(target_boundary, 1)
            if prediction_boundary == 0 and target_boundary == 0:
                boundary_f1 = 1.0
            else:
                boundary_f1 = 2.0 * boundary_precision * boundary_recall / max(
                    boundary_precision + boundary_recall, 1e-12
                )
            row = {
                "lead_steps": int(lead),
                "lead_minutes": int(lead * time_step_minutes),
                "pixels": int(self.count[horizon]),
                "mae_m": float(self.absolute[horizon] / count),
                "rmse_m": float(np.sqrt(self.squared[horizon] / count)),
                "mae_cm": float(100.0 * self.absolute[horizon] / count),
                "rmse_cm": float(100.0 * np.sqrt(self.squared[horizon] / count)),
                "wet_mae_m": float(self.wet_absolute[horizon] / wet_count),
                "wet_rmse_m": float(np.sqrt(self.wet_squared[horizon] / wet_count)),
                "wet_pixels": int(self.wet_count[horizon]),
                "dry_prediction_mean_m": float(self.dry_prediction[horizon] / dry_count),
                "dry_pixels": int(self.dry_count[horizon]),
                "peak_depth_mae_m": float(
                    self.peak_absolute[horizon] / max(int(self.peak_count[horizon]), 1)
                ),
                "boundary_precision": float(boundary_precision),
                "boundary_recall": float(boundary_recall),
                "boundary_f1": float(boundary_f1),
                "boundary_tolerance_pixels": 1,
                "depth_bin_metrics": {},
                "threshold_metrics": {},
            }
            for label, _, _ in DEPTH_BINS:
                bin_count = int(self.depth_bins[label]["count"][horizon])
                denominator = max(bin_count, 1)
                row["depth_bin_metrics"][label] = {
                    "pixels": bin_count,
                    "mae_m": float(self.depth_bins[label]["absolute"][horizon] / denominator),
                    "rmse_m": float(np.sqrt(self.depth_bins[label]["squared"][horizon] / denominator)),
                }
            for threshold in self.thresholds:
                tp, fp, fn, tn = (int(value) for value in self.confusion[threshold][horizon])
                csi = tp / max(tp + fp + fn, 1)
                pod = tp / max(tp + fn, 1)
                far = fp / max(tp + fp, 1)
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                false_positive_rate = fp / max(fp + tn, 1)
                metrics = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "csi": csi,
                    "pod": pod,
                    "far": far,
                    "f1": f1,
                    "false_positive_rate": false_positive_rate,
                }
                row["threshold_metrics"][f"{threshold:.2f}"] = metrics
                if abs(threshold - self.primary_threshold) < 1e-9:
                    row.update(
                        {
                            "csi": csi,
                            "pod": pod,
                            "far": far,
                            "f1": f1,
                            "dry_false_positive_rate": false_positive_rate,
                            "threshold_m": threshold,
                        }
                    )
            rows.append(row)
        return rows


def evaluate(
    model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    lead_times: Sequence[int],
    thresholds: Sequence[float],
    primary_threshold: float,
    depth_scale_m: float,
    amp: bool,
    include_per_event: bool = False,
) -> dict:
    if model is not None:
        model.eval()
    model_metrics = PhysicalMetricAccumulator(len(lead_times), thresholds, primary_threshold)
    persistence_metrics = PhysicalMetricAccumulator(len(lead_times), thresholds, primary_threshold)
    event_model_metrics: dict[str, PhysicalMetricAccumulator] = {}
    event_persistence_metrics: dict[str, PhysicalMetricAccumulator] = {}
    event_samples: dict[str, int] = defaultdict(int)
    event_peak_time_errors: dict[str, list[float]] = defaultdict(list)
    event_persistence_peak_time_errors: dict[str, list[float]] = defaultdict(list)
    peak_time_errors: list[float] = []
    persistence_peak_time_errors: list[float] = []
    losses: list[float] = []
    lead_minutes = torch.tensor([int(lead) * 5 for lead in lead_times], device=device)
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            persistence = (x[:, -1, 0:1] * depth_scale_m).repeat(1, len(lead_times), 1, 1)
            if model is None:
                prediction = persistence
            else:
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    prediction = model(x)
                    loss, _ = masked_physical_loss(prediction, target, mask, threshold=primary_threshold)
                losses.append(float(loss.detach().item()))
            model_metrics.update(prediction, target, mask)
            persistence_metrics.update(persistence, target, mask)
            if include_per_event:
                valid = mask.expand_as(target).bool()
                floor = torch.finfo(prediction.dtype).min
                prediction_peak = prediction.masked_fill(~valid, floor).amax(dim=(-2, -1))
                persistence_peak = persistence.masked_fill(~valid, floor).amax(dim=(-2, -1))
                target_peak = target.masked_fill(~valid, floor).amax(dim=(-2, -1))
                target_peak_time = lead_minutes[target_peak.argmax(dim=1)]
                prediction_errors = (
                    lead_minutes[prediction_peak.argmax(dim=1)] - target_peak_time
                ).abs().float().cpu().tolist()
                persistence_errors = (
                    lead_minutes[persistence_peak.argmax(dim=1)] - target_peak_time
                ).abs().float().cpu().tolist()
                peak_time_errors.extend(prediction_errors)
                persistence_peak_time_errors.extend(persistence_errors)

                for sample_index, event_id_value in enumerate(batch["event_id"]):
                    event_id = str(event_id_value)
                    if event_id not in event_model_metrics:
                        event_model_metrics[event_id] = PhysicalMetricAccumulator(
                            len(lead_times), thresholds, primary_threshold
                        )
                        event_persistence_metrics[event_id] = PhysicalMetricAccumulator(
                            len(lead_times), thresholds, primary_threshold
                        )
                    sample_slice = slice(sample_index, sample_index + 1)
                    event_model_metrics[event_id].update(
                        prediction[sample_slice], target[sample_slice], mask[sample_slice]
                    )
                    event_persistence_metrics[event_id].update(
                        persistence[sample_slice], target[sample_slice], mask[sample_slice]
                    )
                    event_samples[event_id] += 1
                    event_peak_time_errors[event_id].append(float(prediction_errors[sample_index]))
                    event_persistence_peak_time_errors[event_id].append(
                        float(persistence_errors[sample_index])
                    )
    result = {
        "loss": float(np.mean(losses)) if losses else None,
        "per_horizon": model_metrics.finalize(lead_times),
        "persistence_per_horizon": persistence_metrics.finalize(lead_times),
        "samples": int(len(loader.dataset)),
    }
    if include_per_event:
        result.update(
            {
                "peak_time_mae_min": float(np.mean(peak_time_errors)) if peak_time_errors else 0.0,
                "persistence_peak_time_mae_min": float(np.mean(persistence_peak_time_errors))
                if persistence_peak_time_errors
                else 0.0,
                "per_event": [
                    {
                        "event_id": event_id,
                        "samples": int(event_samples[event_id]),
                        "peak_time_mae_min": float(np.mean(event_peak_time_errors[event_id])),
                        "persistence_peak_time_mae_min": float(
                            np.mean(event_persistence_peak_time_errors[event_id])
                        ),
                        "per_horizon": event_model_metrics[event_id].finalize(lead_times),
                        "persistence_per_horizon": event_persistence_metrics[event_id].finalize(
                            lead_times
                        ),
                    }
                    for event_id in sorted(event_model_metrics)
                ],
            }
        )
    return result


def benchmark(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    batches: int = 10,
) -> dict:
    cached = []
    for index, batch in enumerate(loader):
        cached.append(batch["x"])
        if index + 1 >= batches:
            break
    if not cached:
        raise ValueError("Cannot benchmark an empty loader")
    model.eval()
    with torch.inference_mode():
        for x in cached[:2]:
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                model(x.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        samples = 0
        for x in cached:
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                model(x.to(device))
            samples += int(x.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "latency_ms_per_sample": float(1000.0 * elapsed / max(samples, 1)),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0,
        "benchmark_samples": samples,
    }


def _event_splits(args: argparse.Namespace):
    if args.dataset == "urbanflood24":
        official_train = discover_urbanflood24(args.urban_root, "train", args.location)
        test = discover_urbanflood24(args.urban_root, "test", args.location)
        validation_count = args.validation_events or 8
    else:
        official_train = discover_larno_ukea(args.larno_root, "train")
        test = discover_larno_ukea(args.larno_root, "test")
        validation_count = args.validation_events or 2
    train, validation = split_train_validation(official_train, validation_count, args.split_seed)
    return train, validation, test


def _plot_results(history: dict, test_metrics: dict, output_dir: Path, model_label: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Physical masked loss")
    plt.title(f"{model_label} external training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=180)
    plt.close()

    rows = test_metrics["per_horizon"]
    minutes = [row["lead_minutes"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(minutes, [row["mae_cm"] for row in rows], marker="o", label="Model")
    axes[0].plot(
        minutes,
        [row["mae_cm"] for row in test_metrics["persistence_per_horizon"]],
        marker="s",
        label="Persistence",
    )
    axes[0].set(xlabel="Lead time (min)", ylabel="MAE (cm)")
    axes[0].legend()
    axes[1].plot(minutes, [row["csi"] for row in rows], marker="o", label="Model")
    axes[1].plot(
        minutes,
        [row["csi"] for row in test_metrics["persistence_per_horizon"]],
        marker="s",
        label="Persistence",
    )
    axes[1].set(xlabel="Lead time (min)", ylabel="CSI at 0.10 m", ylim=(0, 1))
    axes[1].legend()
    figure.suptitle(f"{model_label} physical benchmark")
    figure.tight_layout()
    figure.savefig(figure_dir / "horizon_metrics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train copied architecture variants on physical flood datasets.")
    parser.add_argument("--dataset", choices=["urbanflood24", "larno_ukea"], required=True)
    parser.add_argument("--urban_root", default="../urbanflood24")
    parser.add_argument("--larno_root", default="../external_datasets/larno_ukea_8m_5min")
    parser.add_argument("--location", default="location1", choices=["location1", "location2", "location3"])
    parser.add_argument("--model_type", default="convlstm", help=f"One of: {', '.join(EXTERNAL_MODEL_TYPES)}")
    parser.add_argument("--output_dir", default="runs/external_physical/pilot")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_times", default="1,3,6,12")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--train_patch_stride", type=int, default=32)
    parser.add_argument("--eval_patch_stride", type=int, default=64)
    parser.add_argument("--max_train_samples_per_event", type=int, default=64)
    parser.add_argument("--max_eval_samples_per_event", type=int, default=0)
    parser.add_argument("--validation_events", type=int, default=0)
    parser.add_argument("--depth_scale_m", type=float, default=3.5)
    parser.add_argument("--rain_scale_mm_5min", type=float, default=35.0)
    parser.add_argument("--thresholds", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--primary_threshold", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--fno_modes", type=int, default=8)
    parser.add_argument("--fno_layers", type=int, default=3)
    parser.add_argument("--simvp_blocks", type=int, default=4)
    parser.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Stop after validation-model selection without reading the test loader.",
    )
    args = parser.parse_args()

    args.model_type = normalize_external_model_type(args.model_type)
    lead_times = parse_int_list(args.lead_times)
    thresholds = parse_float_list(args.thresholds)
    if args.primary_threshold not in thresholds:
        thresholds = tuple(sorted(set(thresholds + (args.primary_threshold,))))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    train_events, validation_events, test_events = _event_splits(args)
    dataset_kwargs = dict(
        input_len=args.input_len,
        lead_times=lead_times,
        patch_size=args.patch_size,
        depth_scale_m=args.depth_scale_m,
        rain_scale_mm_5min=args.rain_scale_mm_5min,
    )
    train_dataset = ExternalFloodDataset(
        train_events,
        patch_stride=args.train_patch_stride,
        max_samples_per_event=args.max_train_samples_per_event,
        seed=args.seed,
        **dataset_kwargs,
    )
    eval_cap = args.max_eval_samples_per_event or None
    validation_dataset = ExternalFloodDataset(
        validation_events,
        patch_stride=args.eval_patch_stride,
        max_samples_per_event=eval_cap,
        seed=args.split_seed,
        **dataset_kwargs,
    )
    test_dataset = ExternalFloodDataset(
        test_events,
        patch_stride=args.eval_patch_stride,
        max_samples_per_event=eval_cap,
        seed=args.split_seed,
        **dataset_kwargs,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_external_model(
        args.model_type,
        input_channels=len(EXTERNAL_CHANNEL_NAMES),
        output_channels=len(lead_times),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        transformer_heads=args.transformer_heads,
        output_max=args.depth_scale_m,
        max_input_len=args.input_len,
        use_residual=args.use_residual,
        residual_scale=args.residual_scale,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
        simvp_blocks=args.simvp_blocks,
    ).to(device)
    parameter_count = count_external_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metric_dir = ensure_dir(output_dir / "metrics")
    save_json(train_dataset.manifest(), metric_dir / "train_manifest.json")
    save_json(validation_dataset.manifest(), metric_dir / "validation_manifest.json")
    save_json(test_dataset.manifest(), metric_dir / "test_manifest.json")

    model_label = external_model_display_name(args.model_type)
    print(f"Dataset: {args.dataset}; device: {device}; model: {model_label}")
    print(f"Events train/val/test: {len(train_events)}/{len(validation_events)}/{len(test_events)}")
    print(f"Samples train/val/test: {len(train_dataset)}/{len(validation_dataset)}/{len(test_dataset)}")
    print(f"Parameters: {parameter_count:,}")
    history = {"train_loss": [], "val_loss": [], "val_mae_cm": [], "val_csi": []}
    start_time = time.time()
    best_epoch = 0

    def save_best_checkpoint() -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "schema_version": "external_physical_v2",
                "model_type": args.model_type,
                "model_label": model_label,
                "input_channels": len(EXTERNAL_CHANNEL_NAMES),
                "channel_names": list(EXTERNAL_CHANNEL_NAMES),
                "hidden_channels": args.hidden,
                "num_layers": args.num_layers,
                "input_len": args.input_len,
                "lead_times": list(lead_times),
                "time_step_minutes": 5,
                "depth_scale_m": args.depth_scale_m,
                "rain_scale_mm_5min": args.rain_scale_mm_5min,
                "use_residual": args.use_residual,
                "residual_scale": args.residual_scale,
                "parameter_count": parameter_count,
                "dataset": args.dataset,
                "location": args.location if args.dataset == "urbanflood24" else "ukea",
                "seed": args.seed,
                "split_seed": args.split_seed,
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            checkpoint_dir / "best.pt",
        )

    initial_validation = evaluate(
        model,
        validation_loader,
        device,
        lead_times,
        thresholds,
        args.primary_threshold,
        args.depth_scale_m,
        args.amp,
    )
    best_loss = float(initial_validation["loss"])
    save_best_checkpoint()
    print(f"Epoch 0 persistence initialization: val={best_loss:.5f}")
    training_start = time.time()
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not args.progress)
        for batch in progress:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                prediction = model(x)
                loss, _ = masked_physical_loss(
                    prediction, target, mask, threshold=args.primary_threshold
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().item()))
            progress.set_postfix(loss=float(np.mean(epoch_losses)))

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            lead_times,
            thresholds,
            args.primary_threshold,
            args.depth_scale_m,
            args.amp,
        )
        train_loss = float(np.mean(epoch_losses))
        validation_loss = float(validation_metrics["loss"])
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)
        history["val_mae_cm"].append(float(np.mean([row["mae_cm"] for row in validation_metrics["per_horizon"]])))
        history["val_csi"].append(float(np.mean([row["csi"] for row in validation_metrics["per_horizon"]])))
        print(
            f"Epoch {epoch}: train={train_loss:.5f}, val={validation_loss:.5f}, "
            f"val_mae={history['val_mae_cm'][-1]:.3f} cm, val_csi={history['val_csi'][-1]:.4f}"
        )
        if validation_loss < best_loss - args.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint()
        else:
            epochs_without_improvement += 1
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    training_time_sec = time.time() - training_start
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if args.selection_only:
        selection_metrics = evaluate(
            model,
            validation_loader,
            device,
            lead_times,
            thresholds,
            args.primary_threshold,
            args.depth_scale_m,
            args.amp,
        )
        selection_metrics.update(
            {
                "schema_version": "external_selection_v1",
                "dataset": args.dataset,
                "location": args.location if args.dataset == "urbanflood24" else "ukea",
                "model_type": args.model_type,
                "model_label": model_label,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "learning_rate": args.lr,
                "best_epoch": int(checkpoint.get("best_epoch", best_epoch)),
                "epochs_ran": int(len(history["train_loss"])),
                "best_validation_loss": float(best_loss),
                "training_time_sec": float(training_time_sec),
                "validation_events": [event.event_id for event in validation_events],
                "test_evaluated": False,
            }
        )
        save_json(history, metric_dir / "train_history.json")
        save_json(selection_metrics, metric_dir / "selection_metrics.json")
        print(f"Selection-only run finished without test evaluation: {metric_dir / 'selection_metrics.json'}")
        return
    evaluation_start = time.time()
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        lead_times,
        thresholds,
        args.primary_threshold,
        args.depth_scale_m,
        args.amp,
        include_per_event=True,
    )
    evaluation_time_sec = time.time() - evaluation_start
    test_metrics.update(
        {
            "schema_version": "external_physical_v2",
            "metric_schema_version": "physical_metrics_v2",
            "dataset": args.dataset,
            "location": args.location if args.dataset == "urbanflood24" else "ukea",
            "model_type": args.model_type,
            "model_label": model_label,
            "parameter_count": parameter_count,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "best_epoch": int(checkpoint.get("best_epoch", best_epoch)),
            "epochs_requested": int(args.epochs),
            "epochs_ran": int(len(history["train_loss"])),
            "train_events": [event.event_id for event in train_events],
            "validation_events": [event.event_id for event in validation_events],
            "test_events": [event.event_id for event in test_events],
            "thresholds_m": list(thresholds),
            "primary_threshold_m": args.primary_threshold,
            "training_time_sec": float(training_time_sec),
            "evaluation_time_sec": float(evaluation_time_sec),
            "runtime_sec": float(time.time() - start_time),
            "protocol": "state-aware flood nowcast at 8 m / 5 min; event-disjoint split",
            "protocol_details": {
                "rain_forcing": "past_only",
                "prediction_mode": "residual" if args.use_residual else "absolute",
                "input_minutes": int(args.input_len * 5),
                "lead_minutes": [int(lead * 5) for lead in lead_times],
                "early_stop_patience": int(args.early_stop_patience),
                "minimum_improvement": float(args.min_delta),
                "evaluation_sampling_seed": int(args.split_seed),
            },
        }
    )
    test_metrics.update(benchmark(model, test_loader, device, args.amp))
    save_json(history, metric_dir / "train_history.json")
    save_json(test_metrics, metric_dir / "test_metrics.json")
    _plot_results(history, test_metrics, output_dir, model_label)
    print(f"Finished. Metrics: {metric_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
