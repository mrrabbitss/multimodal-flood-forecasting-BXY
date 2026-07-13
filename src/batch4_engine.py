from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .batch4_dataset import MultiHorizonFloodDataset
from .metrics import all_metrics
from .training.losses import LossConfig, build_loss


def parse_lead_times(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [int(item) for item in value]
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise ValueError("lead_times must be unique positive integers")
    return tuple(sorted(values))


def configure_determinism(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled


def evaluate_batch4_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lead_times: Sequence[int],
    threshold: float,
    loss_config: LossConfig,
    include_per_event: bool = False,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_values: dict[str, list[float]] = defaultdict(list)
    event_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    event_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    dataset = loader.dataset
    sample_cursor = 0
    with torch.inference_mode():
        for x, y in loader:
            prediction = model(x.to(device, non_blocking=True))
            breakdown = build_loss(prediction, y.to(device, non_blocking=True), loss_config)
            prediction_np = prediction.cpu().numpy()
            target_np = y.numpy()
            predictions.append(prediction_np)
            targets.append(target_np)
            for key, value in breakdown.detached_values().items():
                loss_values[key].append(value)
            if include_per_event:
                if not isinstance(dataset, MultiHorizonFloodDataset):
                    raise TypeError("Per-event evaluation requires MultiHorizonFloodDataset")
                batch_samples = dataset.samples[sample_cursor : sample_cursor + prediction_np.shape[0]]
                for local_index, sample in enumerate(batch_samples):
                    event_id = sample.file_path.stem
                    event_predictions[event_id].append(prediction_np[local_index : local_index + 1])
                    event_targets[event_id].append(target_np[local_index : local_index + 1])
                sample_cursor += prediction_np.shape[0]

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    aggregate = all_metrics(prediction_array, target_array, threshold=threshold)
    aggregate.update({key: float(np.mean(values)) for key, values in loss_values.items()})
    aggregate["loss"] = aggregate["loss_total"]
    aggregate["num_samples"] = int(prediction_array.shape[0])

    per_horizon = []
    for horizon_index, lead_time in enumerate(lead_times):
        row = all_metrics(
            prediction_array[:, horizon_index : horizon_index + 1],
            target_array[:, horizon_index : horizon_index + 1],
            threshold=threshold,
        )
        row.update({"lead_time": int(lead_time), "num_samples": int(prediction_array.shape[0])})
        per_horizon.append(row)

    per_event_horizon = []
    if include_per_event:
        for event_id in sorted(event_predictions):
            event_prediction = np.concatenate(event_predictions[event_id], axis=0)
            event_target = np.concatenate(event_targets[event_id], axis=0)
            for horizon_index, lead_time in enumerate(lead_times):
                row = all_metrics(
                    event_prediction[:, horizon_index : horizon_index + 1],
                    event_target[:, horizon_index : horizon_index + 1],
                    threshold=threshold,
                )
                row.update(
                    {
                        "event_id": event_id,
                        "lead_time": int(lead_time),
                        "num_samples": int(event_prediction.shape[0]),
                    }
                )
                per_event_horizon.append(row)
    return {"aggregate": aggregate, "per_horizon": per_horizon, "per_event_horizon": per_event_horizon}


def benchmark_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 2,
    benchmark_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    batches = []
    for index, (x, _) in enumerate(loader):
        batches.append(x)
        if index + 1 >= max(warmup_batches, benchmark_batches):
            break
    if not batches:
        raise ValueError("Cannot benchmark an empty loader")
    with torch.inference_mode():
        for x in batches[:warmup_batches]:
            model(x.to(device, non_blocking=True))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        sample_count = 0
        for x in batches[:benchmark_batches]:
            model(x.to(device, non_blocking=True))
            sample_count += int(x.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "latency_ms_per_sample": float(1000.0 * elapsed / max(sample_count, 1)),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
    }
