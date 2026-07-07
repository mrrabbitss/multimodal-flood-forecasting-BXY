from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FloodSequenceDataset
from .metrics import all_metrics
from .model_variants import (
    build_model_from_checkpoint,
    checkpoint_model_type,
    count_parameters,
    model_display_name,
)
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def benchmark_latency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    benchmark_batches: int,
    amp: bool,
) -> dict:
    model.eval()
    total_ms = 0.0
    total_samples = 0
    measured_batches = 0

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= warmup_batches:
                break
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= benchmark_batches:
                break
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_ms += elapsed_ms
            total_samples += int(x.shape[0])
            measured_batches += 1

    if measured_batches == 0 or total_samples == 0:
        return {
            "latency_ms_per_batch": 0.0,
            "latency_ms_per_sample": 0.0,
            "benchmark_batches": 0,
            "benchmark_samples": 0,
        }
    return {
        "latency_ms_per_batch": float(total_ms / measured_batches),
        "latency_ms_per_sample": float(total_ms / total_samples),
        "benchmark_batches": int(measured_batches),
        "benchmark_samples": int(total_samples),
    }


def evaluate_checkpoint(
    fused_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
    device_value: str = "auto",
    seed: int = 42,
    threshold: float | None = None,
    warmup_batches: int = 3,
    benchmark_batches: int = 20,
    amp: bool = False,
) -> dict:
    eval_start = time.time()
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = resolve_device(device_value)
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    input_len = int(ckpt["input_len"])
    lead_time = int(ckpt["lead_time"])
    eval_threshold = float(threshold if threshold is not None else ckpt.get("threshold", 0.30))
    split_seed = int(ckpt.get("split_seed", seed))
    shuffle_split = bool(ckpt.get("shuffle_split", False))
    model_type = checkpoint_model_type(ckpt)

    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    _, _, test_idx = FloodSequenceDataset.split_indices(len(files), seed=split_seed, shuffle=shuffle_split)
    test_ds = FloodSequenceDataset(fused_dir, test_idx, input_len=input_len, lead_time=lead_time)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    latency = benchmark_latency(
        model=model,
        loader=loader,
        device=device,
        warmup_batches=warmup_batches,
        benchmark_batches=benchmark_batches,
        amp=amp,
    )

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()

    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = all_metrics(pred_np, target_np, threshold=eval_threshold)
    metrics.update(latency)
    metrics["model_type"] = model_type
    metrics["model_label"] = ckpt.get("model_label", model_display_name(model_type))
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["parameter_count"] = int(ckpt.get("parameter_count", count_parameters(model)))
    metrics["num_test_samples"] = int(len(test_ds))
    metrics["test_events"] = test_idx
    metrics["best_epoch"] = int(ckpt.get("best_epoch", 0))
    metrics["split_seed"] = int(split_seed)
    metrics["shuffle_split"] = bool(shuffle_split)
    metrics["threshold"] = float(eval_threshold)
    metrics["device"] = str(device)
    metrics["amp"] = bool(amp and device.type == "cuda")
    metrics["evaluation_runtime_sec"] = float(time.time() - eval_start)
    if device.type == "cuda":
        metrics["peak_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        metrics["peak_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024**2))
        metrics["cuda_device_name"] = torch.cuda.get_device_name(device)
    else:
        metrics["peak_memory_allocated_mb"] = 0.0
        metrics["peak_memory_reserved_mb"] = 0.0
        metrics["cuda_device_name"] = None

    metric_dir = ensure_dir(Path(output_dir) / "metrics")
    save_json(metrics, metric_dir / "eval_metrics.json")
    save_json(metrics, metric_dir / "architecture_eval_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one architecture checkpoint with metrics, latency, and peak CUDA memory.")
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/architecture_eval/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--benchmark_batches", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        fused_dir=args.fused_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_value=args.device,
        seed=args.seed,
        threshold=args.threshold,
        warmup_batches=args.warmup_batches,
        benchmark_batches=args.benchmark_batches,
        amp=args.amp,
    )
    print(metrics)


if __name__ == "__main__":
    main()
