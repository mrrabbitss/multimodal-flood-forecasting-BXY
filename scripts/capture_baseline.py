from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from src.data.schemas import depth_scale_from_checkpoint, make_risk_threshold
from src.dataset import channel_names_from_checkpoint
from src.experiments.audit import (
    AUDIT_SCHEMA_VERSION,
    build_artifact_manifest,
    build_file_hash_manifest,
    collect_environment,
    collect_repository_state,
    display_path,
    utc_timestamp,
)
from src.experiments.splits import build_event_split_manifest
from src.training.losses import LossConfig
from src.utils import ensure_dir, load_json, save_json


ROOT_ARTIFACTS = (
    "environment.json",
    "repository_state.json",
    "config.json",
    "metrics.json",
    "file_hashes.json",
)


def _checkpoint_config(checkpoint: dict[str, Any], threshold: float, parameter_count: int) -> dict[str, Any]:
    channel_names = channel_names_from_checkpoint(checkpoint)
    depth_scale = depth_scale_from_checkpoint(checkpoint)
    return {
        "architecture": "ConvLSTMForecastNet",
        "input_len": int(checkpoint["input_len"]),
        "lead_time": int(checkpoint["lead_time"]),
        "input_channels": int(checkpoint["input_channels"]),
        "channel_names": list(channel_names),
        "hidden_channels": int(checkpoint["hidden_channels"]),
        "num_layers": int(checkpoint.get("num_layers", 1)),
        "dropout": float(checkpoint.get("dropout", 0.0)),
        "use_residual": bool(checkpoint.get("use_residual", False)),
        "residual_scale": float(checkpoint.get("residual_scale", 0.35)),
        "parameter_count": int(parameter_count),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "training_seed": int(checkpoint.get("seed", 42)),
        "split_seed": int(checkpoint.get("split_seed", checkpoint.get("seed", 42))),
        "shuffle_split": bool(checkpoint.get("shuffle_split", False)),
        "depth_scale": depth_scale.to_dict(),
        "risk_threshold": make_risk_threshold(threshold, depth_scale).to_dict(),
        "loss_config": LossConfig.from_checkpoint(checkpoint).to_dict(),
        "data_schema": checkpoint.get("data_schema"),
    }


def _require_fresh_output(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in (*ROOT_ARTIFACTS, "audit_manifest.json") if (output_dir / name).exists()]
    evaluation_metrics = output_dir / "evaluation" / "metrics" / "eval_metrics.json"
    if evaluation_metrics.exists():
        existing.append(evaluation_metrics)
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Baseline artifacts already exist; pass --overwrite to replace them: {paths}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a reproducible Conv-LSTM baseline audit bundle.")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="artifacts/baseline")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--benchmark_batches", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fused_dir = Path(args.fused_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not fused_dir.is_dir():
        raise FileNotFoundError(fused_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.warmup_batches < 0 or args.benchmark_batches < 1:
        raise ValueError("warmup_batches must be >= 0 and benchmark_batches must be >= 1")
    _require_fresh_output(output_dir, args.overwrite)

    repository_state = collect_repository_state(REPOSITORY_ROOT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.30))
    split_seed = int(checkpoint.get("split_seed", args.seed))
    shuffle_split = bool(checkpoint.get("shuffle_split", False))
    split_manifest = build_event_split_manifest(fused_dir, seed=split_seed, shuffle=shuffle_split)

    evaluation_dir = output_dir / "evaluation"
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--fused_dir",
        str(fused_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--output_dir",
        str(evaluation_dir),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--threshold",
        str(threshold),
        "--warmup_batches",
        str(args.warmup_batches),
        "--benchmark_batches",
        str(args.benchmark_batches),
    ]
    print(">>> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    metrics = load_json(evaluation_dir / "metrics" / "eval_metrics.json")
    model_config = _checkpoint_config(checkpoint, threshold, int(metrics["parameter_count"]))
    model_config["data_schema"] = metrics.get("data_schema", model_config.get("data_schema"))
    config = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "captured_at_utc": utc_timestamp(),
        "fused_dir": display_path(fused_dir, REPOSITORY_ROOT),
        "checkpoint": display_path(checkpoint_path, REPOSITORY_ROOT),
        "output_dir": display_path(output_dir, REPOSITORY_ROOT),
        "evaluation": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "seed": args.seed,
            "threshold": threshold,
            "warmup_batches": args.warmup_batches,
            "benchmark_batches": args.benchmark_batches,
            "command": command,
        },
        "model": model_config,
        "event_split": split_manifest,
    }
    file_hashes = build_file_hash_manifest(fused_dir, checkpoint_path, REPOSITORY_ROOT)
    ensure_dir(output_dir)
    save_json(collect_environment(), output_dir / "environment.json")
    save_json(repository_state, output_dir / "repository_state.json")
    save_json(config, output_dir / "config.json")
    save_json(metrics, output_dir / "metrics.json")
    save_json(file_hashes, output_dir / "file_hashes.json")
    save_json(build_artifact_manifest(output_dir, ROOT_ARTIFACTS), output_dir / "audit_manifest.json")
    print(f"Baseline audit bundle: {output_dir}")


if __name__ == "__main__":
    main()
