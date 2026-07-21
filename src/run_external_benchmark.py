from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .external_models import (
    EXTERNAL_MODEL_TYPES,
    normalize_external_model_type,
)
from .summarize_external import summarize_external_results
from .utils import ensure_dir, save_json


DEFAULT_MODELS = EXTERNAL_MODEL_TYPES
DEFAULT_SEEDS = (42, 44, 52, 77, 2026)


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(normalize_external_model_type(item.strip()) for item in value.split(",") if item.strip())
    if not models or len(set(models)) != len(models):
        raise ValueError("Models must be unique and non-empty")
    unknown = sorted(set(models) - set(EXTERNAL_MODEL_TYPES))
    if unknown:
        raise ValueError(f"Unknown external models: {unknown}")
    return models


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and non-empty")
    return seeds


def parse_model_lrs(value: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Model learning rates must use model=value entries")
        raw_model, raw_value = item.split("=", 1)
        model_type = normalize_external_model_type(raw_model)
        if model_type in output:
            raise ValueError(f"Duplicate learning rate for {model_type}")
        learning_rate = float(raw_value)
        if learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        output[model_type] = learning_rate
    return output


def build_train_command(args: argparse.Namespace, model_type: str, seed: int, output_dir: Path) -> list[str]:
    learning_rate = args.model_lr_map.get(model_type, args.lr)
    command = [
        sys.executable,
        "-m",
        "src.train_external",
        "--dataset",
        args.dataset,
        "--model_type",
        model_type,
        "--output_dir",
        str(output_dir),
        "--input_len",
        str(args.input_len),
        "--lead_times",
        args.lead_times,
        "--patch_size",
        str(args.patch_size),
        "--train_patch_stride",
        str(args.train_patch_stride),
        "--eval_patch_stride",
        str(args.eval_patch_stride),
        "--max_train_samples_per_event",
        str(args.max_train_samples_per_event),
        "--max_eval_samples_per_event",
        str(args.max_eval_samples_per_event),
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
        "--attention_dropout",
        str(args.attention_dropout),
        "--transformer_heads",
        str(args.transformer_heads),
        "--fno_modes",
        str(args.fno_modes),
        "--fno_layers",
        str(args.fno_layers),
        "--simvp_blocks",
        str(args.simvp_blocks),
        "--residual_scale",
        str(args.residual_scale),
        "--lr",
        str(learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--min_delta",
        str(args.min_delta),
        "--seed",
        str(seed),
        "--split_seed",
        str(args.split_seed),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--no-progress",
    ]
    if args.amp:
        command.append("--amp")
    else:
        command.append("--no-amp")
    command.append("--use_residual" if args.use_residual else "--no-use_residual")
    if args.dataset == "urbanflood24":
        command.extend(["--urban_root", args.urban_root, "--location", args.location])
    else:
        command.extend(["--larno_root", args.larno_root])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable multi-seed external physical-data benchmark.")
    parser.add_argument("--dataset", choices=["urbanflood24", "larno_ukea"], required=True)
    parser.add_argument("--urban_root", default="../urbanflood24")
    parser.add_argument("--larno_root", default="../external_datasets/larno_ukea_8m_5min")
    parser.add_argument("--location", default="location1", choices=["location1", "location2", "location3"])
    parser.add_argument("--output_root", default="runs/external_physical/benchmark_v1")
    parser.add_argument("--summary_dir", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_times", default="1,3,6,12")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--train_patch_stride", type=int, default=32)
    parser.add_argument("--eval_patch_stride", type=int, default=64)
    parser.add_argument("--max_train_samples_per_event", type=int, default=64)
    parser.add_argument("--max_eval_samples_per_event", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--model_lrs",
        default="",
        help="Optional comma-separated overrides such as urnn_lite=0.0001,fno=0.0003",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds)
    args.model_lr_map = parse_model_lrs(args.model_lrs)
    unknown_lr_models = sorted(set(args.model_lr_map) - set(models))
    if unknown_lr_models:
        raise ValueError(f"Learning-rate overrides supplied for models not in this run: {unknown_lr_models}")
    root = ensure_dir(args.output_root)
    location = args.location if args.dataset == "urbanflood24" else "ukea"
    dataset_root = ensure_dir(root / args.dataset / location)
    save_json(
        {
            "schema_version": "external_benchmark_config_v2",
            "dataset": args.dataset,
            "location": location,
            "models": list(models),
            "seeds": list(seeds),
            "protocol": {
                "resolution_m": 8,
                "time_step_minutes": 5,
                "rain_forcing": "past_only",
                "prediction_mode": "residual" if args.use_residual else "absolute",
                "split": "event_disjoint",
                "evaluation_sampling_seed": args.split_seed,
            },
            "configuration": vars(args),
        },
        dataset_root / "benchmark_config.json",
    )
    if not args.summarize_only:
        for model_type in models:
            for seed in seeds:
                output_dir = dataset_root / model_type / f"seed_{seed}"
                metrics_path = output_dir / "metrics" / "test_metrics.json"
                if args.skip_existing and metrics_path.exists():
                    print(f"Skipping completed run: {metrics_path}", flush=True)
                    continue
                command = build_train_command(args, model_type, seed, output_dir)
                print("\n>>> " + subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
    if args.dry_run:
        return
    summary_dir = Path(args.summary_dir) if args.summary_dir else root / "summary"
    summary = summarize_external_results(root, summary_dir)
    print(f"External benchmark complete: {summary['run_count']} runs")
    print(f"Summary: {summary_dir / 'external_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
