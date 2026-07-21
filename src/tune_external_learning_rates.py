from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .external_models import EXTERNAL_MODEL_TYPES, external_model_display_name
from .run_external_benchmark import parse_models
from .train_external import parse_float_list
from .utils import ensure_dir, load_json, save_json


def _learning_rate_slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _command(
    args: argparse.Namespace,
    model_type: str,
    learning_rate: float,
    output_dir: Path,
) -> list[str]:
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
        "0",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--num_layers",
        str(args.num_layers),
        "--transformer_heads",
        str(args.transformer_heads),
        "--fno_modes",
        str(args.fno_modes),
        "--fno_layers",
        str(args.fno_layers),
        "--simvp_blocks",
        str(args.simvp_blocks),
        "--lr",
        str(learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed),
        "--split_seed",
        str(args.split_seed),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--no-progress",
        "--selection_only",
        "--use_residual",
    ]
    command.append("--amp" if args.amp else "--no-amp")
    if args.dataset == "urbanflood24":
        command.extend(["--urban_root", args.urban_root, "--location", args.location])
    else:
        command.extend(["--larno_root", args.larno_root])
    return command


def _summarize(root: Path, models: tuple[str, ...], learning_rates: tuple[float, ...]) -> dict:
    rows = []
    validation_signature = None
    for model_type in models:
        for learning_rate in learning_rates:
            path = (
                root
                / model_type
                / f"lr_{_learning_rate_slug(learning_rate)}"
                / "metrics"
                / "selection_metrics.json"
            )
            metrics = load_json(path)
            signature = (int(metrics["split_seed"]), tuple(metrics["validation_events"]))
            if validation_signature is None:
                validation_signature = signature
            elif signature != validation_signature:
                raise ValueError("Learning-rate candidates used different validation events")
            rows.append(
                {
                    "model_type": model_type,
                    "model_label": external_model_display_name(model_type),
                    "learning_rate": float(learning_rate),
                    "best_validation_loss": float(metrics["best_validation_loss"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_ran": int(metrics["epochs_ran"]),
                    "validation_mae_cm": float(
                        np.mean([row["mae_cm"] for row in metrics["per_horizon"]])
                    ),
                    "validation_csi": float(
                        np.mean([row["csi"] for row in metrics["per_horizon"]])
                    ),
                    "training_time_sec": float(metrics["training_time_sec"]),
                    "test_evaluated": bool(metrics["test_evaluated"]),
                }
            )

    selected = []
    for model_type in models:
        candidates = [row for row in rows if row["model_type"] == model_type]
        selected.append(min(candidates, key=lambda row: row["best_validation_loss"]))

    with (root / "learning_rate_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "selected_learning_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for model_type in models:
        values = sorted(
            (row for row in rows if row["model_type"] == model_type),
            key=lambda row: row["learning_rate"],
        )
        axis.plot(
            [row["learning_rate"] for row in values],
            [row["best_validation_loss"] for row in values],
            marker="o",
            label=external_model_display_name(model_type),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Learning rate")
    axis.set_ylabel("Best validation loss (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "learning_rate_selection.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# External Baseline Learning-Rate Selection",
        "",
        "Candidates are selected on the event-disjoint validation split. Test metrics are not computed during this stage.",
        "",
        "| Model | Selected LR | Validation loss | Validation MAE (cm) | Validation CSI | Best epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['model_label']} | {row['learning_rate']:.1e} | "
            f"{row['best_validation_loss']:.5f} | {row['validation_mae_cm']:.3f} | "
            f"{row['validation_csi']:.4f} | {row['best_epoch']} |"
        )
    lines.extend(["", "![Learning-rate selection](learning_rate_selection.png)", ""])
    (root / "LEARNING_RATE_SELECTION.md").write_text("\n".join(lines), encoding="utf-8")

    output = {
        "schema_version": "external_lr_selection_v1",
        "models": list(models),
        "candidate_learning_rates": list(learning_rates),
        "test_evaluated": False,
        "selected": selected,
        "candidates": rows,
    }
    save_json(output, root / "learning_rate_selection.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select external-baseline learning rates without evaluating the test split."
    )
    parser.add_argument("--dataset", choices=["urbanflood24", "larno_ukea"], required=True)
    parser.add_argument("--urban_root", default="../urbanflood24")
    parser.add_argument("--larno_root", default="../external_datasets/larno_ukea_8m_5min")
    parser.add_argument("--location", default="location1", choices=["location1", "location2", "location3"])
    parser.add_argument("--output_root", default="runs/external_physical/lr_selection_v2")
    parser.add_argument("--models", default=",".join(EXTERNAL_MODEL_TYPES))
    parser.add_argument("--learning_rates", default="0.0001,0.0003,0.001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_times", default="1,3,6,12")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--train_patch_stride", type=int, default=32)
    parser.add_argument("--eval_patch_stride", type=int, default=64)
    parser.add_argument("--max_train_samples_per_event", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--fno_modes", type=int, default=8)
    parser.add_argument("--fno_layers", type=int, default=3)
    parser.add_argument("--simvp_blocks", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    models = parse_models(args.models)
    learning_rates = parse_float_list(args.learning_rates)
    location = args.location if args.dataset == "urbanflood24" else "ukea"
    root = ensure_dir(Path(args.output_root) / args.dataset / location)
    for model_type in models:
        for learning_rate in learning_rates:
            output_dir = root / model_type / f"lr_{_learning_rate_slug(learning_rate)}"
            metrics_path = output_dir / "metrics" / "selection_metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping completed selection run: {metrics_path}", flush=True)
                continue
            command = _command(args, model_type, learning_rate, output_dir)
            print("\n>>> " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True)
    selection = _summarize(root, models, learning_rates)
    print("Selected learning rates:")
    for row in selection["selected"]:
        print(f"  {row['model_type']}: {row['learning_rate']:.1e}")


if __name__ == "__main__":
    main()
