from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .external_data import ExternalFloodDataset, discover_larno_ukea, discover_urbanflood24
from .external_models import build_external_model_from_checkpoint
from .model_variants import model_display_name
from .summarize_external import DATASET_LABELS, MODEL_COLORS, MODEL_ORDER
from .utils import ensure_dir, save_json


def _load_checkpoints(root: Path, seed: int) -> dict[str, dict]:
    checkpoints = {}
    for model_type in MODEL_ORDER:
        path = root / model_type / f"seed_{seed}" / "checkpoints" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["_path"] = path.as_posix()
        checkpoints[model_type] = checkpoint
    reference = checkpoints[MODEL_ORDER[0]]
    signature = (
        reference["dataset"],
        reference["location"],
        tuple(reference["lead_times"]),
        tuple(reference["args"].get("thresholds", "").split(",")),
        int(reference["split_seed"]),
    )
    for checkpoint in checkpoints.values():
        candidate = (
            checkpoint["dataset"],
            checkpoint["location"],
            tuple(checkpoint["lead_times"]),
            tuple(checkpoint["args"].get("thresholds", "").split(",")),
            int(checkpoint["split_seed"]),
        )
        if candidate != signature:
            raise ValueError("Checkpoints do not share an evaluation protocol")
    return checkpoints


def _build_test_dataset(checkpoint: dict) -> ExternalFloodDataset:
    args = checkpoint["args"]
    if checkpoint["dataset"] == "urbanflood24":
        events = discover_urbanflood24(args["urban_root"], "test", checkpoint["location"])
    else:
        events = discover_larno_ukea(args["larno_root"], "test")
    expected_events = tuple(checkpoint.get("test_events", []))
    if expected_events and tuple(event.event_id for event in events) != expected_events:
        raise ValueError("Discovered test events differ from checkpoint metadata")
    evaluation_cap = int(args.get("max_eval_samples_per_event", 0)) or None
    return ExternalFloodDataset(
        events,
        input_len=int(checkpoint["input_len"]),
        lead_times=tuple(int(value) for value in checkpoint["lead_times"]),
        patch_size=int(args.get("patch_size", 64)),
        patch_stride=int(args.get("eval_patch_stride", 64)),
        max_samples_per_event=evaluation_cap,
        seed=int(checkpoint["seed"]),
        depth_scale_m=float(checkpoint["depth_scale_m"]),
        rain_scale_mm_5min=float(checkpoint["rain_scale_mm_5min"]),
    )


def _representative_index(dataset: ExternalFloodDataset, requested_index: int) -> tuple[int, float]:
    if requested_index >= 0:
        if requested_index >= len(dataset):
            raise IndexError(f"sample_index={requested_index} exceeds dataset length {len(dataset)}")
        return requested_index, 0.0
    best_index = 0
    best_score = -1.0
    for index in range(len(dataset)):
        sample = dataset[index]
        mask = sample["valid_mask"][0].bool()
        target = sample["target"][-1][mask]
        observed = sample["x"][-1, 0][mask] * dataset.depth_scale_m
        if target.numel() == 0:
            continue
        change = torch.mean(torch.abs(target - observed)).item()
        wet_fraction = torch.mean((target >= 0.10).float()).item()
        score = change + 0.05 * wet_fraction + 0.02 * float(target.max().item())
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def _crop(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        return array
    return array[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def _sample_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    valid_prediction = prediction[mask]
    valid_target = target[mask]
    difference = valid_prediction - valid_target
    predicted_wet = valid_prediction >= threshold
    target_wet = valid_target >= threshold
    true_positive = int(np.logical_and(predicted_wet, target_wet).sum())
    false_positive = int(np.logical_and(predicted_wet, ~target_wet).sum())
    false_negative = int(np.logical_and(~predicted_wet, target_wet).sum())
    return {
        "mae_cm": float(100.0 * np.mean(np.abs(difference))),
        "rmse_cm": float(100.0 * np.sqrt(np.mean(np.square(difference)))),
        "csi": float(true_positive / max(true_positive + false_positive + false_negative, 1)),
    }


def _add_truth_contour(axis: plt.Axes, truth: np.ndarray, threshold: float) -> None:
    wet = truth >= threshold
    if wet.any() and (~wet).any():
        axis.contour(wet.astype(float), levels=[0.5], colors="black", linewidths=0.55)


def _plot_forecasts(
    fields: dict[str, np.ndarray],
    truth: np.ndarray,
    threshold: float,
    title: str,
    path: Path,
) -> None:
    labels = ["Last observation", "Target", "Persistence"] + [model_display_name(model) for model in MODEL_ORDER]
    keys = ["observation", "target", "persistence", *MODEL_ORDER]
    maximum = max(float(np.percentile(fields[key], 99.5)) for key in keys)
    maximum = max(maximum, threshold * 1.2, 1e-3)
    figure, axes = plt.subplots(1, len(keys), figsize=(18, 3.7), constrained_layout=True)
    image = None
    for axis, key, label in zip(axes, keys, labels):
        image = axis.imshow(fields[key], cmap="Blues", vmin=0.0, vmax=maximum)
        if key not in {"observation", "target"}:
            _add_truth_contour(axis, truth, threshold)
        axis.set_title(label, color=MODEL_COLORS.get(key, "#222222"))
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.78, label="Water depth (m)")
    figure.suptitle(f"{title}\nBlack contour: target flood extent at {threshold:.2f} m")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_long_horizon_errors(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    title: str,
    path: Path,
) -> None:
    keys = ["persistence", *MODEL_ORDER]
    labels = ["Persistence", *[model_display_name(model) for model in MODEL_ORDER]]
    errors = {key: np.abs(predictions[key][-1] - target[-1]) for key in keys}
    maximum = max(float(np.percentile(error[mask], 99.0)) for error in errors.values())
    maximum = max(maximum, 1e-4)
    figure, axes = plt.subplots(1, len(keys), figsize=(13, 3.6), constrained_layout=True)
    image = None
    for axis, key, label in zip(axes, keys, labels):
        metric = _sample_metrics(predictions[key][-1], target[-1], mask, threshold)
        image = axis.imshow(errors[key], cmap="magma", vmin=0.0, vmax=maximum)
        axis.set_title(f"{label}\nMAE {metric['mae_cm']:.2f} cm | CSI {metric['csi']:.3f}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.78, label="Absolute error (m)")
    figure.suptitle(title)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_horizon_error_matrix(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    lead_minutes: list[int],
    title: str,
    path: Path,
) -> None:
    keys = ["persistence", *MODEL_ORDER]
    labels = ["Persistence", *[model_display_name(model) for model in MODEL_ORDER]]
    errors = np.stack([np.abs(predictions[key] - target) for key in keys], axis=0)
    maximum = max(float(np.percentile(errors[:, horizon, mask], 99.0)) for horizon in range(target.shape[0]))
    maximum = max(maximum, 1e-4)
    figure, axes = plt.subplots(len(keys), len(lead_minutes), figsize=(13, 10), constrained_layout=True)
    image = None
    for row, (key, label) in enumerate(zip(keys, labels)):
        for column, lead in enumerate(lead_minutes):
            axis = axes[row, column]
            image = axis.imshow(errors[row, column], cmap="magma", vmin=0.0, vmax=maximum)
            mae = 100.0 * float(np.mean(errors[row, column][mask]))
            if row == 0:
                axis.set_title(f"+{lead} min")
            if column == 0:
                axis.set_ylabel(label)
            axis.text(0.03, 0.04, f"MAE {mae:.2f} cm", transform=axis.transAxes, color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.72, label="Absolute error (m)")
    figure.suptitle(title)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spatial predictions from external physical-data checkpoints.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    checkpoints = _load_checkpoints(Path(args.checkpoint_root), args.seed)
    reference = checkpoints[MODEL_ORDER[0]]
    dataset = _build_test_dataset(reference)
    sample_index, selection_score = _representative_index(dataset, args.sample_index)
    sample = dataset[sample_index]
    x = sample["x"].unsqueeze(0).to(device)
    target_full = sample["target"].numpy()
    mask_full = sample["valid_mask"][0].numpy().astype(bool)
    persistence_full = np.repeat((sample["x"][-1, 0].numpy() * float(reference["depth_scale_m"]))[None], len(reference["lead_times"]), axis=0)
    predictions_full = {"persistence": persistence_full}
    for model_type, checkpoint in checkpoints.items():
        model = build_external_model_from_checkpoint(checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.inference_mode():
            predictions_full[model_type] = model(x).squeeze(0).float().cpu().numpy()

    target = np.stack([_crop(frame, mask_full) for frame in target_full])
    mask = _crop(mask_full, mask_full)
    predictions = {
        key: np.stack([_crop(frame, mask_full) for frame in values])
        for key, values in predictions_full.items()
    }
    observation = _crop(sample["x"][-1, 0].numpy() * float(reference["depth_scale_m"]), mask_full)
    lead_minutes = [5 * int(value) for value in reference["lead_times"]]
    dataset_label = DATASET_LABELS.get(str(reference["dataset"]), str(reference["dataset"]))
    location_label = str(reference["location"]).replace("location", "Location ").replace("ukea", "UKEA")
    group_label = dataset_label if reference["dataset"] == "larno_ukea" else f"{dataset_label} / {location_label}"
    title = f"{group_label} representative test sample, +{lead_minutes[-1]} min"
    output = ensure_dir(args.output_dir)
    fields = {
        "observation": observation,
        "target": target[-1],
        "persistence": predictions["persistence"][-1],
        **{model: predictions[model][-1] for model in MODEL_ORDER},
    }
    prefix = f"{reference['dataset']}_{reference['location']}"
    _plot_forecasts(fields, target[-1], args.threshold, title, output / f"{prefix}_spatial_forecast.png")
    _plot_long_horizon_errors(predictions, target, mask, args.threshold, title, output / f"{prefix}_spatial_error.png")
    _plot_horizon_error_matrix(predictions, target, mask, lead_minutes, title, output / f"{prefix}_horizon_error_matrix.png")

    metrics = {}
    for key, values in predictions.items():
        metrics[key] = [
            {"lead_minutes": lead, **_sample_metrics(values[index], target[index], mask, args.threshold)}
            for index, lead in enumerate(lead_minutes)
        ]
    save_json(
        {
            "schema_version": "external_spatial_showcase_v1",
            "dataset": reference["dataset"],
            "location": reference["location"],
            "seed": args.seed,
            "sample_index": sample_index,
            "selection_score": selection_score,
            "event_id": sample["event_id"],
            "start": int(sample["start"]),
            "patch_y": int(sample["patch_y"]),
            "patch_x": int(sample["patch_x"]),
            "lead_minutes": lead_minutes,
            "threshold_m": args.threshold,
            "metrics": metrics,
            "checkpoints": {model: checkpoint["_path"] for model, checkpoint in checkpoints.items()},
        },
        output / f"{prefix}_spatial_showcase.json",
    )
    print(f"Spatial showcase sample {sample_index}: {sample['event_id']}")
    print(f"Figures: {output.resolve()}")


if __name__ == "__main__":
    main()
