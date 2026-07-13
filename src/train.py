from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.schemas import DEFAULT_DEPTH_SCALE, make_depth_scale, make_risk_threshold
from .dataset import (
    FloodSequenceDataset,
    channel_names_for_data,
    channel_names_from_checkpoint,
    infer_num_channels,
    inspect_dataset_schema,
    resolve_channel_names,
)
from .experiments.splits import save_event_split_manifest
from .metrics import all_metrics
from .model import ConvLSTMForecastNet
from .training.losses import (
    LossConfig,
    build_loss,
    mask_loss_terms as shared_mask_loss_terms,
    weighted_depth_loss as shared_weighted_depth_loss,
)
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def parse_float_list(value: str | None) -> list[float]:
    if value is None:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def weighted_depth_loss(pred: torch.Tensor, target: torch.Tensor, high_threshold: float = 0.20) -> torch.Tensor:
    """Backward-compatible wrapper for the original weighted depth loss."""
    return shared_weighted_depth_loss(pred, target, LossConfig(loss_threshold=high_threshold))


def mask_loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return shared_mask_loss_terms(pred, target, threshold, temperature)


def combined_depth_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_threshold: float,
    class_threshold: float,
    class_temperature: float,
    bce_weight: float,
    dice_weight: float,
    focal_weight: float,
) -> torch.Tensor:
    config = LossConfig(
        loss_threshold=loss_threshold,
        class_threshold=class_threshold,
        class_temperature=class_temperature,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
    )
    return build_loss(pred, target, config).total


def select_threshold(pred: np.ndarray, target: np.ndarray, candidates: list[float], metric: str) -> dict:
    best_metrics: dict | None = None
    best_key: tuple[float, float, float] | None = None
    best_threshold = float(candidates[0])
    for threshold in candidates:
        m = all_metrics(pred, target, threshold=float(threshold))
        score = float(m[metric])
        key = (score, float(m["f1"]), -float(m["mae"]))
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = m
            best_threshold = float(threshold)
    if best_metrics is None:
        raise ValueError("threshold candidate list is empty")
    best_metrics["threshold"] = best_threshold
    return best_metrics


def checkpoint_score(metrics: dict, metric_name: str) -> float:
    value = float(metrics[metric_name])
    if metric_name in {"loss", "mae", "rmse"}:
        return -value
    return value


def score_to_metric_value(score: float, metric_name: str) -> float:
    if metric_name in {"loss", "mae", "rmse"}:
        return -score
    return score


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_threshold: float,
    loss_threshold: float,
    threshold_candidates: list[float] | None = None,
    threshold_metric: str = "csi",
    loss_config: LossConfig | None = None,
) -> dict:
    model.eval()
    preds = []
    targets = []
    loss_values: dict[str, list[float]] = {
        "loss_total": [],
        "loss_depth": [],
        "loss_bce": [],
        "loss_dice": [],
        "loss_focal": [],
        "loss_temporal": [],
        "loss_edge": [],
    }
    if loss_config is None:
        loss_config = LossConfig(loss_threshold=loss_threshold, class_threshold=metric_threshold)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            breakdown = build_loss(pred, y, loss_config)
            for key, value in breakdown.detached_values().items():
                loss_values[key].append(value)
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    if threshold_candidates:
        m = select_threshold(pred_np, target_np, threshold_candidates, threshold_metric)
    else:
        m = all_metrics(pred_np, target_np, threshold=metric_threshold)
        m["threshold"] = float(metric_threshold)
    for key, values in loss_values.items():
        m[key] = float(np.mean(values))
    m["loss"] = m["loss_total"]
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--depth_scale_mode", type=str, choices=["normalized"], default="normalized")
    parser.add_argument("--depth_max", type=float, default=DEFAULT_DEPTH_SCALE.max_value)
    parser.add_argument("--output_max", type=float, default=None, help="Deprecated alias for --depth_max")
    parser.add_argument("--residual_scale", type=float, default=0.35)
    parser.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--input_channels",
        type=str,
        default="auto",
        help="auto, default, legacy, or a comma-separated channel list",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--auto_threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold_candidates", type=str, default="0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36")
    parser.add_argument("--threshold_metric", type=str, default="csi", choices=["csi", "f1"])
    parser.add_argument("--class_threshold", type=float, default=None)
    parser.add_argument("--class_temperature", type=float, default=0.04)
    parser.add_argument("--bce_loss_weight", type=float, default=0.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.0)
    parser.add_argument("--focal_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--scheduler_monitor",
        type=str,
        default="loss_total",
        choices=["loss_total", "loss_depth", "mae", "rmse"],
    )
    parser.add_argument("--checkpoint_metric", type=str, default="loss", choices=["loss", "mae", "rmse", "csi", "f1"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--shuffle_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    if args.output_max is not None:
        args.depth_max = float(args.output_max)
    depth_scale = make_depth_scale(args.depth_scale_mode, args.depth_max)
    channel_names = (
        channel_names_for_data(args.fused_dir)
        if args.input_channels == "auto"
        else resolve_channel_names(args.input_channels)
    )
    data_schema = inspect_dataset_schema(args.fused_dir, channel_names)
    threshold_candidates = sorted({float(x) for x in parse_float_list(args.threshold_candidates) + [args.threshold]})
    class_threshold = args.threshold if args.class_threshold is None else args.class_threshold
    risk_threshold = make_risk_threshold(args.threshold, depth_scale)
    loss_config = LossConfig(
        loss_threshold=args.loss_threshold,
        class_threshold=class_threshold,
        class_temperature=args.class_temperature,
        bce_weight=args.bce_loss_weight,
        dice_weight=args.dice_loss_weight,
        focal_weight=args.focal_loss_weight,
    )
    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    if len(files) < 3:
        raise ValueError("At least 3 events are recommended for train/val/test. Generate more events or use the quick demo with 6 events.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    print(f"Using device: {device}")

    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_idx, val_idx, test_idx = FloodSequenceDataset.split_indices(
        len(files),
        seed=split_seed,
        shuffle=args.shuffle_split,
    )
    split_manifest, split_manifest_path = save_event_split_manifest(
        args.fused_dir, args.output_dir, seed=split_seed, shuffle=args.shuffle_split
    )
    train_ds = FloodSequenceDataset(
        args.fused_dir, train_idx, args.input_len, args.lead_time, channel_names=channel_names
    )
    val_ds = FloodSequenceDataset(
        args.fused_dir, val_idx, args.input_len, args.lead_time, channel_names=channel_names
    )
    test_ds = FloodSequenceDataset(
        args.fused_dir, test_idx, args.input_len, args.lead_time, channel_names=channel_names
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    input_channels = infer_num_channels(args.fused_dir, channel_names=channel_names)
    fused_channel = channel_names.index("fused_depth") if "fused_depth" in channel_names else -1
    model = ConvLSTMForecastNet(
        input_channels=input_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_max=depth_scale.max_value,
        residual_scale=args.residual_scale,
        use_residual=args.use_residual,
        fused_channel=fused_channel,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    print(f"Trainable parameters: {parameter_count:,}")
    init_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        init_channel_names = channel_names_from_checkpoint(init_checkpoint)
        if init_channel_names != channel_names:
            raise ValueError(
                "Initial checkpoint channel schema does not match the requested training schema: "
                f"checkpoint={init_channel_names}, requested={channel_names}"
            )
        model.load_state_dict(init_checkpoint["model_state"])
        print(f"Initialized model from {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = ensure_dir(Path(args.output_dir) / "checkpoints")
    fig_dir = ensure_dir(Path(args.output_dir) / "figures")
    metric_dir = ensure_dir(Path(args.output_dir) / "metrics")

    best_score = -float("inf")
    history = {
        "train_loss": [],
        "train_loss_depth": [],
        "train_loss_bce": [],
        "train_loss_dice": [],
        "train_loss_focal": [],
        "train_loss_temporal": [],
        "train_loss_edge": [],
        "val_loss": [],
        "val_loss_depth": [],
        "val_loss_bce": [],
        "val_loss_dice": [],
        "val_loss_focal": [],
        "val_loss_temporal": [],
        "val_loss_edge": [],
        "val_mae": [],
        "val_csi": [],
        "val_threshold": [],
        "lr": [],
    }
    start_time = time.time()
    best_epoch = 0
    epochs_without_improvement = 0
    val_threshold_candidates = threshold_candidates if args.auto_threshold else None

    def save_best_checkpoint(epoch: int, val_metrics: dict) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "input_channels": input_channels,
                "channel_names": list(channel_names),
                "data_schema": data_schema,
                "fused_channel": fused_channel,
                "hidden_channels": args.hidden,
                "parameter_count": parameter_count,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "output_max": depth_scale.max_value,
                "depth_scale": depth_scale.to_dict(),
                "residual_scale": args.residual_scale,
                "use_residual": args.use_residual,
                "input_len": args.input_len,
                "lead_time": args.lead_time,
                "threshold": float(val_metrics["threshold"]),
                "risk_threshold": make_risk_threshold(float(val_metrics["threshold"]), depth_scale).to_dict(),
                "loss_threshold": args.loss_threshold,
                "loss_config": loss_config.to_dict(),
                "auto_threshold": args.auto_threshold,
                "threshold_candidates": threshold_candidates,
                "threshold_metric": args.threshold_metric,
                "class_threshold": class_threshold,
                "class_temperature": args.class_temperature,
                "bce_loss_weight": args.bce_loss_weight,
                "dice_loss_weight": args.dice_loss_weight,
                "focal_loss_weight": args.focal_loss_weight,
                "scheduler_monitor": args.scheduler_monitor,
                "checkpoint_metric": args.checkpoint_metric,
                "best_score": best_score,
                "split_seed": split_seed,
                "shuffle_split": args.shuffle_split,
                "split_manifest": split_manifest,
                "best_epoch": epoch,
                "args": vars(args),
                "val_metrics": val_metrics,
            },
            ckpt_dir / "best.pt",
        )

    if init_checkpoint is not None:
        initial_val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            args.threshold,
            args.loss_threshold,
            threshold_candidates=val_threshold_candidates,
            threshold_metric=args.threshold_metric,
            loss_config=loss_config,
        )
        best_score = checkpoint_score(initial_val_metrics, args.checkpoint_metric)
        save_best_checkpoint(0, initial_val_metrics)
        print(
            f"Initial checkpoint: val_loss={initial_val_metrics['loss']:.5f}, "
            f"val_mae={initial_val_metrics['mae']:.5f}, val_csi={initial_val_metrics['csi']:.4f}, "
            f"val_thr={initial_val_metrics['threshold']:.2f}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: dict[str, list[float]] = {
            "loss_total": [],
            "loss_depth": [],
            "loss_bce": [],
            "loss_dice": [],
            "loss_focal": [],
            "loss_temporal": [],
            "loss_edge": [],
        }
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not args.progress)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                breakdown = build_loss(pred, y, loss_config)
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            values = breakdown.detached_values()
            for key in train_losses:
                train_losses[key].append(values[key])
            pbar.set_postfix(loss=np.mean(train_losses["loss_total"]))

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            args.threshold,
            args.loss_threshold,
            threshold_candidates=val_threshold_candidates,
            threshold_metric=args.threshold_metric,
            loss_config=loss_config,
        )
        scheduler.step(val_metrics[args.scheduler_monitor])
        train_loss = float(np.mean(train_losses["loss_total"]))
        history["train_loss"].append(train_loss)
        history["train_loss_depth"].append(float(np.mean(train_losses["loss_depth"])))
        history["train_loss_bce"].append(float(np.mean(train_losses["loss_bce"])))
        history["train_loss_dice"].append(float(np.mean(train_losses["loss_dice"])))
        history["train_loss_focal"].append(float(np.mean(train_losses["loss_focal"])))
        history["train_loss_temporal"].append(float(np.mean(train_losses["loss_temporal"])))
        history["train_loss_edge"].append(float(np.mean(train_losses["loss_edge"])))
        history["val_loss"].append(val_metrics["loss"])
        history["val_loss_depth"].append(val_metrics["loss_depth"])
        history["val_loss_bce"].append(val_metrics["loss_bce"])
        history["val_loss_dice"].append(val_metrics["loss_dice"])
        history["val_loss_focal"].append(val_metrics["loss_focal"])
        history["val_loss_temporal"].append(val_metrics["loss_temporal"])
        history["val_loss_edge"].append(val_metrics["loss_edge"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_csi"].append(val_metrics["csi"])
        history["val_threshold"].append(val_metrics["threshold"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f}, val_loss={val_metrics['loss']:.5f}, "
            f"val_mae={val_metrics['mae']:.5f}, val_csi={val_metrics['csi']:.4f}, "
            f"val_thr={val_metrics['threshold']:.2f}"
        )

        current_score = checkpoint_score(val_metrics, args.checkpoint_metric)
        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint(best_epoch, val_metrics)
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                best_metric_value = score_to_metric_value(best_score, args.checkpoint_metric)
                print(
                    f"Early stopping at epoch {epoch}; best epoch was {best_epoch} with "
                    f"val_{args.checkpoint_metric}={best_metric_value:.5f}."
                )
                break

    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_threshold = float(checkpoint.get("threshold", args.threshold))
    test_metrics = evaluate_loader(
        model,
        test_loader,
        device,
        test_threshold,
        args.loss_threshold,
        loss_config=loss_config,
    )
    test_metrics["train_events"] = train_idx
    test_metrics["val_events"] = val_idx
    test_metrics["test_events"] = test_idx
    test_metrics["runtime_sec"] = float(time.time() - start_time)
    test_metrics["best_epoch"] = int(checkpoint.get("best_epoch", best_epoch))
    test_metrics["split_seed"] = int(split_seed)
    test_metrics["shuffle_split"] = bool(args.shuffle_split)
    test_metrics["split_manifest_file"] = str(split_manifest_path)
    test_metrics["threshold"] = float(test_threshold)
    test_metrics["risk_threshold"] = make_risk_threshold(test_threshold, depth_scale).to_dict()
    test_metrics["depth_scale"] = depth_scale.to_dict()
    test_metrics["channel_names"] = list(channel_names)
    test_metrics["parameter_count"] = parameter_count
    test_metrics["data_schema"] = data_schema
    test_metrics["loss_threshold"] = float(args.loss_threshold)
    test_metrics["auto_threshold"] = bool(args.auto_threshold)
    test_metrics["threshold_metric"] = args.threshold_metric
    test_metrics["class_threshold"] = float(class_threshold)
    test_metrics["class_temperature"] = float(args.class_temperature)
    test_metrics["bce_loss_weight"] = float(args.bce_loss_weight)
    test_metrics["dice_loss_weight"] = float(args.dice_loss_weight)
    test_metrics["focal_loss_weight"] = float(args.focal_loss_weight)
    test_metrics["scheduler_monitor"] = args.scheduler_monitor
    test_metrics["checkpoint_metric"] = args.checkpoint_metric
    test_metrics["loss_config"] = loss_config.to_dict()
    save_json(test_metrics, metric_dir / "test_metrics.json")
    save_json(history, metric_dir / "train_history.json")

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conv-LSTM Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=180)
    plt.close()

    print("Training finished.")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
