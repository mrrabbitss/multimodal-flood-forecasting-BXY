from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodSequenceDataset, infer_num_channels
from .model_variants import build_forecast_model, count_parameters, model_display_name, normalize_model_type
from .train import (
    checkpoint_score,
    combined_depth_mask_loss,
    evaluate_loader,
    parse_float_list,
    score_to_metric_value,
)
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an architecture variant without changing the original Conv-LSTM scripts.")
    parser.add_argument("--model_type", type=str, default="convlstm_attention", choices=["convlstm", "convlstm_attention", "cnn_temporal_transformer"])
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--output_dir", type=str, default="runs/architecture_variant/outputs")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ffn_mult", type=float, default=4.0)
    parser.add_argument("--output_max", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=0.35)
    parser.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--auto_threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold_candidates", type=str, default="0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36")
    parser.add_argument("--threshold_metric", type=str, default="csi", choices=["csi", "f1"])
    parser.add_argument("--class_threshold", type=float, default=None)
    parser.add_argument("--class_temperature", type=float, default=0.04)
    parser.add_argument("--bce_loss_weight", type=float, default=0.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.0)
    parser.add_argument("--focal_loss_weight", type=float, default=0.0)
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

    args.model_type = normalize_model_type(args.model_type)
    set_seed(args.seed)
    threshold_candidates = sorted({float(x) for x in parse_float_list(args.threshold_candidates) + [args.threshold]})
    class_threshold = args.threshold if args.class_threshold is None else args.class_threshold
    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    if len(files) < 3:
        raise ValueError("At least 3 event files are required for train/val/test splits.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    print(f"Using device: {device}")
    print(f"Training model: {model_display_name(args.model_type)}")

    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_idx, val_idx, test_idx = FloodSequenceDataset.split_indices(
        len(files),
        seed=split_seed,
        shuffle=args.shuffle_split,
    )
    train_ds = FloodSequenceDataset(args.fused_dir, train_idx, args.input_len, args.lead_time)
    val_ds = FloodSequenceDataset(args.fused_dir, val_idx, args.input_len, args.lead_time)
    test_ds = FloodSequenceDataset(args.fused_dir, test_idx, args.input_len, args.lead_time)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    input_channels = infer_num_channels(args.fused_dir)
    model = build_forecast_model(
        model_type=args.model_type,
        input_channels=input_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_max=args.output_max,
        residual_scale=args.residual_scale,
        use_residual=args.use_residual,
        attention_dropout=args.attention_dropout,
        transformer_heads=args.transformer_heads,
        transformer_ffn_mult=args.transformer_ffn_mult,
        max_input_len=args.input_len,
    ).to(device)
    parameter_count = count_parameters(model)
    print(f"Trainable parameters: {parameter_count:,}")

    init_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
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
        "val_loss": [],
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
                "model_type": args.model_type,
                "model_label": model_display_name(args.model_type),
                "input_channels": input_channels,
                "hidden_channels": args.hidden,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "attention_dropout": args.attention_dropout,
                "transformer_heads": args.transformer_heads,
                "transformer_ffn_mult": args.transformer_ffn_mult,
                "max_input_len": args.input_len,
                "output_max": args.output_max,
                "residual_scale": args.residual_scale,
                "use_residual": args.use_residual,
                "input_len": args.input_len,
                "lead_time": args.lead_time,
                "threshold": float(val_metrics["threshold"]),
                "loss_threshold": args.loss_threshold,
                "auto_threshold": args.auto_threshold,
                "threshold_candidates": threshold_candidates,
                "threshold_metric": args.threshold_metric,
                "class_threshold": class_threshold,
                "class_temperature": args.class_temperature,
                "bce_loss_weight": args.bce_loss_weight,
                "dice_loss_weight": args.dice_loss_weight,
                "focal_loss_weight": args.focal_loss_weight,
                "checkpoint_metric": args.checkpoint_metric,
                "best_score": best_score,
                "split_seed": split_seed,
                "shuffle_split": args.shuffle_split,
                "best_epoch": epoch,
                "parameter_count": parameter_count,
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
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not args.progress)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                loss = combined_depth_mask_loss(
                    pred,
                    y,
                    args.loss_threshold,
                    class_threshold,
                    args.class_temperature,
                    args.bce_loss_weight,
                    args.dice_loss_weight,
                    args.focal_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(train_losses))

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            args.threshold,
            args.loss_threshold,
            threshold_candidates=val_threshold_candidates,
            threshold_metric=args.threshold_metric,
        )
        scheduler.step(val_metrics["loss"])
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
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
    test_metrics = evaluate_loader(model, test_loader, device, test_threshold, args.loss_threshold)
    test_metrics["model_type"] = args.model_type
    test_metrics["model_label"] = model_display_name(args.model_type)
    test_metrics["parameter_count"] = int(parameter_count)
    test_metrics["train_events"] = train_idx
    test_metrics["val_events"] = val_idx
    test_metrics["test_events"] = test_idx
    test_metrics["runtime_sec"] = float(time.time() - start_time)
    test_metrics["best_epoch"] = int(checkpoint.get("best_epoch", best_epoch))
    test_metrics["split_seed"] = int(split_seed)
    test_metrics["shuffle_split"] = bool(args.shuffle_split)
    test_metrics["threshold"] = float(test_threshold)
    test_metrics["loss_threshold"] = float(args.loss_threshold)
    test_metrics["auto_threshold"] = bool(args.auto_threshold)
    test_metrics["threshold_metric"] = args.threshold_metric
    test_metrics["class_threshold"] = float(class_threshold)
    test_metrics["class_temperature"] = float(args.class_temperature)
    test_metrics["bce_loss_weight"] = float(args.bce_loss_weight)
    test_metrics["dice_loss_weight"] = float(args.dice_loss_weight)
    test_metrics["focal_loss_weight"] = float(args.focal_loss_weight)
    test_metrics["checkpoint_metric"] = args.checkpoint_metric
    save_json(test_metrics, metric_dir / "test_metrics.json")
    save_json(history, metric_dir / "train_history.json")

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_display_name(args.model_type)} Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=180)
    plt.close()

    print("Training finished.")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
