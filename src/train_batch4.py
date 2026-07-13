from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .batch4_dataset import MultiHorizonFloodDataset
from .batch4_engine import configure_determinism, evaluate_batch4_loader, parse_lead_times
from .batch4_models import MODEL_TYPES, build_batch4_model, count_parameters, model_display_name
from .data.schemas import DEFAULT_DEPTH_SCALE, make_depth_scale, make_risk_threshold
from .dataset import FloodSequenceDataset, inspect_dataset_schema, resolve_channel_names
from .experiments.splits import save_event_split_manifest
from .training.losses import LossConfig, build_loss
from .utils import ensure_dir, list_npz_files, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Batch 4 multi-horizon architecture.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="convlstm_unet")
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_channels", default="full")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_times", default="1,3,6,12,24")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--loss_threshold", type=float, default=0.20)
    parser.add_argument("--high_risk_weight", type=float, default=2.5)
    parser.add_argument("--mse_weight", type=float, default=0.25)
    parser.add_argument("--bce_weight", type=float, default=0.05)
    parser.add_argument("--dice_weight", type=float, default=0.05)
    parser.add_argument("--focal_weight", type=float, default=0.0)
    parser.add_argument("--temporal_weight", type=float, default=0.10)
    parser.add_argument("--edge_weight", type=float, default=0.05)
    parser.add_argument("--depth_max", type=float, default=DEFAULT_DEPTH_SCALE.max_value)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=44)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    configure_determinism(args.deterministic)
    lead_times = parse_lead_times(args.lead_times)
    channel_names = resolve_channel_names(args.input_channels)
    depth_scale = make_depth_scale("normalized", args.depth_max)
    data_schema = inspect_dataset_schema(args.fused_dir, channel_names)
    files = [path for path in list_npz_files(args.fused_dir) if path.name.startswith("event_")]
    train_indices, val_indices, test_indices = FloodSequenceDataset.split_indices(
        len(files), seed=args.split_seed, shuffle=True
    )
    split_manifest, split_manifest_path = save_event_split_manifest(
        args.fused_dir, args.output_dir, seed=args.split_seed, shuffle=True
    )
    datasets = {
        "train": MultiHorizonFloodDataset(args.fused_dir, train_indices, args.input_len, lead_times, channel_names),
        "val": MultiHorizonFloodDataset(args.fused_dir, val_indices, args.input_len, lead_times, channel_names),
        "test": MultiHorizonFloodDataset(args.fused_dir, test_indices, args.input_len, lead_times, channel_names),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_batch4_model(
        args.model_type,
        input_channels=len(channel_names),
        hidden_channels=args.hidden,
        num_horizons=len(lead_times),
        dropout=args.dropout,
        output_max=depth_scale.max_value,
    ).to(device)
    parameter_count = count_parameters(model)
    loss_config = LossConfig(
        loss_threshold=args.loss_threshold,
        high_risk_weight=args.high_risk_weight,
        mse_weight=args.mse_weight,
        class_threshold=args.threshold,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        temporal_weight=args.temporal_weight,
        edge_weight=args.edge_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = ensure_dir(Path(args.output_dir) / "checkpoints")
    metric_dir = ensure_dir(Path(args.output_dir) / "metrics")
    figure_dir = ensure_dir(Path(args.output_dir) / "figures")
    print(f"Training {model_display_name(args.model_type)} on {device}; parameters={parameter_count:,}")

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_csi": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}", disable=not args.progress)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(x)
                breakdown = build_loss(prediction, y, loss_config)
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(breakdown.total.detach().item()))
        validation = evaluate_batch4_loader(
            model, loaders["val"], device, lead_times, args.threshold, loss_config
        )["aggregate"]
        train_loss = float(np.mean(losses))
        val_loss = float(validation["loss"])
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(float(validation["mae"]))
        history["val_csi"].append(float(validation["csi"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            f"Epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f} "
            f"MAE={validation['mae']:.5f} CSI={validation['csi']:.4f}"
        )
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model_type,
                    "model_label": model_display_name(args.model_type),
                    "input_channels": len(channel_names),
                    "channel_names": list(channel_names),
                    "data_schema": data_schema,
                    "hidden_channels": args.hidden,
                    "dropout": args.dropout,
                    "input_len": args.input_len,
                    "lead_times": list(lead_times),
                    "threshold": args.threshold,
                    "risk_threshold": make_risk_threshold(args.threshold, depth_scale).to_dict(),
                    "depth_scale": depth_scale.to_dict(),
                    "loss_config": loss_config.to_dict(),
                    "parameter_count": parameter_count,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "shuffle_split": True,
                    "split_manifest": split_manifest,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate_batch4_loader(
        model, loaders["test"], device, lead_times, args.threshold, loss_config, include_per_event=True
    )
    test_result["aggregate"].update(
        {
            "model_type": args.model_type,
            "model_label": model_display_name(args.model_type),
            "parameter_count": parameter_count,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "best_epoch": best_epoch,
            "runtime_sec": float(time.time() - start_time),
            "split_manifest_file": str(split_manifest_path),
        }
    )
    save_json(history, metric_dir / "train_history.json")
    save_json(test_result, metric_dir / "test_metrics.json")
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_display_name(args.model_type))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=180)
    plt.close()
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
