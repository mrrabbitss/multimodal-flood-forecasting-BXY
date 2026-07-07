from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic multimodal flood Conv-LSTM pipeline end-to-end.")
    parser.add_argument("--num_events", type=int, default=20)
    parser.add_argument("--t", type=int, default=72)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output_max", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=0.35)
    parser.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--checkpoint_metric", type=str, default="loss", choices=["loss", "mae", "rmse", "csi", "f1"])
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--shuffle_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--base_dir", type=str, default=".")
    args = parser.parse_args()

    base = Path(args.base_dir)
    raw_dir = base / "data" / "raw"
    aligned_dir = base / "data" / "aligned"
    fused_dir = base / "data" / "fused"
    output_dir = base / "outputs"

    py = sys.executable
    run([
        py,
        "-m",
        "src.generate_synthetic",
        "--num_events",
        str(args.num_events),
        "--t",
        str(args.t),
        "--h",
        str(args.h),
        "--w",
        str(args.w),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(raw_dir),
    ])
    run([
        py,
        "-m",
        "src.align_modalities",
        "--raw_dir",
        str(raw_dir),
        "--out_dir",
        str(aligned_dir),
        "--mode",
        "realtime",
    ])
    run([
        py,
        "-m",
        "src.fuse_dynamic_gate",
        "--aligned_dir",
        str(aligned_dir),
        "--out_dir",
        str(fused_dir),
    ])
    train_cmd = [
        py,
        "-m",
        "src.train",
        "--fused_dir",
        str(fused_dir),
        "--output_dir",
        str(output_dir),
        "--input_len",
        str(args.input_len),
        "--lead_time",
        str(args.lead_time),
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
        "--output_max",
        str(args.output_max),
        "--residual_scale",
        str(args.residual_scale),
        "--seed",
        str(args.seed),
        "--threshold",
        str(args.threshold),
        "--loss_threshold",
        str(args.loss_threshold),
        "--threshold_candidates",
        args.threshold_candidates,
        "--threshold_metric",
        args.threshold_metric,
        "--class_temperature",
        str(args.class_temperature),
        "--bce_loss_weight",
        str(args.bce_loss_weight),
        "--dice_loss_weight",
        str(args.dice_loss_weight),
        "--focal_loss_weight",
        str(args.focal_loss_weight),
        "--checkpoint_metric",
        args.checkpoint_metric,
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--device",
        args.device,
    ]
    if not args.progress:
        train_cmd.append("--no-progress")
    if args.auto_threshold:
        train_cmd.append("--auto_threshold")
    if args.class_threshold is not None:
        train_cmd.extend(["--class_threshold", str(args.class_threshold)])
    if args.split_seed is not None:
        train_cmd.extend(["--split_seed", str(args.split_seed)])
    if args.use_residual:
        train_cmd.append("--use_residual")
    if not args.shuffle_split:
        train_cmd.append("--no-shuffle_split")
    run(train_cmd)
    run([
        py,
        "-m",
        "src.evaluate",
        "--fused_dir",
        str(fused_dir),
        "--checkpoint",
        str(output_dir / "checkpoints" / "best.pt"),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ])
    run([
        py,
        "-m",
        "src.predict_visualize",
        "--fused_dir",
        str(fused_dir),
        "--checkpoint",
        str(output_dir / "checkpoints" / "best.pt"),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
    ])

    print("\nAll steps finished.")
    print(f"Checkpoint: {output_dir / 'checkpoints' / 'best.pt'}")
    print(f"Metrics:    {output_dir / 'metrics'}")
    print(f"Figures:    {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
