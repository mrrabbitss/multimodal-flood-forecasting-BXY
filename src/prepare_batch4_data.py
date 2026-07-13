from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import ensure_dir, save_json


def run(command: list[str], dry_run: bool) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a longer, larger Batch 4 synthetic dataset.")
    parser.add_argument("--output_root", default="runs/batch4_multihorizon/data")
    parser.add_argument("--num_events", type=int, default=48)
    parser.add_argument("--time_steps", type=int, default=72)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.num_events < 20 or args.time_steps < 48:
        raise ValueError("Batch 4 data should contain at least 20 events and 48 time steps")

    root = Path(args.output_root)
    raw_dir = root / "raw"
    aligned_dir = root / "aligned"
    fused_dir = root / "fused"
    ensure_dir(root)
    python = sys.executable
    run(
        [
            python, "-m", "src.generate_synthetic",
            "--num_events", str(args.num_events),
            "--t", str(args.time_steps),
            "--h", str(args.height),
            "--w", str(args.width),
            "--seed", str(args.seed),
            "--out_dir", str(raw_dir),
        ],
        args.dry_run,
    )
    run(
        [
            python, "-m", "src.align_modalities",
            "--raw_dir", str(raw_dir),
            "--out_dir", str(aligned_dir),
            "--mode", "realtime",
            "--value_decay_mode", "none",
        ],
        args.dry_run,
    )
    run(
        [python, "-m", "src.fuse_dynamic_gate", "--aligned_dir", str(aligned_dir), "--out_dir", str(fused_dir)],
        args.dry_run,
    )
    run(
        [
            python, "-m", "src.data.validation",
            "--raw_dir", str(raw_dir),
            "--aligned_dir", str(aligned_dir),
            "--fused_dir", str(fused_dir),
            "--input_len", "12",
            "--lead_time", "24",
            "--output", str(root / "causality_report.json"),
        ],
        args.dry_run,
    )
    if not args.dry_run:
        save_json(
            {
                "experiment_batch": 4,
                "num_events": args.num_events,
                "time_steps": args.time_steps,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "lead_times": [1, 3, 6, 12, 24],
                "fused_dir": str(fused_dir),
            },
            root / "batch4_data_manifest.json",
        )
    print(f"Batch 4 fused data: {fused_dir}")


if __name__ == "__main__":
    main()
