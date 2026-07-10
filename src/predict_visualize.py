from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.schemas import depth_scale_from_checkpoint
from .dataset import FloodSequenceDataset, channel_names_from_checkpoint
from .model import ConvLSTMForecastNet
from .utils import ensure_dir, list_npz_files, set_seed


def plot_triplet(pred: np.ndarray, target: np.ndarray, current: np.ndarray, out_path: Path, title: str) -> None:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    current = current.astype(np.float32)
    err = np.abs(pred - target)
    vmax = max(float(pred.max()), float(target.max()), float(current.max()), 0.25)
    plt.figure(figsize=(12, 3.5))
    for i, (img, name) in enumerate([(current, "Current fused"), (target, "Future target"), (pred, "Prediction"), (err, "Abs error")]):
        ax = plt.subplot(1, 4, i + 1)
        im = ax.imshow(
            img,
            vmin=0,
            vmax=vmax if name != "Abs error" else max(float(err.max()), 1e-4),
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--crop_border", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    ckpt = torch.load(args.checkpoint, map_location=device)
    input_len = int(ckpt["input_len"])
    lead_time = int(ckpt["lead_time"])
    depth_scale = depth_scale_from_checkpoint(ckpt)
    channel_names = channel_names_from_checkpoint(ckpt)
    split_seed = ckpt.get("split_seed", args.seed)
    shuffle_split = bool(ckpt.get("shuffle_split", False))

    files = [p for p in list_npz_files(args.fused_dir) if p.name.startswith("event_")]
    _, _, test_idx = FloodSequenceDataset.split_indices(len(files), seed=split_seed, shuffle=shuffle_split)
    ds = FloodSequenceDataset(
        args.fused_dir,
        test_idx,
        input_len=input_len,
        lead_time=lead_time,
        channel_names=channel_names,
    )
    sample_idx = min(max(args.sample_idx, 0), len(ds) - 1)
    x, y = ds[sample_idx]

    model = ConvLSTMForecastNet(
        input_channels=ckpt["input_channels"],
        hidden_channels=ckpt["hidden_channels"],
        num_layers=int(ckpt.get("num_layers", 1)),
        dropout=float(ckpt.get("dropout", 0.0)),
        output_max=depth_scale.max_value,
        residual_scale=float(ckpt.get("residual_scale", 0.35)),
        use_residual=bool(ckpt.get("use_residual", False)),
        fused_channel=int(ckpt.get("fused_channel", channel_names.index("fused_depth") if "fused_depth" in channel_names else -1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(x[None].to(device)).cpu().numpy()[0, 0]

    if "fused_depth" not in channel_names:
        raise ValueError("Prediction visualization requires the fused_depth input channel")
    current_fused = x[-1, channel_names.index("fused_depth")].numpy()
    target = y[0].numpy()
    crop = max(0, min(int(args.crop_border), (min(pred.shape) - 1) // 2))
    if crop > 0:
        view = np.s_[crop:-crop, crop:-crop]
        pred = pred[view]
        current_fused = current_fused[view]
        target = target[view]
    fig_dir = ensure_dir(Path(args.output_dir) / "figures")
    out_path = fig_dir / "prediction_triplet.png"
    plot_triplet(
        pred,
        target,
        current_fused,
        out_path,
        title=f"Lead time = {lead_time} min | depth unit = {depth_scale.unit}",
    )
    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
