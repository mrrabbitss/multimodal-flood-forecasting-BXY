from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def list_npz_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npz"))


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)


def gaussian_blob(h: int, w: int, center_y: float, center_x: float, sigma_y: float, sigma_x: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-(((yy - center_y) ** 2) / (2 * sigma_y**2) + ((xx - center_x) ** 2) / (2 * sigma_x**2))).astype(np.float32)


def smooth2d(x: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small dependency-free spatial smoothing using reflected borders."""
    y = x.astype(np.float32).copy()
    if y.shape[-2] < 2 or y.shape[-1] < 2:
        return y
    pad_width = [(0, 0)] * y.ndim
    pad_width[-2] = (1, 1)
    pad_width[-1] = (1, 1)
    for _ in range(passes):
        padded = np.pad(y, pad_width, mode="reflect")
        y = (
            padded[..., 1:-1, 1:-1]
            + padded[..., :-2, 1:-1]
            + padded[..., 2:, 1:-1]
            + padded[..., 1:-1, :-2]
            + padded[..., 1:-1, 2:]
        ) / 5.0
    return y.astype(np.float32)


def to_float32_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            out[k] = v.astype(np.float32)
        else:
            out[k] = v
    return out
