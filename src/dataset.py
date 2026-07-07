from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import list_npz_files


@dataclass
class SampleIndex:
    file_path: Path
    start: int
    event_idx: int


class FloodSequenceDataset(Dataset):
    """Sliding-window dataset for Conv-LSTM.

    X shape returned by __getitem__: [K, C, H, W]
    Y shape returned by __getitem__: [1, H, W]
    """

    def __init__(
        self,
        fused_dir: str | Path,
        event_indices: Sequence[int],
        input_len: int = 12,
        lead_time: int = 6,
        target: str = "gt_depth",
    ) -> None:
        self.fused_dir = Path(fused_dir)
        self.input_len = int(input_len)
        self.lead_time = int(lead_time)
        self.target = target
        all_files = [p for p in list_npz_files(self.fused_dir) if p.name.startswith("event_")]
        if not all_files:
            raise FileNotFoundError(f"No event_*.npz found in {self.fused_dir}")
        self.files = [all_files[i] for i in event_indices if 0 <= i < len(all_files)]
        if not self.files:
            raise ValueError("No files selected for this split.")

        self.samples: List[SampleIndex] = []
        self._cache: dict[Path, dict] = {}
        for local_idx, p in enumerate(self.files):
            with np.load(p) as z:
                t = z["gt_depth"].shape[0]
            max_start = t - self.input_len - self.lead_time
            for s in range(max_start + 1):
                self.samples.append(SampleIndex(p, s, local_idx))
        if not self.samples:
            raise ValueError("No training samples generated. Increase T or reduce input_len/lead_time.")

    @staticmethod
    def split_indices(
        num_events: int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> Tuple[list[int], list[int], list[int]]:
        indices = list(range(num_events))
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = [int(i) for i in rng.permutation(indices)]
        n_train = max(1, int(num_events * train_ratio))
        n_val = max(1, int(num_events * val_ratio)) if num_events >= 3 else 0
        train = indices[:n_train]
        val = indices[n_train : n_train + n_val]
        test = indices[n_train + n_val :]
        if not val:
            val = train[-1:]
        if not test:
            test = val[-1:]
        return train, val, test

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> dict:
        if path not in self._cache:
            z = np.load(path)
            self._cache[path] = {k: z[k] for k in z.files}
        return self._cache[path]

    @staticmethod
    def _scalar_to_map(arr: np.ndarray, h: int, w: int, scale: float = 1.0) -> np.ndarray:
        x = arr.astype(np.float32) / scale
        return np.repeat(x[:, None, None], h, axis=1).repeat(w, axis=2)

    def _build_channels(self, z: dict) -> np.ndarray:
        meteo = z["meteo_depth"].astype(np.float32)
        sat = z["sat_base"].astype(np.float32)
        gis = z["gis_risk"].astype(np.float32)
        soc = z["soc_depth"].astype(np.float32)
        fused = z["fused_depth"].astype(np.float32)
        risk = z["risk_score"].astype(np.float32)
        h, w = meteo.shape[1:]

        miss_sat = self._scalar_to_map(z["miss_sat"], h, w, 1.0)
        miss_soc = self._scalar_to_map(z["miss_soc"], h, w, 1.0)
        dt_sat = np.clip(self._scalar_to_map(z["dt_sat"], h, w, 30.0), 0, 1)
        dt_soc = np.clip(self._scalar_to_map(z["dt_soc"], h, w, 10.0), 0, 1)
        n_soc = np.clip(self._scalar_to_map(z["n_soc"], h, w, 30.0), 0, 1)

        # Static maps are repeated over time.
        exposure = np.repeat(z["exposure"][None, ...].astype(np.float32), meteo.shape[0], axis=0)
        drainage_penalty = np.repeat((1.0 - z["drainage_capacity"])[None, ...].astype(np.float32), meteo.shape[0], axis=0)

        channels = np.stack(
            [
                meteo,
                sat,
                gis,
                soc,
                fused,
                risk,
                miss_sat,
                miss_soc,
                dt_sat,
                dt_soc,
                n_soc,
                exposure,
                drainage_penalty,
            ],
            axis=1,
        )  # [T,C,H,W]
        return channels.astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        z = self._load(item.file_path)
        x_all = self._build_channels(z)
        start = item.start
        end = start + self.input_len
        target_t = end - 1 + self.lead_time
        x = x_all[start:end]
        y = z[self.target][target_t].astype(np.float32)[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y)


def infer_num_channels(fused_dir: str | Path) -> int:
    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {fused_dir}")
    ds = FloodSequenceDataset(fused_dir, [0], input_len=2, lead_time=1)
    x, _ = ds[0]
    return int(x.shape[1])
