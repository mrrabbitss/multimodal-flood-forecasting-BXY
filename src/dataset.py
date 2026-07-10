from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import list_npz_files


LEGACY_CHANNEL_NAMES: tuple[str, ...] = (
    "meteo",
    "satellite",
    "gis",
    "social",
    "fused_depth",
    "risk_score",
    "miss_sat",
    "miss_soc",
    "dt_sat",
    "dt_soc",
    "n_soc",
    "exposure",
    "drainage_penalty",
)

CHANNEL_NAMES: tuple[str, ...] = (
    "meteo",
    "satellite",
    "gis",
    "social",
    "fused_depth",
    "risk_score",
    "miss_sat",
    "miss_gis",
    "miss_soc",
    "dt_sat",
    "dt_gis",
    "dt_soc",
    "q_sat",
    "q_gis",
    "q_soc",
    "n_soc",
    "soc_observation_mask",
    "exposure",
    "drainage_penalty",
)

CHANNEL_INDEX: dict[str, int] = {name: index for index, name in enumerate(CHANNEL_NAMES)}


def resolve_channel_names(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    if value is None or value == "default":
        return CHANNEL_NAMES
    if value == "legacy":
        return LEGACY_CHANNEL_NAMES
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    else:
        names = tuple(str(name) for name in value)
    if not names:
        raise ValueError("input channel list must not be empty")
    unknown = [name for name in names if name not in CHANNEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown input channels: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("input channel list contains duplicates")
    return names


def channel_names_from_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[str, ...]:
    saved = checkpoint.get("channel_names")
    if saved is not None:
        names = resolve_channel_names(saved)
        expected = int(checkpoint.get("input_channels", len(names)))
        if len(names) != expected:
            raise ValueError(f"Checkpoint channel schema has {len(names)} names but input_channels={expected}")
        return names
    count = int(checkpoint["input_channels"])
    if count == len(LEGACY_CHANNEL_NAMES):
        return LEGACY_CHANNEL_NAMES
    if count == len(CHANNEL_NAMES):
        return CHANNEL_NAMES
    raise ValueError(
        f"Checkpoint has {count} channels but no channel_names metadata; "
        f"only legacy ({len(LEGACY_CHANNEL_NAMES)}) and current ({len(CHANNEL_NAMES)}) schemas can be inferred"
    )


def channel_names_for_data(fused_dir: str | Path) -> tuple[str, ...]:
    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {fused_dir}")
    with np.load(files[0]) as data:
        return CHANNEL_NAMES if "soc_observation_mask" in data.files else LEGACY_CHANNEL_NAMES


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
        channel_names: str | Sequence[str] | None = None,
    ) -> None:
        self.fused_dir = Path(fused_dir)
        self.input_len = int(input_len)
        self.lead_time = int(lead_time)
        self.target = target
        self.channel_names = resolve_channel_names(channel_names)
        self.channel_index = {name: index for index, name in enumerate(self.channel_names)}
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

        dynamic = {
            "meteo": meteo,
            "satellite": sat,
            "gis": gis,
            "social": soc,
            "fused_depth": fused,
            "risk_score": risk,
        }
        scalar_fields = {
            "miss_sat": ("miss_sat", 1.0),
            "miss_gis": ("miss_gis", 1.0),
            "miss_soc": ("miss_soc", 1.0),
            "dt_sat": ("dt_sat", 30.0),
            "dt_gis": ("dt_gis", 72.0),
            "dt_soc": ("dt_soc", 10.0),
            "q_sat": ("q_sat", 1.0),
            "q_gis": ("q_gis", 1.0),
            "q_soc": ("q_soc", 1.0),
            "n_soc": ("n_soc", 30.0),
        }
        for channel_name, (field_name, scale) in scalar_fields.items():
            if channel_name not in self.channel_names:
                continue
            if field_name not in z:
                raise KeyError(f"Channel {channel_name!r} requires missing field {field_name!r}")
            dynamic[channel_name] = np.clip(self._scalar_to_map(z[field_name], h, w, scale), 0, 1)

        if "soc_observation_mask" in self.channel_names:
            if "soc_observation_mask" not in z:
                raise KeyError(
                    "Channel 'soc_observation_mask' is unavailable. Regenerate aligned/fused data "
                    "or use the legacy channel schema for historical artifacts."
                )
            dynamic["soc_observation_mask"] = z["soc_observation_mask"].astype(np.float32)

        # Static maps are repeated over time only when selected.
        exposure = np.repeat(z["exposure"][None, ...].astype(np.float32), meteo.shape[0], axis=0)
        drainage_penalty = np.repeat((1.0 - z["drainage_capacity"])[None, ...].astype(np.float32), meteo.shape[0], axis=0)
        dynamic["exposure"] = exposure
        dynamic["drainage_penalty"] = drainage_penalty

        channels = np.stack([dynamic[name] for name in self.channel_names], axis=1)  # [T,C,H,W]
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


def infer_num_channels(fused_dir: str | Path, channel_names: str | Sequence[str] | None = None) -> int:
    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {fused_dir}")
    ds = FloodSequenceDataset(fused_dir, [0], input_len=2, lead_time=1, channel_names=channel_names)
    x, _ = ds[0]
    return int(x.shape[1])
