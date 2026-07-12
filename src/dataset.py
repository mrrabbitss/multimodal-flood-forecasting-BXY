from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.schemas import CHANNEL_REGISTRY_VERSION, DATA_SCHEMA_VERSION
from .data.transforms import RAIN_FEATURE_NAMES, RAIN_FEATURE_VERSION, rain_features_from_mapping
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

BATCH1_CHANNEL_NAMES: tuple[str, ...] = (
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

RAIN_INPUT_CHANNEL_NAMES: tuple[str, ...] = (
    "rain_current",
    "rain_accum_3",
    "rain_accum_6",
    "rain_accum_12",
)

CHANNEL_NAMES: tuple[str, ...] = (
    *BATCH1_CHANNEL_NAMES[:6],
    *RAIN_INPUT_CHANNEL_NAMES,
    *BATCH1_CHANNEL_NAMES[6:],
)

FULL_RAIN_CHANNEL_NAMES: tuple[str, ...] = (
    *CHANNEL_NAMES[:10],
    "rain_max_recent_6",
    "rain_trend_3",
    *CHANNEL_NAMES[10:],
)

LEGACY_RAIN_CURRENT_CHANNEL_NAMES: tuple[str, ...] = (
    *LEGACY_CHANNEL_NAMES[:6],
    "rain_current",
    *LEGACY_CHANNEL_NAMES[6:],
)

LEGACY_RAIN_ACCUM_CHANNEL_NAMES: tuple[str, ...] = (
    *LEGACY_CHANNEL_NAMES[:6],
    *RAIN_INPUT_CHANNEL_NAMES,
    *LEGACY_CHANNEL_NAMES[6:],
)

ALL_CHANNEL_NAMES: tuple[str, ...] = tuple(dict.fromkeys((*FULL_RAIN_CHANNEL_NAMES, *LEGACY_CHANNEL_NAMES)))

CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "default": CHANNEL_NAMES,
    "full": CHANNEL_NAMES,
    "full_rain": FULL_RAIN_CHANNEL_NAMES,
    "batch1": BATCH1_CHANNEL_NAMES,
    "legacy": LEGACY_CHANNEL_NAMES,
    "legacy_rain_current": LEGACY_RAIN_CURRENT_CHANNEL_NAMES,
    "legacy_rain_accum": LEGACY_RAIN_ACCUM_CHANNEL_NAMES,
    "rain_only": RAIN_INPUT_CHANNEL_NAMES,
    "meteo_only": ("meteo",),
    "fused_only": ("fused_depth",),
}

CHANNEL_INDEX: dict[str, int] = {name: index for index, name in enumerate(CHANNEL_NAMES)}

DIRECT_CHANNEL_FIELDS: dict[str, str] = {
    "meteo": "meteo_depth",
    "satellite": "sat_base",
    "gis": "gis_risk",
    "social": "soc_depth",
    "fused_depth": "fused_depth",
    "risk_score": "risk_score",
    "soc_observation_mask": "soc_observation_mask",
}

SCALAR_CHANNEL_FIELDS: dict[str, tuple[str, float]] = {
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

RAIN_CHANNEL_SCALES: dict[str, float] = {
    "rain_current": 1.0,
    "rain_accum_3": 3.0,
    "rain_accum_6": 6.0,
    "rain_accum_12": 12.0,
    "rain_max_recent_6": 1.0,
    "rain_trend_3": 1.0,
}


def resolve_channel_names(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    if value is None:
        return CHANNEL_NAMES
    if isinstance(value, str):
        if value in CHANNEL_SETS:
            return CHANNEL_SETS[value]
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    else:
        names = tuple(str(name) for name in value)
    if not names:
        raise ValueError("input channel list must not be empty")
    unknown = [name for name in names if name not in ALL_CHANNEL_NAMES]
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
    if count == len(BATCH1_CHANNEL_NAMES):
        return BATCH1_CHANNEL_NAMES
    if count == len(CHANNEL_NAMES):
        return CHANNEL_NAMES
    raise ValueError(
        f"Checkpoint has {count} channels but no channel_names metadata; "
        f"only legacy ({len(LEGACY_CHANNEL_NAMES)}), Batch 1 ({len(BATCH1_CHANNEL_NAMES)}), "
        f"and current ({len(CHANNEL_NAMES)}) schemas can be inferred"
    )


def channel_names_for_data(fused_dir: str | Path) -> tuple[str, ...]:
    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {fused_dir}")
    with np.load(files[0]) as data:
        return CHANNEL_NAMES if "soc_observation_mask" in data.files else LEGACY_CHANNEL_NAMES


def _scalar_value(data: Mapping[str, Any], key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = data[key]
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def validate_channel_availability(data: Mapping[str, Any], channel_names: Sequence[str]) -> None:
    missing: list[str] = []
    for channel_name in channel_names:
        if channel_name in RAIN_FEATURE_NAMES:
            if "rain" not in data and "rain_current" not in data:
                missing.append(f"{channel_name} requires rain or rain_current")
        elif channel_name in DIRECT_CHANNEL_FIELDS:
            field = DIRECT_CHANNEL_FIELDS[channel_name]
            if field not in data:
                missing.append(f"{channel_name} requires {field}")
        elif channel_name in SCALAR_CHANNEL_FIELDS:
            field = SCALAR_CHANNEL_FIELDS[channel_name][0]
            if field not in data:
                missing.append(f"{channel_name} requires {field}")
        elif channel_name == "exposure" and "exposure" not in data:
            missing.append("exposure requires exposure")
        elif channel_name == "drainage_penalty" and "drainage_capacity" not in data:
            missing.append("drainage_penalty requires drainage_capacity")
    if missing:
        raise KeyError("Selected input channels are unavailable: " + "; ".join(missing))


def inspect_dataset_schema(fused_dir: str | Path, channel_names: str | Sequence[str] | None = None) -> dict[str, Any]:
    names = resolve_channel_names(channel_names)
    files = [p for p in list_npz_files(fused_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {fused_dir}")
    with np.load(files[0]) as artifact:
        data = {key: artifact[key] for key in artifact.files}
    validate_channel_availability(data, names)
    target = np.asarray(data["gt_depth"])
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "channel_registry_version": CHANNEL_REGISTRY_VERSION,
        "source_artifact_schema_version": int(_scalar_value(data, "data_schema_version", 1)),
        "rain_feature_version": str(_scalar_value(data, "rain_feature_version", RAIN_FEATURE_VERSION)),
        "rain_features_materialized": all(name in data for name in RAIN_FEATURE_NAMES),
        "channel_names": list(names),
        "input_channels": len(names),
        "target": "gt_depth",
        "time_steps": int(target.shape[0]),
        "height": int(target.shape[-2]),
        "width": int(target.shape[-1]),
    }


def validate_checkpoint_data_schema(checkpoint: Mapping[str, Any], fused_dir: str | Path) -> dict[str, Any]:
    names = channel_names_from_checkpoint(checkpoint)
    current = inspect_dataset_schema(fused_dir, names)
    saved = checkpoint.get("data_schema")
    if not isinstance(saved, Mapping):
        current["checkpoint_schema_compatibility"] = "legacy_inferred"
        return current
    saved_names = tuple(str(name) for name in saved.get("channel_names", []))
    if saved_names and saved_names != names:
        raise ValueError(f"Checkpoint data_schema channel order differs from channel_names: {saved_names} != {names}")
    saved_registry = saved.get("channel_registry_version")
    if saved_registry and str(saved_registry) != CHANNEL_REGISTRY_VERSION:
        raise ValueError(
            f"Checkpoint channel registry {saved_registry!r} is incompatible with {CHANNEL_REGISTRY_VERSION!r}"
        )
    saved_rain_version = saved.get("rain_feature_version")
    if any(name in RAIN_FEATURE_NAMES for name in names) and saved_rain_version:
        if str(saved_rain_version) != current["rain_feature_version"]:
            raise ValueError(
                f"Checkpoint rain feature version {saved_rain_version!r} differs from "
                f"dataset version {current['rain_feature_version']!r}"
            )
    current["checkpoint_schema_compatibility"] = "validated"
    return current


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
        with np.load(self.files[0]) as first_artifact:
            validate_channel_availability(first_artifact, self.channel_names)

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
        target = z[self.target]
        t, h, w = target.shape
        dynamic: dict[str, np.ndarray] = {}

        for channel_name in self.channel_names:
            if channel_name in DIRECT_CHANNEL_FIELDS:
                field_name = DIRECT_CHANNEL_FIELDS[channel_name]
                dynamic[channel_name] = z[field_name].astype(np.float32)
            elif channel_name in SCALAR_CHANNEL_FIELDS:
                field_name, scale = SCALAR_CHANNEL_FIELDS[channel_name]
                dynamic[channel_name] = np.clip(self._scalar_to_map(z[field_name], h, w, scale), 0, 1)

        selected_rain = [name for name in self.channel_names if name in RAIN_FEATURE_NAMES]
        if selected_rain:
            rain_features = rain_features_from_mapping(z)
            for channel_name in selected_rain:
                values = rain_features[channel_name] / RAIN_CHANNEL_SCALES[channel_name]
                if channel_name == "rain_trend_3":
                    values = np.clip(values, -1.0, 1.0)
                else:
                    values = np.clip(values, 0.0, 1.0)
                dynamic[channel_name] = self._scalar_to_map(values, h, w)

        if "exposure" in self.channel_names:
            dynamic["exposure"] = np.repeat(z["exposure"][None, ...].astype(np.float32), t, axis=0)
        if "drainage_penalty" in self.channel_names:
            dynamic["drainage_penalty"] = np.repeat(
                (1.0 - z["drainage_capacity"])[None, ...].astype(np.float32), t, axis=0
            )

        for name in self.channel_names:
            if name not in dynamic:
                raise KeyError(f"No channel builder is registered for {name!r}")
            if dynamic[name].shape != (t, h, w):
                raise ValueError(f"Channel {name!r} has shape {dynamic[name].shape}, expected {(t, h, w)}")

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
