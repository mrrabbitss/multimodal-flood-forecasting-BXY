from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


EXTERNAL_CHANNEL_NAMES = (
    "depth_history",
    "rain_current",
    "rain_accum_3",
    "rain_accum_6",
    "dem",
    "impervious",
    "drainage_inlet",
    "valid_mask",
)

UKEA_TRAIN_EVENTS = (
    "r100y_p0.1_d3h_1",
    "r100y_p0.6_d3h_1",
    "r200y_p0.3_d3h_1",
    "r200y_p0.7_d3h_1",
    "r300y_p0.4_d3h_1",
    "r300y_p0.9_d3h_1",
    "r500y_p0.2_d3h_1",
    "r500y_p0.6_d3h_1",
)

UKEA_TEST_EVENTS = (
    "r100y_p0.5_d3h_1",
    "r100y_p0.7_d3h_1",
    "r100y_p0.8_d3h_1",
    "r200y_p0.5_d3h_1",
    "r300y_p0.1_d3h_1",
    "r300y_p0.5_d3h_1",
    "r300y_p0.6_d3h_1",
    "r300y_p0.8_d3h_1",
    "r500y_p0.1_d3h_1",
    "r500y_p0.3_d3h_1",
    "r500y_p0.4_d3h_1",
    "r500y_p0.9_d3h_1",
)


@dataclass(frozen=True)
class ExternalEvent:
    dataset: str
    split: str
    location: str
    event_id: str
    flood_path: Path
    rainfall_path: Path
    dem_path: Path
    impervious_path: Path | None
    drainage_path: Path | None
    time_steps: int
    height: int
    width: int
    spatial_factor: int
    temporal_factor: int
    time_step_minutes: int = 5

    def to_dict(self) -> dict:
        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


@dataclass(frozen=True)
class ExternalSample:
    event_index: int
    start: int
    patch_y: int
    patch_x: int


def _resolve_root(root: str | Path, marker: Path) -> Path:
    root = Path(root).resolve()
    candidates = (root, root / root.name, root / "urbanflood24", root / "larno_ukea_8m_5min")
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {marker} below {root}")


def discover_urbanflood24(
    root: str | Path,
    split: str,
    location: str = "location1",
    spatial_factor: int = 4,
    temporal_factor: int = 5,
) -> list[ExternalEvent]:
    if split not in {"train", "test"}:
        raise ValueError("UrbanFlood24 split must be train or test")
    if location not in {"location1", "location2", "location3"}:
        raise ValueError(f"Unknown UrbanFlood24 location: {location}")
    root = _resolve_root(root, Path("train") / "flood")
    flood_root = root / split / "flood" / location
    geo_root = root / split / "geodata" / location
    events: list[ExternalEvent] = []
    for event_dir in sorted(path for path in flood_root.iterdir() if path.is_dir()):
        flood_path = event_dir / "flood.npy"
        rainfall_path = event_dir / "rainfall.npy"
        flood = np.load(flood_path, mmap_mode="r", allow_pickle=False)
        if flood.ndim != 4 or flood.shape[1] != 1:
            raise ValueError(f"Unexpected UrbanFlood24 flood shape: {flood.shape} at {flood_path}")
        events.append(
            ExternalEvent(
                dataset="urbanflood24",
                split=split,
                location=location,
                event_id=event_dir.name,
                flood_path=flood_path,
                rainfall_path=rainfall_path,
                dem_path=geo_root / "absolute_DEM.npy",
                impervious_path=geo_root / "impervious.npy",
                drainage_path=geo_root / "manhole.npy",
                time_steps=int(flood.shape[0] // temporal_factor),
                height=int(flood.shape[-2] // spatial_factor),
                width=int(flood.shape[-1] // spatial_factor),
                spatial_factor=int(spatial_factor),
                temporal_factor=int(temporal_factor),
            )
        )
    if not events:
        raise FileNotFoundError(f"No UrbanFlood24 events found in {flood_root}")
    return events


def discover_larno_ukea(root: str | Path, split: str) -> list[ExternalEvent]:
    if split not in {"train", "test"}:
        raise ValueError("UKEA split must be train or test")
    root = _resolve_root(root, Path("flood") / "ukea_8m_5min")
    flood_root = root / "flood" / "ukea_8m_5min"
    geo_root = root / "geodata" / "ukea_8m_5min"
    selected = UKEA_TRAIN_EVENTS if split == "train" else UKEA_TEST_EVENTS
    events: list[ExternalEvent] = []
    for event_id in selected:
        event_dir = flood_root / event_id
        flood_path = event_dir / "h.npy"
        rainfall_path = event_dir / "rainfall.npy"
        if not flood_path.exists() or not rainfall_path.exists():
            raise FileNotFoundError(f"Missing canonical UKEA event files in {event_dir}")
        flood = np.load(flood_path, mmap_mode="r", allow_pickle=False)
        rainfall = np.load(rainfall_path, mmap_mode="r", allow_pickle=False)
        if flood.ndim != 3 or rainfall.ndim != 3:
            raise ValueError(f"Unexpected UKEA shapes: flood={flood.shape}, rainfall={rainfall.shape}")
        if rainfall.shape[0] < flood.shape[0] * 5:
            raise ValueError(f"UKEA rainfall is too short for 5-minute aggregation: {rainfall.shape}")
        events.append(
            ExternalEvent(
                dataset="larno_ukea",
                split=split,
                location="ukea",
                event_id=event_id,
                flood_path=flood_path,
                rainfall_path=rainfall_path,
                dem_path=geo_root / "dem.npy",
                impervious_path=None,
                drainage_path=None,
                time_steps=int(flood.shape[0]),
                height=int(flood.shape[-2]),
                width=int(flood.shape[-1]),
                spatial_factor=1,
                temporal_factor=5,
            )
        )
    return events


def split_train_validation(
    events: Sequence[ExternalEvent],
    validation_events: int,
    seed: int,
) -> tuple[list[ExternalEvent], list[ExternalEvent]]:
    if validation_events < 1 or validation_events >= len(events):
        raise ValueError("validation_events must leave at least one training event")
    indices = list(range(len(events)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_events])
    train = [event for index, event in enumerate(events) if index not in validation_indices]
    validation = [event for index, event in enumerate(events) if index in validation_indices]
    return train, validation


def aggregate_ukea_rainfall(rainfall: np.ndarray) -> np.ndarray:
    if rainfall.ndim != 3 or rainfall.shape[0] < 180:
        raise ValueError(f"Expected UKEA rainfall [T,H,W] with T >= 180, got {rainfall.shape}")
    active = np.asarray(rainfall[:180], dtype=np.float32)
    return active.reshape(36, 5, active.shape[1], active.shape[2]).sum(axis=1)


def _stable_seed(seed: int, event_id: str) -> int:
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], byteorder="little", signed=False)


def _patch_positions(size: int, patch_size: int, stride: int) -> list[int]:
    padded = int(math.ceil(size / patch_size) * patch_size)
    positions = list(range(0, max(1, padded - patch_size + 1), stride))
    last = max(0, padded - patch_size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _pad_patch(array: np.ndarray, patch_size: int, fill: float = 0.0) -> np.ndarray:
    output = np.full((patch_size, patch_size), fill, dtype=np.float32)
    output[: array.shape[0], : array.shape[1]] = array.astype(np.float32, copy=False)
    return output


class ExternalFloodDataset(Dataset):
    """Streaming 8 m / 5 min physical-depth benchmark without full-data copies."""

    def __init__(
        self,
        events: Sequence[ExternalEvent],
        input_len: int = 12,
        lead_times: Sequence[int] = (1, 3, 6, 12),
        patch_size: int = 64,
        patch_stride: int | None = None,
        max_samples_per_event: int | None = None,
        seed: int = 42,
        depth_scale_m: float = 3.5,
        rain_scale_mm_5min: float = 35.0,
    ) -> None:
        if not events:
            raise ValueError("ExternalFloodDataset requires at least one event")
        self.events = list(events)
        self.input_len = int(input_len)
        self.lead_times = tuple(sorted(int(value) for value in lead_times))
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride or patch_size)
        self.max_samples_per_event = None if not max_samples_per_event else int(max_samples_per_event)
        self.depth_scale_m = float(depth_scale_m)
        self.rain_scale_mm_5min = float(rain_scale_mm_5min)
        if self.input_len < 1 or not self.lead_times or self.lead_times[0] < 1:
            raise ValueError("input_len and lead_times must be positive")
        if self.patch_size < 4 or self.patch_stride < 1:
            raise ValueError("patch_size must be >= 4 and patch_stride must be positive")
        if self.depth_scale_m <= 0 or self.rain_scale_mm_5min <= 0:
            raise ValueError("normalization scales must be positive")

        self.samples: list[ExternalSample] = []
        max_lead = max(self.lead_times)
        for event_index, event in enumerate(self.events):
            max_start = event.time_steps - self.input_len - max_lead
            if max_start < 0:
                raise ValueError(
                    f"Event {event.event_id} has {event.time_steps} steps, too short for "
                    f"input_len={self.input_len}, max_lead={max_lead}"
                )
            event_samples = [
                ExternalSample(event_index, start, patch_y, patch_x)
                for start in range(max_start + 1)
                for patch_y in _patch_positions(event.height, self.patch_size, self.patch_stride)
                for patch_x in _patch_positions(event.width, self.patch_size, self.patch_stride)
            ]
            if self.max_samples_per_event and len(event_samples) > self.max_samples_per_event:
                rng = random.Random(_stable_seed(seed, event.event_id))
                selected = sorted(rng.sample(range(len(event_samples)), self.max_samples_per_event))
                event_samples = [event_samples[index] for index in selected]
            self.samples.extend(event_samples)

        self._array_cache: dict[Path, np.ndarray] = {}
        self._static_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _array(self, path: Path) -> np.ndarray:
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._array_cache[path]

    def _downsample_static(self, path: Path, factor: int) -> np.ndarray:
        array = np.asarray(self._array(path), dtype=np.float32)
        if factor == 1:
            return array
        height = array.shape[0] // factor
        width = array.shape[1] // factor
        cropped = array[: height * factor, : width * factor]
        return cropped.reshape(height, factor, width, factor).mean(axis=(1, 3))

    def _static_arrays(self, event: ExternalEvent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (event.dem_path, event.spatial_factor)
        if key not in self._static_cache:
            dem = self._downsample_static(event.dem_path, event.spatial_factor)
            dem_min = float(np.min(dem))
            dem_max = float(np.max(dem))
            dem = (dem - dem_min) / max(dem_max - dem_min, 1e-6)
            if event.impervious_path is None:
                impervious = np.zeros_like(dem, dtype=np.float32)
            else:
                impervious = np.clip(
                    self._downsample_static(event.impervious_path, event.spatial_factor), 0.0, 1.0
                )
            if event.drainage_path is None:
                drainage = np.zeros_like(dem, dtype=np.float32)
            else:
                drainage = self._downsample_static(event.drainage_path, event.spatial_factor)
                drainage = drainage / max(float(np.max(drainage)), 1e-6)
            self._static_cache[key] = (
                dem.astype(np.float32),
                impervious.astype(np.float32),
                drainage.astype(np.float32),
            )
        return self._static_cache[key]

    def _valid_shape(self, event: ExternalEvent, patch_y: int, patch_x: int) -> tuple[int, int]:
        return (
            max(0, min(self.patch_size, event.height - patch_y)),
            max(0, min(self.patch_size, event.width - patch_x)),
        )

    def _depth_patch(
        self,
        event: ExternalEvent,
        aligned_time: int,
        patch_y: int,
        patch_x: int,
    ) -> np.ndarray:
        valid_h, valid_w = self._valid_shape(event, patch_y, patch_x)
        if valid_h == 0 or valid_w == 0:
            return np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        flood = self._array(event.flood_path)
        if event.dataset == "urbanflood24":
            factor = event.spatial_factor
            raw_time = aligned_time * event.temporal_factor
            raw = np.asarray(
                flood[
                    raw_time,
                    0,
                    patch_y * factor : (patch_y + valid_h) * factor,
                    patch_x * factor : (patch_x + valid_w) * factor,
                ],
                dtype=np.float32,
            )
            patch = raw.reshape(valid_h, factor, valid_w, factor).mean(axis=(1, 3))
        else:
            patch = np.asarray(
                flood[aligned_time, patch_y : patch_y + valid_h, patch_x : patch_x + valid_w],
                dtype=np.float32,
            )
        return _pad_patch(patch, self.patch_size)

    def _rain_patch(
        self,
        event: ExternalEvent,
        aligned_time: int,
        patch_y: int,
        patch_x: int,
    ) -> np.ndarray:
        valid_h, valid_w = self._valid_shape(event, patch_y, patch_x)
        output = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        if aligned_time < 0 or valid_h == 0 or valid_w == 0:
            return output
        rainfall = self._array(event.rainfall_path)
        start = aligned_time * event.temporal_factor
        stop = min(start + event.temporal_factor, rainfall.shape[0])
        if start >= rainfall.shape[0]:
            return output
        if event.dataset == "urbanflood24":
            value = float(np.asarray(rainfall[start:stop], dtype=np.float32).sum())
            output[:valid_h, :valid_w] = value
        else:
            patch = np.asarray(
                rainfall[
                    start:stop,
                    patch_y : patch_y + valid_h,
                    patch_x : patch_x + valid_w,
                ],
                dtype=np.float32,
            ).sum(axis=0)
            output[:valid_h, :valid_w] = patch
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        event = self.events[sample.event_index]
        end = sample.start + self.input_len
        rain_cache: dict[int, np.ndarray] = {}

        def rain_at(time_index: int) -> np.ndarray:
            if time_index not in rain_cache:
                rain_cache[time_index] = self._rain_patch(
                    event, time_index, sample.patch_y, sample.patch_x
                )
            return rain_cache[time_index]

        dem, impervious, drainage = self._static_arrays(event)
        valid_h, valid_w = self._valid_shape(event, sample.patch_y, sample.patch_x)
        dem_patch = _pad_patch(
            dem[sample.patch_y : sample.patch_y + valid_h, sample.patch_x : sample.patch_x + valid_w],
            self.patch_size,
            fill=1.0,
        )
        impervious_patch = _pad_patch(
            impervious[
                sample.patch_y : sample.patch_y + valid_h,
                sample.patch_x : sample.patch_x + valid_w,
            ],
            self.patch_size,
        )
        drainage_patch = _pad_patch(
            drainage[
                sample.patch_y : sample.patch_y + valid_h,
                sample.patch_x : sample.patch_x + valid_w,
            ],
            self.patch_size,
        )
        valid_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        valid_mask[:valid_h, :valid_w] = 1.0

        frames = []
        for time_index in range(sample.start, end):
            current_rain = rain_at(time_index)
            accumulated_3 = sum((rain_at(t) for t in range(time_index - 2, time_index + 1)), start=np.zeros_like(current_rain))
            accumulated_6 = sum((rain_at(t) for t in range(time_index - 5, time_index + 1)), start=np.zeros_like(current_rain))
            depth = self._depth_patch(event, time_index, sample.patch_y, sample.patch_x)
            frame = np.stack(
                [
                    np.clip(depth / self.depth_scale_m, 0.0, 1.0),
                    np.clip(current_rain / self.rain_scale_mm_5min, 0.0, 1.0),
                    np.clip(accumulated_3 / (self.rain_scale_mm_5min * 3.0), 0.0, 1.0),
                    np.clip(accumulated_6 / (self.rain_scale_mm_5min * 6.0), 0.0, 1.0),
                    dem_patch,
                    impervious_patch,
                    drainage_patch,
                    valid_mask,
                ],
                axis=0,
            )
            frames.append(frame)

        targets = np.stack(
            [
                self._depth_patch(event, end - 1 + lead, sample.patch_y, sample.patch_x)
                for lead in self.lead_times
            ],
            axis=0,
        )
        return {
            "x": torch.from_numpy(np.stack(frames, axis=0).astype(np.float32)),
            "target": torch.from_numpy(targets.astype(np.float32)),
            "valid_mask": torch.from_numpy(valid_mask[None, ...]),
            "event_id": event.event_id,
            "dataset": event.dataset,
            "start": sample.start,
            "patch_y": sample.patch_y,
            "patch_x": sample.patch_x,
        }

    def manifest(self) -> dict:
        return {
            "schema_version": "external_physical_v1",
            "channel_names": list(EXTERNAL_CHANNEL_NAMES),
            "input_len": self.input_len,
            "lead_times": list(self.lead_times),
            "time_step_minutes": 5,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "depth_scale_m": self.depth_scale_m,
            "rain_scale_mm_5min": self.rain_scale_mm_5min,
            "num_events": len(self.events),
            "num_samples": len(self.samples),
            "events": [event.to_dict() for event in self.events],
        }
