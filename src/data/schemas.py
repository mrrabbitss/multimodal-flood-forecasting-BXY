from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


DepthScaleMode = Literal["normalized", "physical"]
DATA_SCHEMA_VERSION = 2
CHANNEL_REGISTRY_VERSION = "rain_schema_v2"


@dataclass(frozen=True)
class DepthScale:
    mode: DepthScaleMode
    min_value: float
    max_value: float
    unit: str

    def __post_init__(self) -> None:
        if self.mode not in {"normalized", "physical"}:
            raise ValueError(f"Unsupported depth scale mode: {self.mode}")
        if self.max_value <= self.min_value:
            raise ValueError("depth max_value must be greater than min_value")
        if not self.unit.strip():
            raise ValueError("depth unit must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DepthScale":
        return cls(
            mode=str(value["mode"]),
            min_value=float(value["min_value"]),
            max_value=float(value["max_value"]),
            unit=str(value["unit"]),
        )


@dataclass(frozen=True)
class RiskThreshold:
    value: float
    unit: str
    meaning: str

    def __post_init__(self) -> None:
        if not self.unit.strip():
            raise ValueError("risk threshold unit must not be empty")
        if not self.meaning.strip():
            raise ValueError("risk threshold meaning must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_DEPTH_SCALE = DepthScale(
    mode="normalized",
    min_value=0.0,
    max_value=1.2,
    unit="normalized_depth",
)

LEGACY_DEPTH_SCALE = DepthScale(
    mode="normalized",
    min_value=0.0,
    max_value=1.0,
    unit="normalized_depth",
)

RISK_THRESHOLD_MEANING = "binary high-risk threshold for the synthetic normalized benchmark"


def make_depth_scale(mode: str = "normalized", depth_max: float = 1.2) -> DepthScale:
    if mode == "normalized":
        return DepthScale(mode="normalized", min_value=0.0, max_value=float(depth_max), unit="normalized_depth")
    raise NotImplementedError("physical depth mode is reserved for a later data-generator phase")


def make_risk_threshold(value: float, depth_scale: DepthScale) -> RiskThreshold:
    meaning = RISK_THRESHOLD_MEANING if depth_scale.mode == "normalized" else "binary high-risk threshold"
    return RiskThreshold(value=float(value), unit=depth_scale.unit, meaning=meaning)


def depth_scale_from_checkpoint(checkpoint: Mapping[str, Any]) -> DepthScale:
    value = checkpoint.get("depth_scale")
    if isinstance(value, Mapping):
        return DepthScale.from_dict(value)
    if "output_max" in checkpoint:
        return DepthScale(
            mode="normalized",
            min_value=0.0,
            max_value=float(checkpoint["output_max"]),
            unit="normalized_depth",
        )
    return LEGACY_DEPTH_SCALE


def depth_scale_from_arrays(arrays: Mapping[str, Any]) -> DepthScale:
    if all(key in arrays for key in ("depth_scale_mode", "depth_min", "depth_max", "depth_unit")):
        return DepthScale(
            mode=str(arrays["depth_scale_mode"].item()),
            min_value=float(arrays["depth_min"].item()),
            max_value=float(arrays["depth_max"].item()),
            unit=str(arrays["depth_unit"].item()),
        )
    return DEFAULT_DEPTH_SCALE
