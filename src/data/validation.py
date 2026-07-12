from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .transforms import RAIN_FEATURE_NAMES, derive_rain_features
from ..utils import list_npz_files, save_json


@dataclass(frozen=True)
class CausalityViolation:
    event: str
    anchor: int | None
    field: str
    observed_timestamp: int | None
    message: str


def _mode_value(data: np.lib.npyio.NpzFile) -> str | None:
    if "mode" not in data.files:
        return None
    return str(data["mode"].item())


def _check_selected_times(
    event: str,
    anchors: np.ndarray,
    values: np.ndarray,
    field: str,
) -> list[CausalityViolation]:
    violations = []
    for anchor, observed in zip(anchors.astype(int), values.astype(int)):
        if observed >= 0 and observed > anchor:
            violations.append(
                CausalityViolation(
                    event=event,
                    anchor=int(anchor),
                    field=field,
                    observed_timestamp=int(observed),
                    message="realtime input selected a future observation",
                )
            )
    return violations


def validate_realtime_causality(
    raw_dir: str | Path | None = None,
    aligned_dir: str | Path | None = None,
    fused_dir: str | Path | None = None,
    input_len: int = 12,
    lead_time: int = 6,
) -> dict:
    if input_len < 1:
        raise ValueError("input_len must be >= 1")
    if lead_time < 1:
        raise ValueError("lead_time must be >= 1 so the target follows the input window")

    violations: list[CausalityViolation] = []
    checked_events: set[str] = set()
    aligned_files = list_npz_files(aligned_dir) if aligned_dir is not None else []
    fused_files = list_npz_files(fused_dir) if fused_dir is not None else []

    for path in [p for p in aligned_files if p.name.startswith("event_")]:
        with np.load(path) as data:
            checked_events.add(path.name)
            mode = _mode_value(data)
            if mode != "realtime":
                violations.append(
                    CausalityViolation(path.name, None, "mode", None, f"expected realtime mode, found {mode!r}")
                )
            anchors = data["anchors"].astype(np.int32)
            for field in ("sat_observation_time", "gis_observation_time", "soc_latest_observation_time"):
                if field in data.files:
                    violations.extend(_check_selected_times(path.name, anchors, data[field], field))
                else:
                    violations.append(
                        CausalityViolation(
                            path.name,
                            None,
                            field,
                            None,
                            "strict causality cannot be audited because selected timestamp metadata is missing",
                        )
                    )

            if "rain" in data.files:
                expected_rain = derive_rain_features(data["rain"])
                for field in RAIN_FEATURE_NAMES:
                    if field in data.files and not np.allclose(data[field], expected_rain[field], atol=1e-6):
                        violations.append(
                            CausalityViolation(
                                path.name,
                                None,
                                field,
                                None,
                                "materialized rain feature does not match the causal rolling definition",
                            )
                        )

            input_end_indices = range(input_len - 1, len(anchors) - lead_time)
            for input_end_index in input_end_indices:
                input_end = int(anchors[input_end_index])
                target_time = int(anchors[input_end_index + lead_time])
                if target_time <= input_end:
                    violations.append(
                        CausalityViolation(
                            path.name,
                            int(input_end),
                            "gt_depth",
                            int(target_time),
                            "target timestamp must follow the input window",
                        )
                    )

    for path in [p for p in fused_files if p.name.startswith("event_")]:
        with np.load(path) as data:
            checked_events.add(path.name)
            mode = _mode_value(data)
            if mode != "realtime":
                violations.append(
                    CausalityViolation(path.name, None, "mode", None, f"fused artifact is not auditable realtime data: {mode!r}")
                )

    if raw_dir is not None:
        for path in [p for p in list_npz_files(raw_dir) if p.name.startswith("event_")]:
            checked_events.add(path.name)
            with np.load(path) as data:
                for field in ("sat_times", "gis_times", "point_t"):
                    if field in data.files and not np.all(np.isfinite(data[field])):
                        violations.append(
                            CausalityViolation(path.name, None, field, None, "source timestamps contain NaN or Inf")
                        )

    return {
        "mode": "realtime",
        "valid": not violations,
        "checked_event_count": len(checked_events),
        "input_len": int(input_len),
        "lead_time": int(lead_time),
        "violations": [asdict(v) for v in violations],
    }


def _format_violations(violations: Iterable[dict]) -> str:
    lines = []
    for item in violations:
        lines.append(
            f"{item['event']} anchor={item['anchor']} field={item['field']} "
            f"timestamp={item['observed_timestamp']}: {item['message']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate realtime multimodal causality.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--aligned_dir", type=str, default="data/aligned")
    parser.add_argument("--fused_dir", type=str, default="data/fused")
    parser.add_argument("--mode", type=str, choices=["realtime"], default="realtime")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = validate_realtime_causality(
        raw_dir=args.raw_dir,
        aligned_dir=args.aligned_dir,
        fused_dir=args.fused_dir,
        input_len=args.input_len,
        lead_time=args.lead_time,
    )
    if args.output:
        save_json(report, args.output)
    if not report["valid"]:
        print(_format_violations(report["violations"]))
        raise SystemExit(1)
    print(f"Realtime causality valid for {report['checked_event_count']} events.")


if __name__ == "__main__":
    main()
