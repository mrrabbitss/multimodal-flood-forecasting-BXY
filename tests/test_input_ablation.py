from __future__ import annotations

from src.run_input_ablation import build_per_event_differences, parse_variants


def test_default_ablation_variants_parse_in_order() -> None:
    variants = parse_variants("A=legacy;B=legacy_rain_current;C=legacy_rain_accum")
    assert [name for name, _ in variants] == ["A", "B", "C"]


def test_per_event_differences_use_first_variant_as_baseline() -> None:
    per_event = {
        "A": [{"event_id": "event_1", "mae": 0.10, "csi": 0.80}],
        "B": [{"event_id": "event_1", "mae": 0.08, "csi": 0.82}],
    }
    rows = build_per_event_differences(per_event, "A")
    result = next(row for row in rows if row["variant"] == "B")
    assert abs(result["mae_delta_vs_A"] + 0.02) < 1e-9
    assert abs(result["csi_delta_vs_A"] - 0.02) < 1e-9
    assert result["csi_outcome_vs_A"] == "win"
