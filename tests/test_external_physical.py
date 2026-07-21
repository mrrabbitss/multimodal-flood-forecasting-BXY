from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.external_data import ExternalEvent, ExternalFloodDataset, aggregate_ukea_rainfall
from src.external_models import (
    EXTERNAL_MODEL_TYPES,
    build_external_model,
    build_external_model_from_checkpoint,
)
from src.run_external_benchmark import parse_model_lrs, parse_models, parse_seeds
from src.summarize_external import (
    compare_models_by_event,
    flatten_external_runs,
    summarize_datasets,
    summarize_models,
    validate_external_runs,
)
from src.train_external import PhysicalMetricAccumulator, evaluate, masked_physical_loss


def _urban_event(tmp_path: Path) -> ExternalEvent:
    event_dir = tmp_path / "event"
    geo_dir = tmp_path / "geo"
    event_dir.mkdir()
    geo_dir.mkdir()
    time = np.arange(40, dtype=np.float32)[:, None, None, None]
    flood = np.broadcast_to(time / 20.0, (40, 1, 16, 16)).copy()
    np.save(event_dir / "flood.npy", flood)
    np.save(event_dir / "rainfall.npy", np.ones(20, dtype=np.float32))
    np.save(geo_dir / "absolute_DEM.npy", np.arange(256, dtype=np.float32).reshape(16, 16))
    np.save(geo_dir / "impervious.npy", np.full((16, 16), 0.7, dtype=np.float32))
    np.save(geo_dir / "manhole.npy", np.eye(16, dtype=np.float32))
    return ExternalEvent(
        dataset="urbanflood24",
        split="train",
        location="location1",
        event_id="fixture",
        flood_path=event_dir / "flood.npy",
        rainfall_path=event_dir / "rainfall.npy",
        dem_path=geo_dir / "absolute_DEM.npy",
        impervious_path=geo_dir / "impervious.npy",
        drainage_path=geo_dir / "manhole.npy",
        time_steps=8,
        height=4,
        width=4,
        spatial_factor=4,
        temporal_factor=5,
    )


def test_ukea_rainfall_aggregation_preserves_five_minute_total() -> None:
    rainfall = np.ones((360, 3, 4), dtype=np.float32)
    aggregated = aggregate_ukea_rainfall(rainfall)
    assert aggregated.shape == (36, 3, 4)
    assert np.allclose(aggregated, 5.0)


def test_external_dataset_aligns_depth_rain_and_padding(tmp_path: Path) -> None:
    dataset = ExternalFloodDataset(
        [_urban_event(tmp_path)],
        input_len=3,
        lead_times=(1, 2),
        patch_size=8,
        patch_stride=8,
        depth_scale_m=3.5,
        rain_scale_mm_5min=35.0,
    )
    sample = dataset[0]
    assert sample["x"].shape == (3, 8, 8, 8)
    assert sample["target"].shape == (2, 8, 8)
    assert sample["valid_mask"].shape == (1, 8, 8)
    assert sample["valid_mask"].sum().item() == 16
    assert torch.allclose(sample["x"][0, 1, :4, :4], torch.full((4, 4), 5.0 / 35.0))
    assert torch.allclose(sample["target"][0, :4, :4], torch.full((4, 4), 0.75))
    assert torch.all(sample["target"][:, 4:, :] == 0)
    assert dataset.manifest()["sampling_seed"] == 42


def test_urban_short_rainfall_is_zero_padded_after_event(tmp_path: Path) -> None:
    dataset = ExternalFloodDataset(
        [_urban_event(tmp_path)],
        input_len=2,
        lead_times=(1,),
        patch_size=4,
        patch_stride=4,
        depth_scale_m=3.5,
        rain_scale_mm_5min=35.0,
    )
    late_index = next(
        index for index, sample in enumerate(dataset.samples) if sample.start == 5
    )
    late = dataset[late_index]
    assert torch.count_nonzero(late["x"][:, 1]) == 0


@pytest.mark.parametrize(
    "model_type",
    EXTERNAL_MODEL_TYPES,
)
def test_external_models_predict_all_horizons(model_type: str) -> None:
    model = build_external_model(
        model_type,
        input_channels=8,
        output_channels=4,
        hidden_channels=8,
        num_layers=1,
        transformer_heads=2,
        output_max=3.5,
        max_input_len=4,
        fno_modes=4,
    )
    x = torch.rand(2, 4, 8, 16, 16)
    output = model(x)
    assert output.shape == (2, 4, 16, 16)
    assert torch.all(output >= 0)
    assert torch.all(output <= 3.5)
    expected = (x[:, -1, 0:1] * 3.5).repeat(1, 4, 1, 1)
    assert torch.allclose(output, expected)


def test_masked_loss_ignores_padded_pixels() -> None:
    target = torch.zeros(1, 2, 4, 4)
    prediction = torch.zeros_like(target)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :2] = 1
    reference, _ = masked_physical_loss(prediction, target, mask)
    prediction[..., 2:, :] = 3.5
    prediction[..., :, 2:] = 3.5
    changed, _ = masked_physical_loss(prediction, target, mask)
    assert torch.allclose(reference, changed)


def test_physical_metrics_include_wet_dry_depth_bins_and_boundary() -> None:
    target = torch.zeros(1, 1, 6, 6)
    target[..., 2:4, 2:4] = 0.2
    prediction = target.clone()
    mask = torch.ones(1, 1, 6, 6)
    accumulator = PhysicalMetricAccumulator(1, (0.1,), 0.1)
    accumulator.update(prediction, target, mask)
    row = accumulator.finalize((1,))[0]
    assert row["wet_rmse_m"] == pytest.approx(0.0)
    assert row["dry_prediction_mean_m"] == pytest.approx(0.0)
    assert row["boundary_f1"] == pytest.approx(1.0)
    assert row["depth_bin_metrics"]["0.10-0.30"]["pixels"] == 4
    assert row["f1"] == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression coverage")
def test_fno_amp_backward_uses_real_optimizer_parameters() -> None:
    model = build_external_model(
        "fno2d_history",
        input_channels=8,
        output_channels=2,
        hidden_channels=8,
        max_input_len=4,
        fno_modes=4,
        fno_layers=2,
    ).cuda()
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.rand(2, 4, 8, 16, 16, device="cuda")
    with torch.amp.autocast("cuda"):
        loss = model(x).square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(parameter.grad is None or not parameter.grad.is_complex() for parameter in model.parameters())
    scaler.step(optimizer)
    scaler.update()


class _EvaluationFixture(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        x = torch.zeros(2, 8, 4, 4)
        x[-1, 0] = 0.1 * (index + 1)
        return {
            "x": x,
            "target": x[-1, 0:1] * 3.5,
            "valid_mask": torch.ones(1, 4, 4),
            "event_id": f"event_{index}",
        }


def test_evaluate_records_event_level_and_peak_time_metrics() -> None:
    metrics = evaluate(
        None,
        DataLoader(_EvaluationFixture(), batch_size=2),
        torch.device("cpu"),
        lead_times=(1,),
        thresholds=(0.1,),
        primary_threshold=0.1,
        depth_scale_m=3.5,
        amp=False,
        include_per_event=True,
    )
    assert metrics["peak_time_mae_min"] == pytest.approx(0.0)
    assert len(metrics["per_event"]) == 2
    assert {row["event_id"] for row in metrics["per_event"]} == {"event_0", "event_1"}


def _external_metrics(model_type: str = "convlstm", seed: int = 42) -> dict:
    model_rows = []
    persistence_rows = []
    for lead, mae, baseline_mae, csi, baseline_csi in (
        (1, 0.8, 1.0, 0.70, 0.60),
        (3, 1.6, 2.0, 0.60, 0.50),
    ):
        threshold = {
            "0.10": {"tp": 7, "fp": 1, "fn": 2, "tn": 90, "csi": csi, "pod": 0.75, "far": 0.125}
        }
        baseline_threshold = {
            "0.10": {"tp": 6, "fp": 2, "fn": 3, "tn": 89, "csi": baseline_csi, "pod": 0.67, "far": 0.25}
        }
        model_rows.append(
            {
                "lead_steps": lead,
                "lead_minutes": lead * 5,
                "mae_cm": mae,
                "rmse_cm": mae * 2,
                "wet_mae_m": mae / 50,
                "peak_depth_mae_m": mae / 100,
                "csi": csi,
                "pod": 0.75,
                "far": 0.125,
                "threshold_metrics": threshold,
            }
        )
        persistence_rows.append(
            {
                "lead_steps": lead,
                "lead_minutes": lead * 5,
                "mae_cm": baseline_mae,
                "rmse_cm": baseline_mae * 2,
                "csi": baseline_csi,
                "pod": 0.67,
                "far": 0.25,
                "threshold_metrics": baseline_threshold,
            }
        )
    return {
        "schema_version": "external_physical_v1",
        "dataset": "larno_ukea",
        "location": "ukea",
        "model_type": model_type,
        "model_label": model_type,
        "seed": seed,
        "split_seed": 44,
        "test_events": ["event_a", "event_b"],
        "thresholds_m": [0.1],
        "primary_threshold_m": 0.1,
        "per_horizon": model_rows,
        "persistence_per_horizon": persistence_rows,
        "parameter_count": 100,
        "latency_ms_per_sample": 2.0,
        "peak_cuda_memory_mb": 10.0,
        "runtime_sec": 3.0,
        "samples": 20,
    }


def test_external_summary_computes_skill_against_persistence() -> None:
    runs = [_external_metrics("convlstm", 42), _external_metrics("convlstm", 44)]
    validate_external_runs(runs)
    per_run, per_horizon, per_threshold = flatten_external_runs(runs)
    summary = summarize_models(per_run)
    dataset_summary = summarize_datasets(per_run)
    assert len(per_run) == 2
    assert len(per_horizon) == 4
    assert len(per_threshold) == 4
    assert summary[0]["seed_count"] == 2
    assert summary[0]["mae_reduction_pct_mean"] == pytest.approx(20.0)
    assert summary[0]["csi_gain_mean"] == pytest.approx(0.1)
    assert dataset_summary[0]["run_count"] == 2
    assert dataset_summary[0]["location_count"] == 1


def test_external_summary_rejects_protocol_drift() -> None:
    first = _external_metrics("convlstm", 42)
    second = _external_metrics("convlstm_attention", 42)
    second["test_events"] = ["different_event"]
    with pytest.raises(ValueError, match="Inconsistent evaluation protocol"):
        validate_external_runs([first, second])


def test_external_summary_rejects_different_test_sample_counts() -> None:
    first = _external_metrics("convlstm", 42)
    second = _external_metrics("convlstm_attention", 42)
    second["samples"] = 21
    with pytest.raises(ValueError, match="Inconsistent evaluation protocol"):
        validate_external_runs([first, second])


def test_event_pairwise_comparison_preserves_metric_direction() -> None:
    rows = []
    for event_index in range(3):
        for model_type, mae, csi in (
            ("convlstm", 2.0, 0.7),
            ("fno2d_history", 1.0, 0.8),
        ):
            rows.append(
                {
                    "dataset": "larno_ukea",
                    "location": "ukea",
                    "model_type": model_type,
                    "event_id": f"event_{event_index}",
                    "mean_mae_cm": mae,
                    "mean_csi": csi,
                }
            )
    comparison = compare_models_by_event(rows)[0]
    assert comparison["model_a"] == "convlstm"
    assert comparison["model_b"] == "fno2d_history"
    assert comparison["mae_delta_a_minus_b_cm"] == pytest.approx(1.0)
    assert comparison["csi_delta_a_minus_b"] == pytest.approx(-0.1)


def test_external_runner_parsers_reject_duplicates() -> None:
    assert parse_models("conv_lstm,cnn_transformer") == ("convlstm", "cnn_temporal_transformer")
    assert parse_models("u-rnn,fno,simvp") == ("urnn_lite", "fno2d_history", "simvp_lite")
    assert parse_seeds("42,44") == (42, 44)
    assert parse_model_lrs("u-rnn=0.0001,fno=0.0003") == {
        "urnn_lite": 0.0001,
        "fno2d_history": 0.0003,
    }
    with pytest.raises(ValueError, match="unique"):
        parse_seeds("42,42")


def test_external_model_rebuilds_from_checkpoint_metadata() -> None:
    checkpoint = {
        "model_type": "convlstm_attention",
        "input_channels": 8,
        "hidden_channels": 8,
        "num_layers": 1,
        "input_len": 4,
        "lead_times": [1, 3],
        "depth_scale_m": 3.5,
        "use_residual": True,
        "residual_scale": 0.5,
        "args": {"dropout": 0.0, "attention_dropout": 0.1, "transformer_heads": 2},
    }
    model = build_external_model_from_checkpoint(checkpoint)
    output = model(torch.rand(1, 4, 8, 8, 8))
    assert output.shape == (1, 2, 8, 8)
