import numpy as np

from src.experiments.statistics import paired_bootstrap, summarize_multiseed


def test_paired_bootstrap_orients_lower_is_better_as_positive_improvement() -> None:
    result = paired_bootstrap(
        baseline=[0.30, 0.20, 0.10],
        candidate=[0.20, 0.10, 0.05],
        metric="mae",
        n_resamples=500,
        seed=7,
    )
    assert result["mean_improvement"] > 0
    assert result["ci_lower"] > 0
    assert (result["wins"], result["ties"], result["losses"]) == (3, 0, 0)


def test_multiseed_summary_is_seed_ordered_and_uses_sample_std() -> None:
    rows = [
        {"variant": "A", "seed": 44, "mae": 0.2, "csi": 0.7},
        {"variant": "A", "seed": 42, "mae": 0.1, "csi": 0.8},
    ]
    summary = summarize_multiseed(rows, metrics=("mae", "csi"))[0]
    assert summary["seeds"] == [42, 44]
    assert summary["mae"]["per_seed"] == [0.1, 0.2]
    assert np.isclose(summary["mae"]["mean"], 0.15)
    assert summary["mae"]["std"] > 0


def test_paired_bootstrap_rejects_unpaired_inputs() -> None:
    try:
        paired_bootstrap([1.0, 2.0], [1.0], metric="csi")
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("Expected mismatched pairs to fail")

