# Model Comparison Report

This report summarizes the preserved Conv-LSTM result and the two added architecture attempts.

## Key Findings

- Best model: **Conv-LSTM** with `CSI=0.9370`, `MAE=0.0547`.
- Compared with the best non-neural persistence baseline `persistence_meteo`, CSI improves by `0.1400`.
- MAE is reduced by `50.1%` versus that baseline.
- Conv-LSTM + Attention and CNN-Temporal Transformer are retained as independent architecture extensions, but the original Conv-LSTM remains the current deployment candidate.

## Architecture Metrics

| Model | MAE | RMSE | CSI | F1 | FAR | ms/sample | CUDA MB | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conv-LSTM | 0.0547 | 0.0715 | 0.9370 | 0.9675 | 0.0253 | 1.6739 | 42.6523 | 86977 |
| Conv-LSTM + Attention | 0.0703 | 0.0911 | 0.8957 | 0.9450 | 0.0483 | 1.8938 | 88.4097 | 87522 |
| CNN-Temporal Transformer | 0.0795 | 0.1001 | 0.8657 | 0.9280 | 0.1097 | 8.0547 | 259.3203 | 48225 |

![Architecture metrics](docs/figures/architecture_metrics_dashboard.png)

![Efficiency tradeoff](docs/figures/efficiency_tradeoff.png)

![Model scorecard](docs/figures/model_scorecard.png)

## Baseline and Threshold Analysis

![Baseline comparison](docs/figures/baseline_methods_comparison.png)

![Threshold sensitivity](docs/figures/threshold_sensitivity.png)

## Training Dynamics

![Training dynamics](docs/figures/training_dynamics.png)

## Normalized Score

| model_label | CSI | MAE | FAR | Latency | Memory | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| Conv-LSTM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Conv-LSTM + Attention | 0.421 | 0.374 | 0.728 | 0.966 | 0.789 | 0.655 |
| CNN-Temporal Transformer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

![Radar score](docs/figures/normalized_score_radar.png)
