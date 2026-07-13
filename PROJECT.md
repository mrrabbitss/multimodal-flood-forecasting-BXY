# Project Report

This file is a concise project report. The main GitHub-facing document is
`README.md`.

## Overview

This project implements an end-to-end multimodal flood-risk forecasting demo.
It simulates urban flood events, aligns asynchronous observations from multiple
modalities, fuses them with dynamic reliability-aware rules, and forecasts
future water-depth maps with spatiotemporal neural networks.

The project is designed to show a complete engineering workflow:

```text
synthetic data -> multimodal alignment -> dynamic fusion -> forecasting model
-> metrics -> visualization -> architecture comparison
```

## Modalities

The synthetic generator first creates a hidden ground-truth water-depth field.
Each modality then observes this field with different frequency, noise,
missingness, and delay:

- Meteorology: high-frequency estimated water depth.
- Remote sensing: low-frequency satellite flood/wet-area proxy.
- GIS: static background risk.
- Social reports: sparse crowdsourced depth observations.
- Metadata: missing flags, time gaps, social-report counts, and reliability
  signals.

## Main Model

The preserved main model is a lightweight Conv-LSTM forecaster:

```text
Input [B,T,C,H,W]
  -> Conv2d encoder
  -> ConvLSTMCell
  -> Conv2d head
  -> Output [B,1,H,W]
```

Current best checkpoint:

```text
runs/large60_grid_h24_h32_l1/h32_l1_d0_seed44/outputs/checkpoints/best.pt
```

Recommended risk threshold:

```text
0.28 normalized_depth
```

This is a normalized synthetic-benchmark threshold, not a centimeter value.

## Architecture Extensions

Two additional architectures were added without modifying or overwriting the
original Conv-LSTM results:

- Conv-LSTM + Attention
- CNN-Temporal Transformer

The extension code lives in:

```text
src/model_variants.py
src/train_architecture.py
src/evaluate_architecture.py
src/compare_architectures.py
```

## Current Results

All rows use the same 60-event fused dataset, split seed `44`, and threshold
`0.28 normalized_depth`.

| Model | MAE | RMSE | CSI | Latency ms/sample | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| Conv-LSTM | 0.054709 | 0.071492 | 0.937035 | 1.674 | 42.65 |
| Conv-LSTM + Attention | 0.070253 | 0.091082 | 0.895708 | 1.894 | 88.41 |
| CNN-Temporal Transformer | 0.079548 | 0.100123 | 0.865670 | 8.055 | 259.32 |

The preserved Conv-LSTM remains the strongest model on the current synthetic
split. These rows use the historical 13-channel schema. Batch 1 correctness
changes use a separate 19-channel schema and do not relabel these results.

CSI is numerically identical to IoU under the current binary flood-mask
definition.

## Batch 2 Rainfall Schema

The current default input schema has 23 named channels and adds causal current
and 3/6/12-step accumulated rainfall. New checkpoints save a versioned data
schema, exact channel order, and rainfall transform version. The 19-channel
Batch 1 and 13-channel historical schemas remain explicitly loadable.

A controlled 20-event, three-epoch experiment found that the 17-channel
legacy-plus-accumulated-rain variant reduced MAE from `0.149075` to `0.077585`
and increased CSI from `0.652258` to `0.691391`. All three held-out events
improved, but this remains a single-seed diagnostic rather than a formal main
benchmark. See `RAIN_INPUT_ABLATION.md`.

## GitHub Packaging

The repository intentionally ignores generated data, checkpoints, and run
outputs:

```text
data/
outputs/
runs/
*.npz
*.pt
```

This keeps the GitHub repository lightweight and source-focused. Large artifacts
can be published separately through GitHub Releases, cloud storage, or model
hosting platforms.

## Batch 3 Experiment System

Batch 3 adds event-disjoint split manifests, paired multi-seed summaries,
bootstrap confidence intervals, per-event baseline tables, configurable input
and modality channel sets, and lead-time evaluation at `1/3/6/12/24` steps.

On the controlled 20-event dataset, the three-seed cumulative-rain Conv-LSTM
reaches `MAE=0.0824 +/- 0.0042` and `CSI=0.6885 +/- 0.0345`, compared with
`MAE=0.1434 +/- 0.0132` and `CSI=0.6515 +/- 0.0013` for the legacy inputs. The
paired MAE improvement has a positive 95% bootstrap interval; the CSI interval
crosses zero and is reported as inconclusive. See `BATCH3_EXPERIMENTS.md`.
