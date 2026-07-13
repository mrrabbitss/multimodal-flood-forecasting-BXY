# P0 Correctness And Reproducibility Closure

This document maps the original P0 checklist to implemented code and auditable
evidence. Historical Conv-LSTM checkpoints and benchmark artifacts remain
unchanged.

## Completion Matrix

| P0 requirement | Status | Implementation and evidence |
|---|---|---|
| Align model and label depth ranges | Complete | Shared `DepthScale` in `src/data/schemas.py`; checkpoint and evaluation metadata |
| Define normalized threshold semantics | Complete | `RiskThreshold` stores value, unit, and meaning; `0.28` is not a physical depth |
| Add rainfall inputs | Complete | Causal current and rolling rainfall features in `src/data/transforms.py` and the 23-channel registry |
| Remove double remote-sensing/GIS decay | Complete | Corrected alignment default with explicit `legacy` compatibility mode |
| Add social-observation mask | Complete | Spatial observation, count, confidence, and age fields distinguish zero reports from missing reports |
| Complete quality metadata | Complete | Named missingness, age, quality, count, exposure, and drainage channels |
| Use one train/validation loss | Complete | Shared versioned `LossConfig` and component-level reporting |
| Enforce strict causality | Complete | Realtime timestamp and rainfall-derived-field validation plus automated tests |
| Provide multi-seed experiments | Complete | Batch 3 paired seeds and Batch 4 five-seed architecture benchmark |
| Version data and results | Complete | `scripts/capture_baseline.py` and the committed `artifacts/baseline/` audit bundle |

## Baseline Audit Bundle

The audit command records the exact code state, runtime environment, model
configuration, event split, evaluation metrics, latency protocol, and hashes
for the checkpoint and every fused event file.

| Artifact | Contents |
|---|---|
| `environment.json` | OS, Python, packages, PyTorch, CUDA, cuDNN, CPU, memory, and GPU |
| `repository_state.json` | Commit, branch, origin, and pre-capture clean status |
| `config.json` | Model, schema, threshold, loss, evaluator settings, command, and event split |
| `metrics.json` | Regression, risk-mask, loss, latency, memory, and sample-count metrics |
| `file_hashes.json` | Checkpoint SHA-256 and all 60 event file hashes |
| `audit_manifest.json` | Hashes of the five root audit artifacts |
| `evaluation/metrics/` | Original evaluator output and per-event evidence |

Current preserved baseline identity:

```text
checkpoint SHA-256:
388a5ebd7517a54b2d12dad0a73ede0f6587d9bc8a0c96e91b180507958b598f

dataset aggregate SHA-256:
d508ff249aee205f8946d01f847fd07a701ca63e468667accb458357678de3fa
```

## Reproduced Baseline

The committed bundle was captured from a clean `045c5cf` code state. Metric
inference is performed before efficiency warmup and uses deterministic cuDNN,
so two consecutive runs reproduced all core metrics and confusion-matrix
counts exactly.

| MAE | RMSE | CSI | F1 | FAR | Test samples |
|---:|---:|---:|---:|---:|---:|
| 0.0547086373 | 0.0714920014 | 0.9370353465 | 0.9674943138 | 0.0253155724 | 495 |

Efficiency values are tied to their measurement configuration and hardware:

| Device | Batch | Warmup / measured batches | Latency ms/sample | Peak CUDA MB |
|---|---:|---:|---:|---:|
| RTX 5060 Laptop GPU | 8 | 2 / 20 | 1.3716 | 84.67 |

The latency and memory values should only be compared with measurements using
the same batch size, input shape, device, warmup, and software environment.

## Reproduction

```bash
python scripts/capture_baseline.py \
  --fused_dir runs/large60_h24_l1_seed42/data/fused \
  --checkpoint runs/large60_grid_h24_h32_l1/h32_l1_d0_seed44/outputs/checkpoints/best.pt \
  --output_dir artifacts/baseline \
  --batch_size 8 --threshold 0.28 --device auto \
  --warmup_batches 2 --benchmark_batches 20 \
  --overwrite
```

The script refuses to overwrite an existing bundle unless `--overwrite` is
explicitly provided. Full data and checkpoint binaries remain ignored by git;
their identities are represented by the committed hashes.

## Remaining Boundary

P0 closure means the current synthetic benchmark is internally consistent and
auditable. It does not validate transfer to a real city, physical water-depth
units, operational warning thresholds, or sensor-domain shift. Those remain
later-stage data and model-validation tasks.
