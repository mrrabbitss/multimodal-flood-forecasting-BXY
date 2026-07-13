# Changelog

All notable engineering changes are recorded here. Historical benchmark
results are not rewritten unless they are reproduced by the current code.

## Unreleased - P0 Baseline Audit Closure

### Added

- One-command baseline capture for environment, repository state, model/data
  configuration, event split, metrics, and file identities.
- SHA-256 records for the preserved checkpoint and all 60 fused event files,
  plus an aggregate dataset identity and audit-artifact digest.
- Inference latency and peak CUDA memory metadata with explicit batch, warmup,
  measured-batch, device, and software configuration.
- A committed lightweight audit bundle and P0 requirement-to-evidence matrix.

### Corrected

- Historical metric inference now runs deterministically.
- Efficiency warmup runs after metric inference so cuDNN algorithm selection
  cannot perturb the reproducibility result.
- The final evaluation input tensor is released before peak-memory measurement.

### Verified

- Two consecutive GPU evaluations produced byte-identical core metrics and
  confusion-matrix counts.
- The preserved Conv-LSTM reproduced `MAE=0.0547086373` and
  `CSI=0.9370353465` from 495 windows across nine held-out events.
- All root audit JSON files reload, their aggregate digest validates, the split
  is event-disjoint, and the historical checkpoint hash remains unchanged.

## Unreleased - Batch 4 Multi-Horizon Benchmark

### Added

- Reproducible 48-event, 72-step synthetic data preparation with realtime
  causality validation at maximum lead 24.
- Joint `1/3/6/12/24` target construction without changing the historical
  single-horizon dataset behavior.
- Strong single-frame U-Net, 3D CNN, and ConvGRU baselines.
- Multi-scale, multi-horizon Conv-LSTM U-Net candidate.
- Shared training/evaluation entry points with per-horizon, per-event,
  latency, peak CUDA memory, and parameter-count reporting.
- Five-seed aggregation, paired event bootstrap intervals, and three curated
  comparison figures.
- Optional temporal-consistency and spatial-edge loss terms. Their defaults
  remain zero in historical training paths.

### Verified

- Four models across seeds `42/44/52/77/2026` under one three-epoch protocol.
- Event-disjoint `33/7/8` split with 296 test windows at every forecast lead.
- 37 unit tests, including multi-horizon alignment, model output shape/range,
  and temporal/edge loss behavior.
- Historical Conv-LSTM checkpoint SHA-256 remained unchanged and its 60-event
  metrics reproduced exactly through the compatibility path.

### Result

- 3D CNN is the Batch 4 accuracy winner (`MAE=0.0817`, `CSI=0.8779`).
- The first Conv-LSTM U-Net configuration did not beat the three strong
  baselines and is retained as an honest negative result.

## Unreleased - Batch 3 Experiment System

### Added

- Event-level split manifests with disjointness evidence and event names.
- Paired multi-seed mean/std/min/max summaries and per-seed tables.
- Paired event bootstrap 95% confidence intervals and win/tie/loss counts.
- Lead-time runner for `1/3/6/12/24` steps with a common evaluation threshold.
- Raw, fused, metadata, and leave-one-modality-out channel configurations.
- Canonical last-frame persistence and linear-extrapolation baselines with
  per-event metrics.

### Verified

- Three-seed controlled rainfall comparison at fixed split seed `44`.
- Five-lead single-seed diagnostic and six-variant modality smoke test.
- Historical Conv-LSTM files and checkpoints were not overwritten.

## Unreleased - Batch 2 Rainfall And Schema

### Added

- Causal `rain_current`, `rain_accum_3/6/12`, `rain_max_recent_6`, and
  `rain_trend_3` event fields.
- Versioned channel registry with named `legacy`, `batch1`, `default`, and
  rainfall ablation channel sets.
- Checkpoint `data_schema` manifests and pre-inference compatibility checks.
- Per-event evaluation CSV/JSON output.
- A/B/C input-ablation runner with aggregate and per-event delta figures.

### Verified

- 23-channel CPU end-to-end smoke test.
- Legacy 13-channel checkpoint reproduced its original 60-event metrics.
- Controlled 20-event, single-seed rainfall ablation completed with fixed
  split, budget, and threshold. The cumulative-rain variant improved all three
  held-out events, but is not presented as a formal multi-seed conclusion.

## Unreleased - Batch 1 Trustworthiness

### Added

- Shared `DepthScale` and `RiskThreshold` schemas with checkpoint metadata.
- Spatial social observation, count, confidence, and age maps.
- Current 19-channel input schema and explicit legacy 13-channel compatibility.
- Shared train/validation `LossConfig` and component-level loss reporting.
- Realtime causality validator with machine-readable reports and nonzero exit
  status on violations.
- CSI/IoU equivalence metadata plus HSS, ETS, frequency bias, flood extent
  error, and peak-depth error.
- Unit tests and isolated CPU/GPU smoke-test workflows.

### Changed

- New model runs use a maximum output of `1.2 normalized_depth`, matching the
  synthetic label range.
- Aligned satellite and GIS values are no longer time-decayed by default;
  observation age affects reliability during fusion.
- Missing modalities receive exactly zero fusion weight.
- Validation uses the same configured loss terms as training.

### Compatibility

- `--output_max` remains available as a deprecated alias for `--depth_max`.
- `--value_decay_mode legacy` reproduces the former aligned-value decay.
- Checkpoints without `channel_names` infer the legacy schema when
  `input_channels=13`.
- The preserved Conv-LSTM checkpoint reproduces its original metrics through
  the compatibility path at threshold `0.28`.
