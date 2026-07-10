# Changelog

All notable engineering changes are recorded here. Historical benchmark
results are not rewritten unless they are reproduced by the current code.

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
