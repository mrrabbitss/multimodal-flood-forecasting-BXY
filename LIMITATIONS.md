# Limitations

- The primary benchmark is synthetic. Reported scores do not establish
  performance for any real city, storm, or emergency workflow.
- `normalized_depth` is dimensionless and must not be converted to centimeters
  or meters without a separately defined and validated physical mapping.
- The generator can create cleaner relationships among rainfall, proxy
  modalities, and targets than real sensors provide.
- The preserved headline benchmark uses the historical 13-channel schema and
  legacy generated artifacts. It is retained for reproducibility, not relabeled
  as a corrected-schema result.
- The current CNN-Temporal Transformer result applies only to the implemented
  architecture, split, and training budget. It is not evidence that
  Transformers are generally weaker than Conv-LSTM.
- CSI and IoU are the same statistic for the current binary flood-mask
  definition.
- `uncertainty_low` and `uncertainty_high` are heuristic
  modality-disagreement bounds. They are not calibrated 95% confidence
  intervals.
- Threshold selection, hyperparameter tuning, and early stopping must use
  validation data. The test set must remain fixed for final reporting.
- The project now includes external validation on the public UrbanFlood24 and
  LarNO UKEA physical simulation datasets. It still lacks validation against
  field sensors, gauge observations, surveyed inundation, or operational city
  emergency workflows.
- The UKEA benchmark uses all available windows and five training seeds, but it
  contains only 20 events from one domain.
- The UrbanFlood24 benchmark covers all three locations and three seeds, but
  caps each event at eight sampled windows. It is a controlled sparse-sampling
  benchmark, not the final full-window five-seed result.
- External benchmark seed means and deviations are reported, but paired
  per-event bootstrap intervals and cross-dataset transfer tests remain future
  work.
- The Batch 2 rainfall comparison is a controlled 20-event, single-seed,
  three-epoch diagnostic with three held-out events. Its improvement must be
  confirmed by the later multi-seed experiment batch before becoming a main
  benchmark claim.
- Batch 4 uses 48 synthetic events and only eight held-out events. Its paired
  bootstrap intervals quantify variation across those events, not across
  cities, sensors, or real storms.
- Batch 4 fixes the data, split, seed set, epochs, hidden width, threshold, and
  loss settings, but does not match parameter counts across architectures.
- The Batch 4 three-epoch budget may favor faster-converging models. The 3D CNN
  win and Conv-LSTM U-Net loss apply only to this controlled configuration.
- Historical synthetic forecast leads are simulation steps with no validated
  mapping to hours. The external benchmark separately uses source-aligned
  5-minute steps and reports `5/15/30/60 min` physical horizons.
