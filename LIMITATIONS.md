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
- The project does not yet include multi-seed corrected-schema experiments,
  public-data external validation, calibrated uncertainty, or a physical
  hydraulic model.
