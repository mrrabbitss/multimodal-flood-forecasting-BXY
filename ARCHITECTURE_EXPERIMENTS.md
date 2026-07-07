# Architecture Extension Experiments

This note documents the added architecture experiments without changing the
original Conv-LSTM training scripts or existing Conv-LSTM checkpoints.

## What Was Added

- `src/model_variants.py`
  - Keeps compatibility with the original `ConvLSTMForecastNet`.
  - Adds `ConvLSTMWithAttentionForecastNet`.
  - Adds `CNNTemporalTransformerForecastNet`.
  - Provides checkpoint-aware model construction for old and new checkpoints.
- `src/train_architecture.py`
  - Trains a selected architecture variant into a separate output directory.
  - Reuses the existing dataset split, loss, threshold, and metrics utilities.
- `src/evaluate_architecture.py`
  - Evaluates MAE, RMSE, CSI/F1/FAR.
  - Benchmarks inference latency.
  - Records peak CUDA memory.
- `src/compare_architectures.py`
  - Evaluates the existing Conv-LSTM checkpoint read-only.
  - Trains/evaluates the two new models.
  - Writes CSV/JSON summaries and comparison figures.

## Preservation Rule

The original Conv-LSTM result is preserved. The comparison script reads:

```text
runs/large60_grid_h24_h32_l1/h32_l1_d0_seed44/outputs/checkpoints/best.pt
```

and writes all new outputs under:

```text
runs/architecture_comparison/
```

No original checkpoint or original `outputs/` folder is overwritten.

## Reproduction Command

```powershell
C:\Users\23173\Anaconda3\envs\floodwatch\python.exe -m src.compare_architectures `
  --output_root runs\architecture_comparison `
  --epochs 8 `
  --batch_size 4 `
  --hidden 32 `
  --transformer_heads 4 `
  --transformer_layers 2 `
  --seed 44 `
  --threshold 0.28 `
  --device cuda `
  --no-progress
```

To regenerate only evaluation tables and figures from existing checkpoints:

```powershell
C:\Users\23173\Anaconda3\envs\floodwatch\python.exe -m src.compare_architectures `
  --output_root runs\architecture_comparison `
  --skip_training `
  --batch_size 4 `
  --hidden 32 `
  --transformer_heads 4 `
  --transformer_layers 2 `
  --seed 44 `
  --threshold 0.28 `
  --device cuda `
  --no-progress
```

## Result Summary

All rows use the same fused dataset, split seed `44`, test events, and threshold
`0.28`.

| Model | MAE | RMSE | CSI | Latency ms/sample | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| Conv-LSTM | 0.054709 | 0.071492 | 0.937035 | 1.674 | 42.65 |
| Conv-LSTM + Attention | 0.070253 | 0.091082 | 0.895708 | 1.894 | 88.41 |
| CNN-Temporal Transformer | 0.079548 | 0.100123 | 0.865670 | 8.055 | 259.32 |

Current conclusion: the preserved Conv-LSTM remains the best model on this
test split. The new models are useful ablation/extension attempts and are saved
as separate checkpoints for future tuning.

## Output Files

```text
runs/architecture_comparison/architecture_comparison.csv
runs/architecture_comparison/architecture_comparison.json
runs/architecture_comparison/figures/architecture_metrics.png
runs/architecture_comparison/figures/architecture_efficiency.png
runs/architecture_comparison/figures/architecture_training_curves.png
runs/architecture_comparison/convlstm_attention/outputs/checkpoints/best.pt
runs/architecture_comparison/cnn_temporal_transformer/outputs/checkpoints/best.pt
```
