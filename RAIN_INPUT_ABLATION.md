# Rain Input Ablation

This controlled experiment tests whether direct rainfall inputs add useful
information on the corrected synthetic pipeline. It does not modify or replace
the preserved 60-event Conv-LSTM benchmark.

## Configuration

```text
events: 20 synthetic events
T: 36
grid: 32 x 32
train/validation/test events: 14 / 3 / 3
test event indices: 14, 6, 7
seed and split seed: 44
epochs: 3
hidden channels: 12
lead time: 6
threshold: 0.28 normalized_depth
device: RTX 5060 Laptop GPU
```

All variants use the same generated data, event split, optimizer settings,
training budget, and fixed test threshold.

| Variant | Input definition | Channels | Parameters | Best epoch | MAE | RMSE | CSI | F1 | FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Historical legacy inputs | 13 | 13,177 | 1 | 0.149075 | 0.177112 | 0.652258 | 0.789535 | 0.347742 |
| B | A + `rain_current` | 14 | 13,285 | 1 | 0.106507 | 0.141369 | 0.658394 | 0.794014 | 0.341606 |
| C | A + current and 3/6/12-step accumulated rain | 17 | 13,609 | 2 | 0.077585 | 0.097540 | 0.691391 | 0.817541 | 0.308345 |

![Aggregate rain input ablation](docs/figures/rain_input_ablation.png)

Compared with A, B reduced MAE by `28.6%` and increased CSI by `0.0061`.
C reduced MAE by `48.0%` and increased CSI by `0.0391`, while increasing
trainable parameters by `3.3%`.

![Per-event deltas](docs/figures/rain_per_event_deltas.png)

B and C improved CSI on all three held-out events. The detailed generated
tables are available in
[`docs/experiments/rain_input_ablation.csv`](docs/experiments/rain_input_ablation.csv)
and
[`docs/experiments/rain_per_event_differences.csv`](docs/experiments/rain_per_event_differences.csv).

## Interpretation

The result supports the engineering hypothesis that direct causal rainfall is
useful on this synthetic generator, particularly when recent accumulation is
included. Because this is a single-seed short-budget experiment, it is evidence
for continuing the experiment, not a final superiority claim. Batch 3 should
repeat the comparison across multiple seeds and report mean, standard
deviation, paired event differences, and confidence intervals.
