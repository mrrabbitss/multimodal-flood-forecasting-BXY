# Synthetic Data Card

## Scope

This repository generates synthetic multimodal urban-flood events for
engineering validation. It is not a measured city dataset and must not be used
as evidence of real-world operational accuracy.

## Depth Scale

The implemented default is:

```text
mode: normalized
minimum: 0.0
maximum: 1.2
unit: normalized_depth
```

Values such as `0.28` are normalized benchmark values, not centimeters or
meters. Physical-unit generation is reserved for a later phase.

## Event Fields

| Group | Fields | Definition |
|---|---|---|
| Latent target | `gt_depth[T,H,W]` | Synthetic normalized water-depth field |
| Rain | `rain[T]`, `rain_current`, `rain_accum_3/6/12`, `rain_max_recent_6`, `rain_trend_3` | Normalized storm hyetograph and causal rolling features |
| Meteorology | `meteo_depth[T,H,W]` | Frequent noisy depth proxy |
| Satellite | `sat_times`, `sat_base`, `sat_quality` | Sparse wet-area proxy and source quality |
| GIS | `gis_times`, `gis_risk`, `gis_quality` | Static event-scale risk map and version quality |
| Social source | `point_t/y/x/value/conf` | Sparse timestamped reports |
| Static context | `topo`, `lowland`, `impervious`, `drainage_capacity`, `exposure` | Synthetic urban context |

## Aligned Social Maps

| Field | Range/meaning |
|---|---|
| `soc_value_map` / `soc_depth` | Confidence-, age-, and distance-weighted local depth |
| `soc_observation_mask` | `1` where a report kernel provides coverage, else `0` |
| `soc_count_map` | Local report-density proxy, nonnegative |
| `soc_confidence_map` | Mean local source confidence in `[0,1]` |
| `soc_age_map` | Mean local report age in time steps |

A zero value with mask `1` means an effective report observed zero depth. A
zero value with mask `0` means there was no valid local observation.

## Quality Metadata

The aligned and fused artifacts preserve `miss_sat`, `miss_gis`, `miss_soc`,
`dt_sat`, `dt_gis`, `dt_soc`, `q_sat`, `q_gis`, `q_soc`, and `n_soc`. Selected
observation timestamps are also stored for causality auditing.

## Input Schemas

Current runs use 23 named channels, including current and accumulated rainfall.
Batch 1's 19-channel set remains available as `batch1`. Historical checkpoints
with 13 channels are loaded through an explicit `legacy` schema. Channel names,
order, registry version, and rainfall feature version are saved in new
checkpoints; silent shape-based remapping is not allowed.

Accumulated rainfall fields store causal rolling sums. Before model input they
are divided by their 3/6/12-step window lengths to keep input scale stable.
`rain_trend_3` is signed; the other rainfall input channels are in `[0,1]`.

## Alignment Modes

- `realtime`: only observations with source timestamp at or before the anchor.
- `offline`: nearest observations may include future timestamps and must not be
  presented as realtime forecasting.
- `value_decay_mode=none`: values remain observations; age affects reliability.
- `value_decay_mode=legacy`: reproduces the former double-decay experiment.

## Split And Bias

Splits are event-level. The generator uses simplified recurrence and proxy
relationships, which can make the task more regular and easier than real urban
flood forecasting. Social coverage, sensor errors, terrain, and drainage are
also synthetic and do not represent a specific population or city.
