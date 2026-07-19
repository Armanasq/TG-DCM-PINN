# Evaluation

Primary metric:

```text
tilt = 2 * arccos(sqrt(w^2 + z^2))
```

The metric is computed from the quaternion error and ignores yaw.

Secondary metrics:

- roll RMSE after heading alignment
- pitch RMSE after heading alignment

Baselines:

- RIANN official ONNX model
- VQF 6D orientation filter

Evaluation set:

| Group | Sequences |
|---|---:|
| Original test split | 137 |
| DIODEM good segments | 422 |
| Total | 559 |

DIODEM files under `questionable/` are excluded unless
`--include-diodem-questionable` is passed.

## Aggregation

Each HDF5 sequence contributes one RMSE value to dataset and pooled means. The
pooled 559-sequence mean is therefore dominated by DIODEM and is not a balanced
mean across datasets. Standard-split and DIODEM results are reported separately.

Confidence intervals are paired bootstraps of TG-DCM minus baseline error.
Standard datasets resample sequences. DIODEM resamples raw experiment-motion
groups, retaining the repeated segment measurements within each sampled group.
Negative differences favor TG-DCM.

| Scope | Baseline | Mean difference | 95% CI | Win rate |
|---|---|---:|---:|---:|
| Standard, 137 sequences | RIANN | -0.091° | [-0.147°, -0.036°] | 50.4% |
| Standard, 137 sequences | VQF | -0.247° | [-0.320°, -0.177°] | 69.3% |
| DIODEM, 422 segments / 86 motions | RIANN | -0.195° | [-0.440°, +0.043°] | 50.5% |
| DIODEM, 422 segments / 86 motions | VQF | -5.617° | [-7.548°, -3.857°] | 63.5% |

The mean improvement over RIANN on DIODEM is not resolved at 95%. DIODEM tail
error is also slightly higher for TG-DCM: p95 is 44.20° versus 43.46° for
RIANN. Mean, median, p95, maximum, standard deviation, and coefficient of
variation are stored in `results/full_evaluation/tables/reliability.csv`.

## Artifacts

| Path | Content |
|---|---|
| `tables/per_sequence.csv` | One row per sequence with all metrics and timing |
| `tables/per_dataset.csv` | Dataset means, medians, tails, and win rates |
| `tables/effects.csv` | Paired differences and cluster-bootstrap intervals |
| `tables/reliability.csv` | Spread and tail metrics by method and dataset |
| `tables/summary.json` | Versioned machine-readable summary |
| `repository/all_trials_by_dataset.*` | Every trial grouped by dataset |
| `poster/evaluation_summary.*` | Four-panel A0-poster results figure |

Figures are written in PNG, PDF, and SVG. The vector files retain editable text.
