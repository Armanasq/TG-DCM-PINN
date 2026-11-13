# Evaluation

Primary metric:

```text
tilt = 2 * arccos(sqrt(w^2 + z^2))
```

The metric is computed from the quaternion error and ignores yaw.

Secondary metrics:

- roll RMSE after **full SO(3) ground-truth-derived initial correction** (source function is named align_heading, but does not restrict the correction to yaw)
- pitch RMSE after the same ground-truth-derived full rotation (secondary, GT-calibrated; not independent tilt)

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


## Second-pass independence and preparation sensitivity (2026-11-13)

The tables above are retained as **historical published protocol results**, not retroactively changed model scores. The 137 standard sequence files contain 18 physical Caruso recordings twice, once as Caruso-Sassari and once as Caruso-Sassari_orig. The [current converter](../src/tg_dcm/prep/caruso.py) intends only quaternion antipodal sign continuity between these variants, yet all 18 pairs have differing released tilt RMSE and some unequal GT-valid frame counts. The source/data artifact mismatch needs a paired HDF5 field-by-field comparison.

When the orig variants are omitted, the paired TG-minus-RIANN standard mean is **-0.043945° over 119 files**, and an independent 20k paired bootstrap gives an approximate **95% CI [-0.1004°, +0.0105°]**. Thus the earlier -0.090626° fixed-checkpoint improvement is **sensitive to duplicate-weighting**, not a universally resolved generalization advantage.

DIODEM is not 86 independent recording sessions: its [primary 2025 descriptor](https://doi.org/10.1038/s41597-025-05468-w) states **11 continuous experiments** with 86 sub-motions and 422 assessed segment files. Experimental group/bootstrap choice must reflect this hierarchy. Independent experiment-cluster resampling of released errors gives an approximate [-0.433°, +0.021°] RIANN-difference CI, also crossing zero. Reference-frame and rigid/foam sensor provenance have not been recovered for the exported HDF5 files.

[EuRoC-MAV prep](../src/tg_dcm/prep/euroc.py) subtracts reference-state **estimated gyro bias** before writing the benchmark inputs, and chooses a GT/gyro time offset. Therefore this is not an untouched raw gyro experiment. DIODEM timestamp increments are previous-interval indexed; a future gyro-transition loss would need to use forward dt. Per-window reset/live-weight validation also differs from full-stream stateful/EMA evaluation.

The exact source trail, new research findings and unexecuted verification steps are recorded in [second-pass audit](../sources/review/09_second_pass_verification.md).

**Independent check:** python sources/tools/verify_result_sensitivity.py --csv results/full_evaluation/tables/per_sequence.csv --reps 20000 --seed 7. This reproduces the published-data sensitivity with the documented bootstrap implementation, not the authors' original NumPy random sequence.
