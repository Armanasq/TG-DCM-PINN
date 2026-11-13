# Second-pass verification: publication-relevant discrepancies

**Frozen evidence:** [TG-DCM-PINN commit 23069b4](https://github.com/Armanasq/TG-DCM-PINN/tree/23069b4314b1d4e79198d1b7a35e921f41642e92); [per-sequence results](../../results/full_evaluation/tables/per_sequence.csv); [released effect table](../../results/full_evaluation/tables/effects.csv). **Date:** 2026-11-13. **Method:** independent CSV aggregation, source-code inspection, and primary-paper reading. No raw HDF5 or checkpoint re-execution. P0 indicates a claim/protocol blocker, not proof that the released neural weights are incorrect.

## P0-1. Duplicated Caruso recordings invalidate an unqualified 137-independent-sequence interpretation

In [prep/caruso.py](../../src/tg_dcm/prep/caruso.py), Caruso-Sassari and Caruso-Sassari_orig are emitted from the **same** accelerometer, gyroscope, magnetometer and dt arrays for each sensor, with target quaternions respectively fixed by [antipodal continuity](../../src/tg_dcm/so3.py) or stored unmodified. [data.py](../../src/tg_dcm/data.py) includes both directories in test. Therefore **36 entries are 18 repeated underlying sensor streams**, not 36 independent recordings. Antipodal quaternion sign changes must not affect yaw-invariant tilt, so the corrected versus original files should have the same orientation error for each sample with identical validity masks if this were their sole difference.

However the released [per-sequence CSV](../../results/full_evaluation/tables/per_sequence.csv) has **18/18 different paired tilt RMSEs** despite matching duration and norm-gate fraction: mean absolute paired changes are 0.1457565 degrees (TG-DCM), 0.3151442 degrees (RIANN), and 0.2074307 degrees (VQF). Some valid-GT frame counts differ by 1–2. These differences **cannot be explained by quaternion sign flipping alone**. Possible historic-preprocessing mismatch, changed GT validity and source-artifact skew remain unverified without comparing the actual HDF5 contents. This is a verification target, not an established explanation.

Recomputed from all original per-sequence rows (20,000 independent paired sequence-bootstrap replicates with deterministic custom resampling; the released interval uses a different RNG):

| Definition | N | TG-minus-RIANN mean deg | 95% CI |
|---|---:|---:|---|
| Released standard with both Caruso variants | 137 | -0.090626 | [-0.14748,-0.03564] (released) |
| Keep corrected Caruso, omit duplicate orig sensor variants | 119 | **-0.043945** | **[-0.1004,+0.0105]** (sensitivity) |

The deduplicated interval includes zero. The original fixed-checkpoint statistical conclusion is **not robust to a reasonable recording-level weighting choice**; the sensitivity interval is not a new official result or a multiseed confidence bound.

## P0-2. EuRoC uses estimated reference bias in its supposed raw six-axis stream

In [prep/euroc.py](../../src/tg_dcm/prep/euroc.py), the preparer reads gyro bias columns from the EuRoC state_groundtruth_estimate0 reference file, interpolates them onto IMU timestamps and subtracts them from gyro measurements **before writing HDF5**. It also selects a gyro/GT time offset using reference attitudes. This is **reference-assisted preparation**, not untouched raw gyro operation, even though all models later consume only 6-axis IMU + dt. The same processed stream can be provided fairly to all methods, but claims about unassisted raw deployment then require a separate evaluation with a physically available calibration protocol. [EuRoC dataset documentation](https://projects.asl.ethz.ch/datasets/euroc-mav/) distinguishes raw IMU and ground-truth/calibration streams.

## P0-3. DIODEM has only 11 continuous experiment sessions above 86 motion groups

The [original DIODEM descriptor](https://doi.org/10.1038/s41597-025-05468-w) explicitly identifies **eleven continuous experiments**, subdivided into motion types. Five kinematic segments, rigid and foam-mounted IMUs, optical marker orientations, sensor-to-segment alignment and optical-to-earth frame transformation are integral to data interpretation. The processed published IMU vectors are already in the body/segment frame; the target repo reads derived HDF5 but does not provide a source-to-HDF5 DIODEM conversion script proving its tested variant, alignment or exclusion steps. Therefore a large error might reflect sensor attachment, frame inconsistency or label differences as well as learned-model inadequacy. The exact cause remains unknown.

The released DIODEM table has 422 segment entries nested in 86 exp-motion groups and 11 exp experiments. The authors' reported RIANN paired cluster CI is [-0.440,+0.043] over 86 motion groups. Independent source-session bootstrap (20,000 replicates, resample experiments and all contained motions/segments with replacement, then compute the segment-weighted mean) gave approximately **[-0.433,+0.021]**. Both include zero. Session-cluster intervals do not have to be wider on any finite dataset; **interpretation and independent experimental sample count** are the issue. There are only 11 top-level sessions. Also DIODEM p95 sequence-level RMSE is **44.20 deg TG vs 43.46 deg RIANN** in the released reliability table.

## P1-4. DIODEM dt is previous-interval-indexed, but gyro loss uses forward transition

In [scripts/evaluate.py](../../scripts/evaluate.py), DIODEM timestamp differences are assigned dt[0]=median and dt[k]=t[k]-t[k-1] for k>=1. The [target-gated gyro term](../../src/tg_dcm/losses.py) propagates q[k] to q[k+1] using dt[k]. For irregular timestamps the correct forward duration would be dt[k+1]. Thus **the interval convention disagrees**. DIODEM is test-only in current training; do not claim this harmed the released training. It matters for any future DIODEM training, physics consistency, or sample-to-output timing interpretation. The loader also replaces invalid time intervals with median dt, hiding clock anomalies.

## P1-5. Validation and release inference do not use the same state/weights metric

[train_impl.py](../../src/tg_dcm/train_impl.py) chooses its checkpoint using per-window, hidden-state-reset validation with pooled per-frame tilt RMSE, whereas [scripts/evaluate.py](../../scripts/evaluate.py) carries recurrent state through long sequences. Live weights are selected and saved along with an EMA shadow; the evaluator uses EMA by default unless --raw is passed, consistent with [release metadata](../../training/summary.json). **EMA is not separately selected on the published validation objective**; a streaming-validation/EMA sensitivity study is absent. This is a protocol/selection discrepancy, not evidence of inferior EMA predictions.

## P1-6. The best-score narrative ignores two held-out losses

[per_dataset.csv](../../results/full_evaluation/tables/per_dataset.csv) reports TG-minus-RIANN = **+0.130163 deg on all six EuRoC recordings (0% wins)** and **+0.079612 deg on 71 OxIOD recordings (29.6% wins)**. The aggregate standard mean favors TG because other domains, including both Caruso variants, offset these losses. “Generalizes better across all domains” is **not supported**. Report device-balanced means and pair effects alongside the pooled metric.

## P1-7. The author's modern RIANN training code was missing

[Daniel Weber's RIANN training reimplementation](https://github.com/daniel-om-weber/riann_dev/tree/435274141c08ab318c01be904de9754943b4feed) (Apache-2.0) provides train.py, riann/data.py, dataset preparers, HDF5 comparison and VQF validation scripts. It explicitly describes itself as a **reimplementation** because original training code was lost; it must not be identified as the original released training run. Its code uses a two-layer GRU, known augmentation, Ranger, and very similar Myon/TUM-VI splits and multi-rate resampling. Its EuRoC preparer also subtracts reference-estimated gyro bias. The meaningful independent TG-DCM claim concerns its *target-gated objective and measured effect*, not the GRU/data pipeline alone. Canonical RIANN paper DOI: **[10.3390/ai2030028](https://doi.org/10.3390/ai2030028)**, rather than an unrelated IEEE LRA identifier.

## P1-8. Additional geometry papers limit novelty claims

[Optical-flow/IMU almost-global observer (2025)](https://arxiv.org/abs/2508.21163), [Reduced Order Observers for Monocular VIO (2026)](https://arxiv.org/abs/2609.26612) and [Nonlinear Observer Design for VIO (2026)](https://arxiv.org/abs/2601.07156) already use **body-frame gravity state**, geometry and bias/velocity coupling. Their stability guarantees rely on **vision, optical-flow or bearing information** and excitation assumptions. They are not equal six-axis-only baselines, but **body-frame gravity/S2 state itself is prior art**. Novelty must concern a new update, identifiability condition, robustness mechanism or controlled empirical result.

## P2-9. Source acquisition and independent replication remain incomplete

The existing sources/ contains authored notes and citations, but no downloaded full publisher paper corpus, cloned external repositories or executed external checkpoints. Local Git cloning failed because the execution container could not resolve github.com. Remote GitHub source inspection and scholarly full text were available and are distinguished from local execution. Do not label these experiments reproduced.

## Verification priorities and responsible claim

**First:** hash/diff the 18 Caruso paired HDF5 files, confirm exact input and GT validity transforms, and rerun yaw-invariant metrics. **Second:** reconstruct the frozen 137 standard results with physical-recording-balanced resampling and a non-duplicated set. **Third:** recover DIODEM source preprocessing metadata, original session groups and rigid/foam flags, then stratify failures. **Fourth:** run raw and reference-debiased EuRoC as separate tasks. **Fifth:** unify validation window/state/weights and timestamp convention, with NaN-safe physics tests.

The one previously “statistically resolved” standard mean advantage becomes **not resolved under a reasonable deduplication**, while DIODEM has 11 higher-level continuous experiments and no resolved RIANN advantage. These findings limit **evidence and attribution**, not the correctness of the TG-DCM mathematical idea.

**Reproduction utility:** [verify_result_sensitivity.py](../tools/verify_result_sensitivity.py). Run it on the immutable [per-sequence CSV](../../results/full_evaluation/tables/per_sequence.csv) to regenerate the deduplication and experiment/motion bootstrap summaries. The script has a standard-library-only Python runtime, provides a SHA256 for the supplied CSV, and was smoke-tested on a synthetic fixture; the original full CSV was reanalyzed through the GitHub connector in this audit, not executed by the container script.
