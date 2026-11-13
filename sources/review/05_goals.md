# 05 — Research goals, hypotheses, metrics and decision criteria

**Research program:** discover and validate algorithms that improve causal gravity-referenced attitude estimation using 6-axis accelerometer/gyroscope data. **Date cutoff:** 2026-11-13. Literature evidence guides hypotheses; no unrun experiments are marked successful. Read [architectures](01_architectures.md), [mathematics](03_mathematics.md), [challenges](04_challenges.md), and the [existing evaluation protocol](../../docs/evaluation.md).

## 1. Primary scientific objective

Estimate the true world-vertical direction expressed in the sensor frame, h(t)=R_WB(t)ᵀe_z, **without magnetometer or externally supplied runtime orientation**, with statistically smaller yaw-invariant tilt error under a held-out distribution of devices and motion regimes. The objective is *not* arbitrary absolute yaw, position, or velocity; those states are not universally observable from the six-axis stream.

The current baseline is the [released TG-DCM-PINN](../../README.md), GRU 2×200 with wxyz quaternion output and target-gated gyro / gravity auxiliary losses. Its published standard-domain 137-sequence mean is 1.521° tilt RMSE and its paired improvement against RIANN is -0.091° (95% interval [-0.147,-0.036]). DIODEM's 422 segments/86 motions have mean 10.41°, median 2.70°, p95 44.20° and max 62.38°; its clustered RIANN-difference interval includes zero. These are **released repository figures**, not independent checkpoint inference.

**Main research goal:** improve unseen-motion p95 and raw-motion failure rate *without sacrificing calibrated uncertainty, latency and reproducibility*, while retaining or reducing the standard-domain mean. Avoid optimizing mean RMSE alone.

## 2. Measurable research questions

| ID | Research question | Baseline to falsify | Required test | Observable outcome |
|---|---|---|---|---|
| G1 | Does strict time/GT integrity change training signals enough to matter? | Current prepared-data pipeline | Reprepare data with gap-aware SLERP, dt validity and frozen clock alignment | Invalid-target fraction, coverage change, paired tilt RMSE |
| G2 | Are gyro and gravity auxiliary gates independently beneficial? | Same GRU, supervised tilt only | Factorial no-gates / gyro / gravity / both, ≥5 seeds | Paired effect by domain, p95 and gate occupancy |
| G3 | Can lever-arm compensation recover useful gravity evidence during rotation? | Heuristic gravity confidence | Known/unknown r, sharp angular acceleration, stationary rotor test | Tilt peak, recovery, false-accept ratio |
| G4 | Does explicit bias state outperform learned implicit memory? | Baseline GRU and augmented GRU | Constant bias, random-walk bias, held-out device, long horizon | Tilt drift, p95, bias RMSE when truth available |
| G5 | Does S² correction improve OOD robustness and calibration? | Quaternion GRU, same sensor/context budget | Equal-compute geometric propagation + tangent residual | Tilt RMSE, NEES, NLL, runtime |
| G6 | Does equivariant architecture improve remount invariance beyond augmentation? | Strong SO(3) augmentation baseline | Algebraic Q tests and real remounts with changed lever arm | Group-law error and physical OOD gap |
| G7 | Do multi-carrier experts or temporal multiscale features address rare motion? | Shared causal encoder | Device/carrier held-out, same training samples and delay | Per-motion effect, router entropy, worst domain |
| G8 | Are improvements reproducible and attributable? | Selected single released run | Multi-seed ablations, identical preprocessing, immutable artifact set | Stable effect CI, exact source-to-table mapping |

Each question describes a **test plan**. A positive answer must be demonstrated; publication alone does not validate any new model.

## 3. Primary and secondary metrics

### 3.1 Primary accuracy estimand

For sequence s of T_s valid frames, let theta_{s,t} be geodesic angular error on S² in degrees. Then

    RMSE_s = sqrt( (1/T_s) sum_t theta_{s,t}² )

and the primary standard-domain metric is the mean of RMSE_s over the 137 original test sequences, **not** a pooled all-frame RMSE.

For DIODEM, raw-motion ID is the resampling cluster. Report (a) segment-weighted mean for comparability with released results and (b) raw-motion-balanced mean over the 86 original motions. These are different estimands and should be explicitly separated, not substituted. Bootstrap a paired mean difference against matched baselines by the correct independent group.

### 3.2 Error reliability

Report median, p90, p95, p99 where sample count supports them, maximum per-sequence RMSE, fractions above prespecified error thresholds, and reconvergence after motion transition. Track **both** standard mean and DIODEM p95, not only aggregate percent improvement. Percent improvement must identify denominator and aggregation unit.

### 3.3 Observation quality and uncertainty

Report GT-valid fraction, target-gate accepted fraction, gravity-confidence mass, and rejected-fraction by angular acceleration/linear acceleration. If a covariance head exists, add 2-D tangent NLL, NEES expected approximately 2 only under proper calibration, uncertainty coverage at 50/90/95%, and calibration under OOD. A low NLL does not guarantee correct coverage.

### 3.4 Causality and computational cost

At minimum, report available IMU axes, whether future samples are consumed, future-lookahead ms, sequence warm-up period, model parameters, CPU/GPU batch-1 latency, sampling frequency, peak memory and real-time factor. Learned velocity models or LiDAR-assisted methods may be included only as *transfer references*, not head-to-head tilt competitors.

## 4. Dataset and protocol goals

1. Freeze test identities and prevent training/validation contamination. The baseline [data.py](../../src/tg_dcm/data.py) uses Myon IDs, TUM-VI room splits and external held-out datasets; preserve the comparison but inspect overlap with raw motion recording IDs.
2. Validate ground truth and timestamp alignment, including exact source endpoints and missing-data discontinuities. Record any GT-informed time-offset choice and apply one consistent protocol across all methods.
3. Establish a direct six-axis comparison set: original TG-DCM, RIANN official, VQF6D, matched supervised GRU and a reproduced adaptive 6D ESKF. A direct 2026 CNN–BiLSTM candidate must be implemented from complete accessible methods if no code is available and evaluated at equal lookahead.
4. Add held-out remount, known bias, nonstationary bias, deliberately invalid timestamps, irregular sampling, hard translation and nonzero sensor lever arms.
5. Define motion taxonomies independent of the evaluated method: angular speed, angular acceleration, jerk, norm deviation, orientation speed, device mount, carrier type, raw experiment. Avoid bins chosen after seeing a favorable result.

## 5. Statistical acceptance policy

A method qualifies as an **empirical improvement** only if it meets all of the following:

- Same test input modalities, initialization and future lookahead as compared baselines; no GT prediction-time information.
- Train/validation/test group disjointness and strict separation of calibration from final test.
- At least five independent seeds per learned configuration and unchanged hyperparameter selection budget.
- Full paired per-sequence outputs, raw-motion cluster bootstrap confidence intervals for DIODEM and explicit reporting of uncertainty under multiple comparisons.
- A predeclared primary measure and p95 guardrail. If the mean improves but DIODEM p95 worsens, call the method a *mean-only tradeoff*, not uniformly better.
- Exact repository commit, checkpoint SHA256, dataset hashes, training/inference commands, frame convention, environment/dependency versions and source-to-table mapping.

**Predefined feasibility gates (engineering targets, not sourced published facts):** no statistically supported regression on standard-domain mean and no material p95 deterioration on DIODEM; independent multi-device improvement required before stating better generalization. The precise numerical acceptable difference should be chosen on validation and signed off before final test evaluation, not retroactively invented from observed results.

## 6. Scientific goals versus engineering goals

**Scientific contributions** should isolate a previously unreported mechanism or validated interaction (e.g. identifiable, calibrated tangent correction; physically compensated gravity reliability under known lever arms; exact full-system gauge/remount law), derive the assumptions, prove or test an invariant, and demonstrate improvement beyond known components. **Engineering contributions** include reproducible data pipelines, robust format normalization, incremental runtime and exact evaluation. Both are valuable; neither substitutes for a demonstrably novel mathematical or empirical claim.

A plausible publication must show what earlier work already contributes:
- [Wei 2025](https://doi.org/10.1016/j.measurement.2024.116097): adaptive dynamic-acceleration filter.
- [Brändle 2025](https://arxiv.org/abs/2502.03681): lever-arm/angular acceleration limitations and bandwidth tradeoff.
- [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784): motion acceleration compensation.
- [EqNIO 2024](https://arxiv.org/html/2408.06321): gravity-aligned O(2) symmetry.
- [GINIO 2026](https://arxiv.org/abs/2609.25338): full SO(3) equivariant networks and covariance.
- [KISS-IMU 2026](https://arxiv.org/html/2603.06205): rare-motion weighting and learned uncertainty.
- [MosaicIMU 2026](https://arxiv.org/html/2606.09355): carrier-specific expert composition.
- [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485): frequency-aware CNN–BiLSTM six-axis quaternion predictor.

Do not claim gravity confidence, equivariance, physics-informed losses or heterogeneous MoE architecture as new merely because combined in one project.

## 7. Decision-oriented milestones

**M0 — Audit:** unit/property tests for quaternion geometry, exact timestamp endpoints, missing GT spans, clock offset protocol, dt validity and consistent evaluation grouping. Deliver a clean regenerated data manifest and corrected published baseline report.

**M1 — Fair direct baseline:** reproduce comparable 6D ESKF and a direct 2026 learned architecture at matched lookahead, then reconstruct standard/DIODEM per-sequence tables, with source hashes.

**M2 — Attribution:** multi-seed factorial gates/uncertainty/bias experiments. Kill configurations that fail error or p95 guardrails, even if accepted-only physics loss decreases.

**M3 — Mechanistic novelty candidate:** geometric S² correction with bounded covariance and/or calibrated rotational-acceleration compensation, followed by exact remount tests and OOD stress.

**M4 — Publication-ready evidence:** complete mathematical statement, released reproducibility commands, ablation table, negative cases, dataset and motion-family breakdown, compute/latency and claim-to-artifact audit.

Each milestone ends with a **falsifiable decision**, not a promise of improvement. Preserve the released baseline unchanged for comparison.
