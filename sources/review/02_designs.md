# 02 · Design — estimator and benchmark are one system

**Operating rule:** design for information parity, not merely software parity. An estimator can appear better because its benchmark changes initialization, supplies future context, rescales time or applies ground-truth alignment. All findings below come from source control flow; their numerical impact on published results is not yet measured.

## 1. Contracts that determine what accuracy means

| Contract | Valid six-axis test | Risk in current code |
|---|---|---|
| Sensor | Body specific force, body gyro rate, physically valid per-sample \(\Delta t\) | RIANN and VQF use a fixed median-rate interval; TG uses stored dt |
| State | \(h=R_{WB}^{T}e_z\in S^2\) and optional relative-yaw representative | Network emits four quaternion components but heading is not observable |
| Initialization | Same prior, same first observation and fixed physical warm-up | VQF early outputs are overwritten with a later prediction |
| Timing | Input history, maximum future lookahead, output timestamp, compute delay | BiLSTM future context is often omitted from “real-time” claims |
| Reference | Non-overlapping finite GT segments, synchronized clocks, independently frozen alignment | SLERP spans missing truth; optional GT-gyro offset optimization |
| Metric | Unaligned yaw-invariant tilt, invalid truth excluded, independent group IDs | All-invalid truth fallback and full-rotation secondary alignment |
| Calibration | Declared axis/sign/scale/bias and lever arm | Rotation equivariance does not correct physical mounting geometry |

## 2. Four source-level evaluation hazards

**D1. Noncausal VQF backfill.** [run_vqf](../../scripts/evaluate.py) computes a normal VQF sequence, then sets its first min(500, floor(T/4)) output quaternions equal to the quaternion at that later index. The stated approximately five-second warm-up is not a fixed number of seconds: 500 samples span 10 seconds at 50 Hz or 1 second at 500 Hz. This alters early errors (possibly up or down) and is not an online output. Compare (a) untouched startup sequences and (b) all methods after a common predeclared time exclusion. Never backfill one method with future information.

**D2. “Heading alignment” is full SO(3).** [align_heading](../../scripts/evaluate.py) averages the full relative rotations \(R_{\mathrm{GT},i}R_{\mathrm{est},i}^{-1}\) from the first 100 valid GT frames, and applies their quaternion mean to the whole estimated trajectory. The correction is not projected onto yaw SO(2). A constant roll bias can therefore be corrected using GT, suppressing the very roll error measured by the secondary roll/pitch RMSE. The primary tilt RMSE is computed before alignment and is unaffected by D2. Rename the secondary metric “initial full-rotation GT-aligned RMSE” or use a strictly yaw-only projection.

**D3. Differential timestep inputs.** [run_riann_onnx](../../scripts/evaluate.py) generates constant \(1/f_s\), [run_vqf](../../scripts/evaluate.py) configures a constant-step filter, while [run_model](../../scripts/evaluate.py) uses the recorded dt array. For constant-rate recordings there may be negligible effect; for irregular or imputed dt this is a protocol difference. Measure dt dispersion and run a same-rate sensitivity test before declaring directional bias.

**D4. Identity comparison with absent truth.** [validate_sequence](../../scripts/evaluate.py) replaces invalid GT quaternion rows with identity and marks them invalid. Downstream evaluation chooses an all-true metric mask when none of the GT rows is valid. Therefore an entirely untracked sequence can be scored against identity quaternions rather than rejected. The validator also truncates streams to their shortest length; a missing sample can masquerade as valid alignment. Require at least one valid GT frame (preferably a minimum contiguous duration), equal-length or timestamp-indexed sensor streams, and explicit exclusion reasons.

These are direct code observations. No claim is made that the entire released benchmark or published headline numbers necessarily suffer from all four; replay the actual files to establish affected counts and effect sizes.

## 3. Less visible data and experimental design problems

- **Timestamp imputation before validation.** The DIODEM loading path replaces invalid/nonpositive increments with median dt; the original error is no longer visible to physical propagation or gate-coverage statistics. Preserve the original timestamp vector and validity flags. Do not convert a time reversal to an apparently plausible positive interval.
- **Ground-truth interpolation crosses missing segments.** [interpolate_quat](../../src/tg_dcm/so3.py) removes nonfinite rows before neighbor search. A valid–NaN–valid source pattern becomes an interpolable pair; exact last endpoints are also excluded. Interpolate only inside original contiguous valid components.
- **Sampling-rate diversity is not motion diversity.** [data.py](../../src/tg_dcm/data.py) resamples each training recording across 50–500 Hz, then samples overlapping windows. Those frequency variants inherit essentially the same motion and truth errors. Stratify generalization into held-out *rate*, *device*, *carrier* and *motion*, and do not treat correlated windows as independent experimental subjects.
- **Initialization changes the problem.** [evaluate_sequence](../../scripts/evaluate.py) trims all methods to the first GT-valid frame. It is a valid GT-scored benchmark choice, but not a full unassisted cold-start deployment evaluation. Report both protocols with identical initialization and warm-up.
- **The “static” label is norm-only.** Its mask checks whether the acceleration norm is near g, not whether world translational acceleration or angular acceleration is zero. Long translation at \(\|f\|=g\) can pass. Rename the metric “gravity-magnitude-near-g” unless independent stationary truth is provided.
- **Reported runtime is not controlled latency.** CPU RIANN/VQF are timed alongside possible GPU TG inference and concurrent backends. Separate device, threading, batch-1 warm-up, compute time, future-buffer delay and first-output delay.

## 4. Three distinct reliability variables

1. **Reference reliability** is available in training/evaluation through ground-truth timestamps and gyro–GT consistency. It can justify masking supervision, but is **not** a runtime confidence output.
2. **Physical measurement reliability** concerns whether accelerometer specific force approximates gravity. It may use calibrated lever-arm compensation, innovation statistics and a bounded covariance; a norm threshold alone cannot establish it.
3. **Predictive uncertainty** concerns empirical error of the model's output. A softplus-positive covariance or NLL objective does not establish calibrated probabilities without withheld coverage/NEES tests.

The released [loss](../../src/tg_dcm/losses.py) combines these ideas as conditional training penalties, not as a probabilistic inference-time observer. Its gravity loss divides weighted penalties by the **number** of accepted frames rather than by accepted confidence mass. Thus effective branch strength depends on average confidence, not merely the configured coefficient.

## 5. Design that preserves scientific control

Treat every estimator as a replaceable module behind the same validated data and metric interface:

    Raw IMU/GT + timestamps + calibration IDs
       -> integrity checks and immutable provenance
       -> frame/time transform with validity masks
       -> same observable input/history/start protocol
       -> {RIANN,VQF,ESKF,TG-GRU,S2-correction}
       -> raw per-frame tilt with original GT-valid mask
       -> per-sequence and per-raw-motion statistics
       -> grouped paired CI + OOD tails + actual delay

Do not choose an architecture by matching model count alone. A modern BiLSTM [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) gains access to a specified future-frame buffer; [MosaicIMU](https://arxiv.org/html/2606.09355) uses a 500-sample window with bidirectional velocity/variance heads. A physically remounted IMU changes lever arm and sensor error statistics, which [EqNIO](https://arxiv.org/html/2408.06321)/[GINIO](https://arxiv.org/abs/2609.25338) coordinate symmetry cannot automatically neutralize.

**Statistical design:** distinguish mean over segments, raw motions, datasets and frames. Resample DIODEM by original raw motion, not segment. A sequence bootstrap is conditional on fixed model weights; five or more independently trained seeds are required to estimate training variance. Select tuning only on validation, publish all negative results, and generate each table cell from one immutable checkpoint/config.

**Decision:** freeze the experiment protocol, repair integrity, then reassess architecture rankings. A new model cannot receive scientific credit for correcting an evaluation artifact.

## Second-pass design correction: dataset provenance is part of the observation model

The [EuRoC preparer](../../src/tg_dcm/prep/euroc.py) subtracts a bias interpolated from the ground-truth/reference-state file before writing the six-axis benchmark signal. “Model receives six channels” and “only six-axis sensors were ever used to construct the input” are **not equivalent statements**. Freeze two tracks: truly raw gyroscope with independent calibration, and reference-estimated-bias gyro with explicitly declared teacher assistance.

The [Caruso converter](../../src/tg_dcm/prep/caruso.py) writes each physical stream in both corrected and orig directories, changing only intended quaternion antipodal continuity. The 137-entry standard evaluation consequently contains 18 duplicate sensor recordings. The actual released paired results nevertheless differ in **all 18 corrected/orig tilt scores**; the paired HDF5 sources are therefore a required provenance-diff before any dataset-robustness claim.

DIODEM consists of 11 continuous acquisition experiments; 86 motion groups and 422 segment scores are **nested observations**, not 422 or even 86 independent recording sessions. The [2025 descriptor](https://doi.org/10.1038/s41597-025-05468-w) supplies rigid/foam attachment and reference-frame alignment details missing from the derived HDF5 ingest contract. Test both raw-motion- and experiment-session-level clustered uncertainty. [Source and numerical evidence](09_second_pass_verification.md).
