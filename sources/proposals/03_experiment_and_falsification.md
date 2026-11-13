# CCTO — pre-registered experimental and falsification program

**Goal:** decide whether [closure-constrained tilt](01_closure_constrained_tilt.md) adds measurable value beyond classic preintegration, ZUPT and modern learning. **Date:** 2026-11-13. **Status:** plan, not completed external evaluation.

## 1. Claims and evidence requirements

| Claim | Necessary evidence | Evidence that is insufficient |
|---|---|---|
| Gravity identity under endpoint velocity closure | Independent sensor-point velocity reference and dynamic motion with exact relative rotations | A network score without checking closure |
| Correct feasible cap under bounded closure/noise | Truth-in-cap on verified windows, using separately calibrated hard bounds | Probabilistic NLL or softplus covariance called a guarantee |
| Six-axis runtime inference | No velocity, optical GT, contact signal or future sample used at test runtime | GT-derived oracle event flags passed to the model |
| New dynamic regime | Nonstationary equal-but-nonzero endpoint velocities, high intermediate acceleration | Static stance, where ordinary ZUPT works |
| Distinct mechanism | Compare against an ESKF using the **same closure event and integrated measurement** | Uncalibrated filter vs larger learned model |
| Cross-device validity | Device/recording-group split, false-positive risk, tail error, multi-seed study | Duplicate Caruso or 422 correlated DIODEM segments |

## 2. Stage 0: dataset and benchmark correctness

1. Pair [Caruso](../../src/tg_dcm/prep/caruso.py) corrected/orig sensor streams and compare acc/gyr/dt/GT/valid arrays and SHA256. Select one canonical file or cluster by physical source; the released standard CI changes sign-of-significance when the 18 orig duplicates are dropped.
2. Rebuild EuRoC raw-gyro tracks without subtracting reference-state estimated gyro bias; retain a distinct **teacher-debiased** track and record GT/gyro time-offset choices.
3. Trace all 422 DIODEM segments to the [eleven original experiments](https://doi.org/10.1038/s41597-025-05468-w), optical marker positions, sensor-to-body alignment and rigid/foam attachment. Do not assume the optical segment is the instantaneous sensor-point velocity under a nonrigid foam mount.
4. Repair quaternion interpolation gaps, zero-norm finite GT, invalid dt and 0-times-NaN masked loss, VQF future backfill and full-SO(3) GT offset in secondary roll/pitch.
5. Freeze raw-file hashes, checkpoint/EMA identity and sensor frame conventions before architecture trials.

**Training closure labels:** independently obtain sensor-point velocity change from instrumented optical/robot reference, after physically justified smoothing. Such supervision may be used in **training**, but not in a strict six-axis **runtime** mode. Rigid-versus-foam and marker-to-IMU offset must be modeled.

## 3. Stage 1: distinguishable physical motion scenarios

| Scenario | Ground-truth property | Expected behavior |
|---|---|---|
| Static calibration | Sensor point truly stationary | Small bias and gravity baseline; ordinary complementary filter strong |
| Rotational and translational motion returning to same nonzero velocity | Delta v_sensor = 0; high intermediate acceleration | Interval observation helpful even if every strong-motion gravity gate rejects |
| Nearly closing motion | 0 < norm(delta v) <= epsilon_v | Truth inside correctly bounded cap |
| False norm-g gravity | Constant force norm g, zero jerk/gyro, nonzero delta v | No unconditional confidence; false-closure detector stress |
| False closure with plausible spectral motion | Repeatable accelerometer pattern but delta v exceeds assumed bound | Measure classifier false-positive effects |
| Large known/unknown lever arm | Base and IMU endpoint velocities may differ | Correct sensor-point closure; fallback if r unknown |
| Varying gyro bias/clock jitter | Integration error grows across window | Inflate cap radius or reject |
| No valid closure | End velocities always differ appreciably | Explicit gyro/ESKF fallback; do not hallucinate gravity |

**Sensing:** controlled 6-axis IMU + *independent position/velocity reference only for evaluation or training labels*, never hidden at runtime. Benchmark initially with known stationary bias before adding a learned bias model.

## 4. Stage 2: a causal closure-hypothesis detector

Predict the event \(\|\Delta v_s\|\le\epsilon_v\) from historical IMU alone; calibrate on source-disjoint validation. The detector's output is a **probabilistic hypothesis**, not a certified fact or a deterministic hard bound.

- Enforce past-only causal context and predeclared lookback horizons; no future samples.
- Report precision/recall and PR-AUC for **true sensor-point closure**, false-positive rate on devices and motions absent from training, calibration and risk conditional on wrong acceptance.
- Classify a sensor-specific endpoint event rather than generic “walking/periodic” semantics.
- Test a deliberately misleading constant norm-g force: the closure-magnitude residual can equal zero while actual delta v is large.
- Compare *oracle events* (aided, upper bound), a learned IMU-only detector, a conventional ZUPT detector, and no closure.
- If learned events are not reliable under device shift, restrict the paper claim to verified-closure-assisted mode, not universal six-axis reliability.

## 5. Stage 3: required head-to-head baselines

| Baseline | Key controlled difference |
|---|---|
| Untouched VQF6D | Correct causal warm-up and identical initialization |
| Official RIANN ONNX | Frozen neural reference |
| Author RIANN 2026 training reimplementation | Same architecture/data policy, retrained on fixed split |
| TG-DCM released and retrained | Target gate vs interval observation |
| Calibrated 6D adaptive ESKF | Classical gyro and trusted gravity |
| S² observer + classic norm/jerk covariance | Geometry without motion-closure info |
| Preintegration + known closure factor + ESKF | **Hardest novelty comparator:** exactly same informative event/measurement |
| Preintegration + learned closure ESKF | Is the proposed selective cap/update needed? |
| Set-valued attitude observer with equivalent bounds | Does guaranteed feasible geometry alone explain outcomes? |
| Closure integral center, no neural component | Does learning add anything? |
| Learned gate without cap, and cap without learned gate | Attribution to admission versus geometric bound |
| Bias correction alone | Exclude ordinary debiasing as hidden confounder |

Run all at matched sensor calibration, gyro time steps, future-lookahead budget and initial state. Trainable methods: at least **5 independently initialized seeds**, validation-selected checkpoints and held-out physical devices. Record runtime CPU throughput and output delay separately.

## 6. Stage 4: mathematics and implementation property tests

- Check identity \(u_{ij}=R_i^T\Delta v_s+gT h_i\) for nonidentity initial pose, compound rotation and nonzero intermediate acceleration.
- For proven closure/noise bounds, require true h in the computed cap; report bound violations.
- Empty cap, full sphere, small/zero norm u, antipodal geometry and near-pole behavior must be finite and correctly handled.
- Coordinate equivariance \(u'(Q f,Q\omega)=Qu(f,\omega)\) for random proper Q; test complete learned detector and filter state, not only the physical factor.
- Correlated bias, missing samples and lever-arm angular velocity affect preintegration; test error growth and conditional bound tightness.
- Ring buffer should allow O(1) per candidate interval from cumulative preintegrals, O(H) for fixed H horizons; updated bias requires re-integration or Jacobians.
- Use deterministic floating-point tests and repeated real-device initialization. A soft covariance is not an adversarial error guarantee.

## 7. Stage 5: grouped real-world evaluation and go/no-go

**Primary endpoint:** independently prepared *per-physical-recording* yaw-invariant tilt RMSE on held-out devices and session-disjoint high-dynamics closure segments. **Secondary:** session-grouped DIODEM p95, false-closure conditional p95, cap angular size/conditional containment, recovery time, closure event frequency, duration until drift uncorrected, model latency.

**Bootstrap units:** Caruso variants share original physical recording; DIODEM has 422 segments nested in 86 motions within 11 continuous experiments; sensor/session is the evaluation unit for real transfer, not every segment. Paired bootstrap conditions on trained weights; separately quantify seed-to-seed variation. Final hyperparameters must be frozen before confirmatory test.

**Scientific pass condition:** measurable improvement above a **same-event, same-input preintegration/ESKF** on moving intervals with verified equal **nonzero** endpoint velocity, plus consistent positive or at least nonregressive results under learned-event OOD tests. A method that only works with oracle optical closure is **aided**, not an independent IMU-only estimator. A cap that is too wide to change predictions is theoretically correct but empirically unhelpful.

**No-go:** false event admission produces worse tails, signal rates/mounts break guarantees, or ordinary ZUPT/preintegration with the same event matches the proposed architecture. Preserve all negative results; do not add more neural layers to hide a violated physical assumption.

## 8. Paper architecture

1. Derive the **limits** of norm/jerk and closure-magnitude trust, with explicit indistinguishable observations.
2. Derive interval velocity-closure feasibility on \(S^2\), conditional sensor-point and error assumptions.
3. Design the selective causal admission/fallback mechanism; avoid declaring learned closure probabilities to be proofs.
4. Demonstrate coordinate equivariance, efficient rolling integration and failure cases.
5. Compare with **direct six-axis networks**, standard preintegration/ZUPT, same-observation ESKF and bounded set observer.
6. Present a fully audited benchmark correcting duplicate recordings and reference-assisted inputs; publish negative and OOD results.

A paper is credible only when both the *new method* and *the corrected experiment* survive peer-level audit. [Prior art and novelty risk](02_prior_art_and_novelty.md) · [Derivation](01_closure_constrained_tilt.md).
