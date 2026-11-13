# Reproducible research experiment protocol

## Questions to answer

- Does target consistency gating improve tilt on unseen devices, high-acceleration motions and raw motion groups, or merely remove problematic training signals?
- Is a norm/jerk/rate confidence function informative during true translational acceleration with unchanged accelerometer magnitude?
- Does geometric S^2 correction with gyro bias and uncertainty improve tail RMSE and calibration without increasing lookahead?
- Does source-frame remount equivariance survive physical IMU reinstallation, miscalibration and lever-arm changes?
- Can a modern causal TCN or matched-delay BiLSTM outperform the 2x200 GRU at the same compute budget?

## Data and split

Preserve the current 137-sequence standard test set and DIODEM 422 segments/86 motions as immutable held-out sets; inspect for identity, timestamp and motion-source overlaps with training and validation. Use motion-group clustering (all DIODEM segments from a raw motion stay in one partition). Add leave-one-device, leave-one-dataset and high-angular-acceleration OOD protocols. Group controls by static, slow rotation, long dynamic motion, translation acceleration, rapid gyro impulses, uncalibrated bias/temperature proxy and sensor remount.

Each sample must carry a canonical schema: SI units, body/world transform direction and handedness, gravity sign, quaternion order, synchronized timestamps, original index, integrity mask and raw-motion group ID. No interpolation across invalid ground-truth gaps, discontinuities or nonpositive dt. Record calibration/timing offset decisions and do not select them on the final test labels.

## Matched comparison

Minimum: released TG-DCM; supervised tilt-only GRU; supervised+gyro gate; supervised+gravity gate; supervised+both; VQF6D; RIANN official; direct adaptive 6D ESKF; a reproducibly implemented direct 2026 CNN–BiLSTM if the missing primary method and weights can be obtained. Use exactly same held-out sequences, tilt definition, initialization period, causal observation budget and sampling convention.

Do not place a learned velocity/ATE paper in the same accuracy ranking. For BiLSTM report lookahead samples and corresponding milliseconds. For all methods report prediction interval, device, peak memory, compute FLOPs/latency and real-time factor.

## Minimum ablations (factorial and single changes)

- Target gate on/off; residual threshold (0.5, 1, 2, 5 degrees); accepted-pair fraction versus motion regime.
- Gravity gate on/off; each of norm, jerk and rate factors removed independently; count-normalized versus mass-normalized confidence loss.
- Joint dynamics vs gravity loss; alternate weights; Charbonnier vs robust angular alternatives; deadzones zero/nonzero.
- Physics target on S^2 vs raw quaternion output (with consistent gauge treatment).
- Gyro constant/slowly varying bias augmentation, correct/incorrect sensor coordinate frame, drift and reset.
- 0/1/5/20-sample future-lookahead budgets at target frequencies; compare throughput, TTFT of first estimate and steady-state.
- Continuous calibration quality (ECE, NLL, coverage, NEES) for any learned covariance, with SPD assertions and sensor rotation law.

## Mathematical property tests

1. q and -q produce same tilt, loss and gradient interpretation.
2. h(R_z(theta) R)=h(R); absolute yaw-only transformation must not change tilt.
3. h(q_{t+1}) matches exp(-[omega]x dt) h(q_t) for valid single-axis/combined increments.
4. Invalid GT gap is never bridged; exact valid endpoints remain usable.
5. Nonfinite, negative, zero, duplicate and nonmonotone dt are masked, not changed into artificial physical measurements.
6. Rotating raw gyro and accel by a random Q changes h and covariance according to the defined frame law; treat physical lever-arm remount as a separate test.
7. Bootstrap sampling unit equals source motion; do not resample DIODEM segments independently.
8. Every plotted table row is generated from one immutable checkpoint, config, dataset manifest and command; no checkpoint mixing or reused labels.

## Statistical claim threshold

Run >=5 independent seeds per learned configuration, fixed search budget and frozen hyperparameter selection on validation only. Primary unit of measurement: sequence-level tilt RMSE (degrees). Secondary: dataset-balanced mean, per-sequence median, 90th/95th percentile and max, conditional gate coverage, per-motion failure fraction and per-device OOD gap. Compute paired bootstrap CI on sequences (standard) and cluster paired bootstrap on raw motions (DIODEM). Correct for multiplicity if selecting among many configurations. Distinguish statistically significant aggregate improvements from deployment-important tail deterioration. Track per-seed compute cost and bootstrap random seed.

## Provenance for every result

Immutable source commit; environment lock; model checkpoint SHA256; dataset URI and dataset hash; GT alignment and calibration config; seed; training and inference commands; per-frame predictions/masks; per-sequence CSV; derived tables/figures; unit tests; explicitly marked unknowns and failures. Use statistical/evaluation scripts from actual code rather than manually copied paper tables.

## Research claims

A method is only **demonstrated** when it improves a matched, preregistered held-out metric without violating causal or observability assumptions and reproducibly passes property tests. Otherwise mark **hypothesis**, **reported by authors**, or **not independently verified**.

## Protocol blockers discovered in source review (2026-11-13)

Before testing a new architecture, fix or measure these benchmark differences:

- VQF's first min(500,floor(T/4)) quaternions are replaced with its later estimate; provide raw causal startup and common physical-time burn-in, **never future backfill**.
- Secondary "heading" alignment applies a GT-derived full SO(3) offset, which can remove roll/pitch error. Keep the primary raw tilt metric, and redesign or relabel these secondary results.
- RIANN/VQF use fixed median-rate dt while TG uses recorded per-sample dt; log per-sequence dt dispersion and run matched-input sensitivity tests.
- Exclude wholly missing GT rather than substituting identity as reference; reject silent unequal-length truncation and bad/reversed/duplicate timestamps.
- Build NaN-safe physics losses by cleaning inputs **before** angular residuals; test 0*NaN and finite zero-norm quaternion edge cases; reject nonpositive dt before any clamp.
- Report independent raw-motion count, sequence/segment vs raw-motion estimand, and a five-seed hierarchical experiment instead of treating a single-checkpoint bootstrap as model-training uncertainty.
- Do not treat abs(||acc||-g) as proof of static physical motion; test sustained constant-magnitude false gravity explicitly.
- Separate actual sample-to-output delay from model throughput and from any BiLSTM future-sample budget.

The source-level audit documents **possible metric effects**, not measured corrections. Reproduce dataset incidence and per-sequence before/after tables prior to changing the research narrative.

The [2025 RSS continuous bias NODE](https://arxiv.org/html/2504.09495) and [2026 empirical bias preprint](https://doi.org/10.31224/7144) strengthen the requirement to control bias-initialization provenance and label preprocessing independently of network family.

