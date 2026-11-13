# 04 · Challenges — ranked failure mechanisms, not a list of topics

**Scope:** 2024–2026 six-axis tilt and transferable inertial learning. An item is a *challenge* only when the failure mechanism and distinguishing experiment are explicit. Source-level evidence is separated from model behavior that still requires a checkpoint/data replay.

| Priority | Failure mechanism | What is usually missed | Discriminating experiment | Evidence |
|---|---|---|---|---|
| P0 | Invalid truth/time contaminates supervised physics | Smooth SLERP can manufacture observations inside GT dropout | finite–NaN–finite, exact endpoint, bad dt | [SO(3) preprocessing](../../src/tg_dcm/so3.py) |
| P0 | Benchmark applies different causality/GT calibration policies | VQF backfill uses a *future* orientation; “heading” correction is full 3D GT rotation | Untouched startup, common burn-in; raw versus yaw-only R/P | [evaluation](../../scripts/evaluate.py) |
| P0 | NaN masking is algebraically insufficient | 0 × NaN remains NaN after invalid nonlinear residual | Inject invalid gyro/acc at masked frame, inspect forward and gradients | [losses](../../src/tg_dcm/losses.py) |
| P1 | Gravity and translation are non-identifiable | Constant acceleration can preserve norm and have zero jerk | Constant-norm 30° false-gravity adversary | [mathematics](03_mathematics.md) |
| P1 | Selection induced by hard training gate | Low auxiliary loss may reflect fewer difficult motions | Gate survival versus error/motion severity | [losses](../../src/tg_dcm/losses.py) |
| P1 | Rare motion dominates deployment risk | 86 original DIODEM motions, not 422 independent test subjects | Raw-motion cluster p95 and leave-group-out | [evaluation](../../src/tg_dcm/evaluation.py) |
| P1 | Bias, calibration and motion priors are confounded | A “bias model” can learn carrier motion from supervised sequence context | Held-out devices, temperature proxies and synthetic bias jumps | [RSS bias NODE 2025](https://arxiv.org/html/2504.09495) |
| P2 | Covariance is plausible but not calibrated | A large variance can hide errors without better point estimates | NLL, NEES, coverage and sharpness under OOD | [EqNIO](https://arxiv.org/html/2408.06321), [MosaicIMU](https://arxiv.org/html/2606.09355) |
| P2 | Architectural symmetry is not system symmetry | Extra attitude, bias, covariance or time inputs break transform laws | Full-system Q action and real remount | [GINIO issues](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) |
| P2 | Paper-to-code correspondence fails | Headline table may mix checkpoints/training domains | Row-to-commit/checkpoint/dataset manifest mapping | [GINIO code audit](../implementation/external-code-audit.md) |

## 1. Identifiability versus learned plausibility

A model's error on the dataset distribution is not an observability guarantee. If two physically possible states and accelerations generate the same IMU sequence, no deterministic estimator can be correct for both without an additional prior. In six-axis sensing, world-yaw is an exact gauge; translational acceleration and gravity can be indistinguishable over finite windows.

**Hidden consequence:** a high-capacity architecture can exploit correlations between gait, carrier identity, sensor mounting and tilt labels. When that correlation changes, the model may become confidently wrong. The right tests are *counterfactual pairs* with matched sensor statistics but different physical causes, not only randomized train/test windows. [Brändle 2025](https://arxiv.org/abs/2502.03681) and [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784) show that even the apparent gravity direction is shaped by lever-arm physics and filter bandwidth.

## 2. Bias is not noise, and neither is the trajectory prior

Ordinary white-noise injection cannot simulate all gyro error mechanisms: constant bias, time-correlated random walk, scale/cross-axis miscalibration, temperature drift and vibration have distinct temporal spectra. [Liu et al., RSS 2025](https://arxiv.org/html/2504.09495) model explicit continuous bias dynamics, but use pose GT and pose-derived initial bias estimates during training; a causal spline and a stationarity-based initial condition are part of the model. These details limit portability to unknown startup or arbitrary excitation.

The [2026 large-scale bias study](https://doi.org/10.31224/7144), a **non-peer-reviewed preprint**, compares MLP/LSTM/TCN/Transformer in 2,811 experiments and reports that dataset choice and label quality dominate architecture choice for its EuRoC/TUM-VI setup. It also observes failure of an SO(3) auxiliary loss in its tested configurations. These are not universal percentages. They do challenge an untested belief that “adding physics” or “using a transformer” necessarily improves a bias model. Demand equal teacher-label quality and an independent held-out hardware condition before attributing improvement to architecture.

## 3. The physics loss may vanish on precisely the difficult regime

The gyro dynamics branch uses GT propagation agreement to choose which transitions receive regularization. The gravity branch multiplies three trust factors that can approach zero as motion becomes severe. A physically *difficult* sequence can therefore receive much less auxiliary supervision even if its **primary supervised tilt error is high**. This is not inherently wrong—corrupt targets should be excluded—but the intended claim must be conditional: better regularization on label-consistent and gravity-like samples.

**Discriminating tests:** record per-raw-motion survival \(P(m=1\mid\|\omega\|,\|\dot\omega\|,\|f\|-g)\), average confidence mass, effective sample size and unconditioned tilt error of accepted versus rejected pairs. Then perform no-gate, hard-gate and robust soft-gate ablations with fixed data and seeds. A 0.1-degree mean improvement accompanied by 45-degree p95 errors is not evidence of better reliability.

## 4. Data processing can invert research conclusions

The current evaluation [run_vqf](../../scripts/evaluate.py) retrospectively replaces initial estimates, while [align_heading](../../scripts/evaluate.py) applies full GT-derived orientation to secondary results. [load_h5](../../scripts/evaluate.py) imputes invalid DIODEM dt with median dt; [validate_sequence](../../scripts/evaluate.py) may compare all-invalid truth against identity. [so3.py](../../src/tg_dcm/so3.py) bridges missing reference intervals during interpolation. Each changes the *measurement problem*, not just code style.

A strong prepublication study should rerun the released checkpoint after repairs, then publish a table separating **algorithm change**, **data-protocol change** and **metric change**. Never attribute a preprocessing gain to a neural block.

## 5. Motion-dependent failure and sampling-unit mismatch

The released figures show DIODEM mean tilt RMSE 10.41°, median 2.70°, p95 44.20° and max 62.38° over 422 segments derived from 86 raw motions. This is a heavy tail at the **sequence/segment metric** level, not a per-frame 95th percentile. A confidence interval that clusters raw motions addresses dependence but a segment-weighted mean still answers a different question from an equal-motion mean.

**Challenge:** determine whether failures arise from reference alignment, initialization, motor vibration, raw gyroscope quality, motion class, body frames or a learned network prior. Without this decomposition, a motion-reweighting method may merely amplify badly labelled segments. [KISS-IMU](https://arxiv.org/html/2603.06205) provides a useful GMM-based weighting example, but its LiDAR pseudo-label quality is an assumption too.

## 6. Closed-loop filter stability and covariance

Under dynamic acceleration, simply lowering gravity gain increases a filter's time constant. [Brändle](https://arxiv.org/abs/2502.03681) analyzes this tradeoff with rotational acceleration. Learned NLL-based covariance can become large on hard examples; a filter may become accurate on easy sequences because it effectively ignores observations. Validate **innovation coverage, recovery time and closed-loop effect**, not just a lower numerical training NLL.

For a proposed two-dimensional tilt innovation, require SPD covariance, calibrated NEES near dimension two *under the assumed Gaussian model*, low selective risk and bounded maximum variance. A learned variance head that says “unknown” during all difficult cases is a useful abstention signal only if the downstream observer handles the outage and the fraction of such cases is reported.

## 7. The non-obvious statistical trap

The standard split's reported paired mean difference against RIANN, -0.091° with interval [-0.147,-0.036], estimates test-sequence variability **conditional on the selected trained model and preparation pipeline**. It does not quantify seed-to-seed training variation, hyperparameter search selection, hardware shifts, or uncertain label alignment. No controlled multi-seed loss ablation is available. Treating a sequence-bootstrap exclusion of zero as proof of *algorithmic novelty* is a category error.

**Standard required:** at least five training seeds, unique raw-motion IDs, untouched test data, per-device and per-motion error, paired cluster CI, and source-to-checkpoint regeneration for every reported cell.

## 8. Research-risk decision

A 2024–2026 survey can produce a visually impressive architecture taxonomy without reducing the error budget. The defensible order is: **correct evaluation semantics → establish baseline truth → identify tail failures → test a mechanistic correction → benchmark at matched information budget → claim novelty only from demonstrated residual improvements.** See [goals](05_goals.md), [limits](06_limitations.md), [future hypotheses](07_future_works.md).

## Second-pass risk reprioritization

1. **P0 — Independence and identity of test samples:** the 137 standard files include 18 duplicated physical Caruso streams; a reasonable deduplication makes the CI for the standard mean effect cross zero. Investigate source artifacts, not just a new architecture.
2. **P0 — Reference-assisted inputs:** EuRoC bias estimates from its reference-state file are subtracted from gyro before benchmarking. This is not an untouched raw-six-axis test and changes the deployment meaning of accuracy.
3. **P0 — Alignment and motion artifacts:** the [DIODEM 2025 source](https://doi.org/10.1038/s41597-025-05468-w) describes rigid/foam mounts, 11 continuous sessions, multiple frame transforms and broad-band nonrigid disturbances. The HDF5 test pipeline lacks enough conversion provenance to separate these causes.
4. **P1 — Validation objective mismatch:** the checkpoint is selected from independent reset validation windows using live weights, while the evaluation processes whole sequences with hidden-state carry and EMA weights by default.
5. **P1 — Domain-specific negative results:** mean paired TG-RIANN differences are +0.13016 degrees on EuRoC (zero wins in six sequences) and +0.07961 degrees on OxIOD (29.6% wins). An aggregate gain is not universal OOD improvement.

All numbers are from [released evaluation tables](../../results/full_evaluation/tables/per_dataset.csv) and independently checked CSV aggregation. See [verification details](09_second_pass_verification.md).
