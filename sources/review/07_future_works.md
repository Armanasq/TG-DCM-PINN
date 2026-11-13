# 07 — Future research: prioritized novel hypotheses and experimental programs

**Cutoff:** 2026-11-13. **Purpose:** scientifically defensible directions for improving TG-DCM-PINN, not claims of demonstrated novelty or achieved accuracy. Every direction specifies antecedent literature, hypothesis, mechanism, assumptions, experiment, falsification and integration cost. See [mathematics](03_mathematics.md), [goals](05_goals.md), and [limitations](06_limitations.md).

## 1. Research strategy and novelty standard

Novelty must be framed narrowly. A new arrangement of GRU + Charbonnier + gravity norm gate + gyro propagation is **not automatically novel**: geometric constraints, learned covariance, equivariant layers, adaptive filters and motion-balanced training all have prior work. A defensible contribution could be a **new identifiable correction formulation**, a proven **full-system equivariance guarantee** under a specified observation contract, a robust **uncertainty calibration mechanism**, or a statistically demonstrated failure-mode improvement from a carefully controlled combination. Novelty classification requires exact literature comparison, including the works in [full review](08_literature_review.md).

Strong candidate selection criteria: (i) correct observable state and sensor budget; (ii) measurable improvement on held-out motion groups, especially DIODEM tails; (iii) causal/bounded computation; (iv) clean mathematical assumptions and unit tests; (v) identifiable ablations distinguishing each component; (vi) unchanged baseline test artifacts for comparisons.

## 2. P0 — Repair ground-truth and timestamp integrity before developing new models

**Why:** a 0.091° standard mean advantage is small enough that label time alignment and quaternion interpolation deserve direct verification. [TG-DCM's so3.py](../../src/tg_dcm/so3.py) can interpolate across invalid reference gaps and excludes exact final valid timestamps; [losses.py](../../src/tg_dcm/losses.py) clamps invalid nonpositive dt before checking physical validity.

**Actionable implementation proposal:** 
1. Replace time interpolation with separate strictly monotonic finite segments, exact endpoint handling and a maximum source timestamp gap.
2. Generate explicit input_valid, gt_valid and adjacent_pair_valid masks before normalization or clamping.
3. Freeze the GT/gyro offset selection protocol and log reference clocks separately from test predictions.
4. Rerun all raw preparation, train/validation/test split manifests and released per-sequence evaluations.
5. Preserve old release checkpoint and old results as a non-recomputed comparison; report changed metrics as a new data-protocol version.

**Falsification:** if data hashes and all sampled outputs are unchanged after tests, the bug may be irrelevant to the present benchmark; retain the regression test regardless. Do not attribute an accuracy gain to a new estimator when only corrupted data have been repaired.

## 3. P1 — Physics-residual-conditioned gravity reliability with known rigid-body compensation

**Prior art:** [Wei RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097) adaptive acceleration handling; [Brändle 2025](https://arxiv.org/abs/2502.03681) angular-acceleration transfer-function analysis; [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784) mechanical calibration and compensation.

**Hypothesis:** for devices with a known lever arm, compensating alpha×r and omega×(omega×r) before gravity trust restores valid tilt observations that heuristic gates would discard, improving fast-rotation p95 without increasing false trust during pure translation.

**Proposed mechanism:** sensor calibration produces f_cal, omega_cal, alpha_est and lever arm r. Compute

    f_corr = f_cal - alpha_est x r - omega_cal x (omega_cal x r).

Then use a bounded robust residual between f_corr/||f_corr|| and propagated h^- to form an innovation gate; combine with rate/jerk but do **not** declare confident merely because ||f_corr||≈g. A learned heteroscedastic covariance may control the update gain. If lever-arm estimation is unavailable or unreliable, report fallback to the original baseline rather than silently applying a wrong compensator.

**Risk:** gyro differentiation amplifies noise, high angular acceleration and inaccurate mounting geometry create false corrections, and true platform translation remains unobserved. **Experiments:** r=0/known r/perturbed r/unknown r, controlled tilt+rotation and translation, thermal/bias injection, paired p95 and convergence time. **Falsification:** no advantage over simple covariance inflation at identical compute and calibration budget, or worsening of unknown-r conditions.

**Novelty boundary:** known compensation and robust confidence individually exist; a novel claim would require a new justified coupling/guarantee and demonstrated gains.

## 4. P1 — Bias-aware causal S² observer with a two-dimensional correction

**Prior art:** SO(3) observers, [Altawaitan et al. 2025](https://arxiv.org/abs/2505.06748) learned bias + invariant VIO, [Oveissi 2024](https://arxiv.org/abs/2401.13092) multiplicative adaptive correction, [EqNIO](https://arxiv.org/html/2408.06321) and [GINIO](https://arxiv.org/abs/2609.25338) geometry-aware features.

**Hypothesis:** by restricting the state to observable tilt h in S², the estimator can use a smaller causal network to predict correction in T_h S² and slow gyro bias, yielding better out-of-device transfer than direct 4-D quaternion regression with the same parameters and observation window.

**Proposed mechanism:**
- Propagate h^- = Exp_SO3(-[omega_m-bhat]x dt) h.
- Encode a causal IMU history into a bounded two-dimensional tangent correction xi and optionally a three-dimensional slowly varying bias increment.
- Update h^+ = Exp_h(B(h)xi), where B(h) is an orthonormal tangent basis.
- Avoid an arbitrary absolute yaw gauge in the output loss; if full quaternion API is required, maintain a relative-yaw representative using gyro propagation but score it only under a declared heading reference.
- Penalize innovation magnitude and prevent degenerate trust by explicit confidence coverage constraints.

**Assumptions:** calibrated gyro scale/axes, a sensible initial h, finite monotonic dt, sufficient time-varying excitation for bias separation; a bias estimate cannot be arbitrarily identified under all motion/acceleration patterns.

**Tests:** S² norm preservation, yaw-gauge invariance, antipodal and near-pole numerical stability, random frame Q laws, long static versus prolonged translation, sensor bias jumps, 5+ seeds on equal-parameter GRU. **Falsification:** no OOD p95 gain, failed bias identifiability, or extra update cost without improved calibration.

**Open theoretical problem:** sufficient excitation/acceleration assumptions for locally observable bias and tilt correction in the proposed mixed learned/geometric architecture.

## 5. P1 — Calibrated and transportable uncertainty in tangent space

**Prior art:** [EqNIO](https://arxiv.org/html/2408.06321) canonical-frame covariance; [GINIO](https://arxiv.org/abs/2609.25338) spectral SPD covariance; [KISS-IMU](https://arxiv.org/html/2603.06205) learned inertial covariance and weighted optimization; [MosaicIMU](https://arxiv.org/html/2606.09355) variance heads.

**Hypothesis:** a bounded 2×2 SPD covariance for tangent correction improves selective tilt reliability and fallback decisions compared with a fixed norm/jerk/rate heuristic, with equal expected error.

**Proposed design:** output Cholesky parameters L with strictly positive diagonal (softplus + epsilon) and Sigma=LLᵀ. Train on tangent logarithm residual with NLL plus a supervised tilt guardrail and an explicit covariance-scale regularizer. Transport covariance between tangent bases by their local rotation. Use a calibration-only split for covariance post-calibration, then evaluate untouched held-out sequences.

**Metrics:** NLL, NEES (target expectation approximately 2 when model is calibrated), empirical 50/90/95% coverage, mean tilt, selective-risk curves at fixed coverage, p95 under OOD. **Falsification:** covariance broadening without improved coverage or mean error, collapse to near-zero, non-SPD numerical values, or missed exact coordinate transformation law.

**Critical method caveat:** uncertainty learned under one domain is not a reliable out-of-distribution probability without tests. A smaller NLL on training examples is not evidence of calibrated test confidence.

## 6. P1 — Decompose measurement quality from reference-label quality

**Prior art:** target consistency gating already exists in [TG-DCM](../../src/tg_dcm/losses.py); robust filtering and confidence heads appear in [Wei](https://doi.org/10.1016/j.measurement.2024.116097), [KISS](https://arxiv.org/html/2603.06205), [MosaicIMU](https://arxiv.org/html/2606.09355).

**Hypothesis:** decoupling (a) invalid reference transitions, (b) physical gyro integration reliability, and (c) gravity measurement trust prevents gate-induced selection bias and makes physics losses more useful on difficult motion.

**Mechanism:** use independently generated masks for numerical validity, source GT integrity, intersample timestamp quality, gyro saturation/bias, and model/measurement innovation. Preserve the main supervised loss on all truly valid reference frames even when the dynamics loss is skipped. Track coverage and a “hard but valid” motion stratum. Optionally replace a fixed 2° gate with a robust continuous residual calibration selected on validation only.

**Tests:** group accepted/rejected pairs by raw motion, angular acceleration, dataset, device, and GT clock offset; evaluate gate-on/off, hard vs soft consistency and target time-warp. **Falsification:** any proposed soft gate improves total loss only by rejecting large-error frames while unconditional per-sequence tilt worsens.

## 7. P2 — Strict remount equivariance and symmetry-aware data augmentation

**Prior art:** [EqNIO](https://arxiv.org/html/2408.06321) gravity-preserving O(2) canonicalization and reflection parity; [GINIO](https://arxiv.org/abs/2609.25338) SO(3) equivariant layers. GINIO's [published artifact audit](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) identifies how a with-attitude Log(R) channel breaks the physical remount contract.

**Hypothesis:** exact full-estimator SO(3) covariance for the **coordinate-change** law reduces need for synthetic rotation augmentation and improves held-out sensor mounts without harming ordinary tilt.

**Design:** define a group action on all raw channels, bias state, gravity, model outputs, hidden vector features and covariance; either use strictly equivariant vector features or a canonical-frame method with proven reconstruction. Do not test with input vectors rotated while leaving attitude or bias untransformed. Distinguish mathematical coordinate remount and physically moving the sensor (which changes r, noise and offset).

**Tests:** exact numerical equivariance residual under 1000 random rotations at zero and nonzero bias, end-to-end stateful sequence tests, real remount stress, reflections with det(Q) parity when using O(2). Compare matched augmentation baseline. **Falsification:** increased calibration sensitivity, failed full model group law or no OOD benefit.

## 8. P2 — Motion-aware sampling and representation without test leakage

**Prior art:** [KISS-IMU](https://arxiv.org/html/2603.06205) motion-GMM reweighting and [MosaicIMU](https://arxiv.org/html/2606.09355) carrier-conditioned experts; [Tartan IMU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) cross-platform pretraining; [iMoT](https://doi.org/10.1609/aaai.v39i6.32664) temporal motion particles; [IONext](https://arxiv.org/abs/2507.17089) local/global feature mixing.

**Hypothesis:** an imbalance-aware training sampler or small causal multirate encoder improves DIODEM high-error modes more efficiently than a larger BiLSTM or four-carrier MoE.

**Design:** cluster or weight training motions using **only training-set** observables (velocity-free: angular rate, acceleration magnitude, jerk and spectrum); enforce raw-motion-group partition; sample rare motions without overfitting invalid GT. Compare shared GRU, causal TCN, lightweight multi-rate state and fixed expert models under identical input lookahead.

**Tests:** leave-one-device/carrier, rare high-angular-acceleration, static, rapid direction change; monitor p95 and gating support. **Falsification:** no improvement after controlling for extra compute/training examples or negative effect in the most difficult motion group.

## 9. P2 — Reliable self-supervision with clear auxiliary modalities

**Prior art:** [KISS](https://arxiv.org/html/2603.06205) relies on LiDAR ICP/PGO teacher; [SSIO 2026](https://arxiv.org/abs/2610.25988) uses IMU-preintegration teacher and pose-graph distillation; [MoRPI-PINN](https://doi.org/10.1038/s41598-026-50630-y) uses physics constraints under a locomotion model.

**Hypothesis:** short-horizon IMU-consistency self-supervision may improve robustness where GT orientation is sparse, provided its target cannot simply reward gyro integration bias.

**Design:** use physics-consistent short-window differential tilt only where target validity is independently established or a calibration/teacher observation exists. Model teacher uncertainty. Separate “no manual labels”, “no external sensors at inference”, “no external sensors at training”, and “unsupervised adaptation”; these are different research contracts.

**Tests:** vary teacher errors, pseudo-label confidence and motions, then evaluate against independently recorded GT. **Falsification:** student just inherits teacher drift, loses calibration, or performs worse than supervised baseline at equal data budget.

## 10. P3 — Publication-grade experimental artifact and falsification package

**Critical dependency:** most published source-to-number disputes arise from checkpoint/metric mismatch rather than an intrinsically wrong layer. Every experiment should produce:
- immutable commit, dataset SHA/manifest, calibration and time transform, partition definition;
- runtime Python/PyTorch versions, full config, seed, checkpoint SHA256, code/dependency lock;
- train/validation/test per-sequence output and raw-motion ID with masks and failure flags;
- exact per-sequence statistics, paired effects, confidence interval, p95, throughput;
- a machine-generated manuscript table row sourced from one configuration and checkpoint;
- explicit negative results and failed assumptions.

Use the [GINIO author-side discrepancies](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) as concrete anti-patterns: mixed checkpoint epochs, mislabeled pretrained domains, unsupported NEES, absent table regenerators and ambiguous remount protocol.

## 11. Suggested decision sequence

| Milestone | Deliverable | Go/no-go condition |
|---|---|---|
| M0 data integrity | Rebuilt data manifest, gap/time validity tests, documented clock offsets | Every target frame traceable to unbroken finite GT source |
| M1 credible baselines | Released checkpoint re-run, RIANN, VQF6D, adaptive ESKF, CNN–BiLSTM at fixed latency | Same sensors, targets and complete per-sequence outputs |
| M2 gate attribution | Factorial gyro/gravity loss, 5+ seeds, coverage and hard-motion stratification | Reproducible paired gain that survives tail guardrails |
| M3 geometric observer | S² tangent update, bias state and covariance calibration | Correct manifold tests, better held-out p95/coverage |
| M4 physical compensation | Calibrated lever arm, measured alpha, robust fallback | Improvement in fast-rotation tests without translation regression |
| M5 group/cross-device transfer | SO(3) stateful remount invariance, optional multi-carrier learning | Exact transformation law + real OOD performance |
| M6 synthesis | Final novel claim, counterexamples, complete source-to-table audit | All reported numbers independently reconstructible |

**Non-negotiable:** if a candidate fails a controlled test, publish the limitation. Research integrity matters more than adding another named algorithm.
