# 04 — Research challenges, failure mechanisms and unresolved engineering problems

**Scope:** six-axis inertial tilt and related 2024–2026 research. Each entry separates the underlying failure mechanism, the specific research evidence, and a falsifiable test. See [math](03_mathematics.md), [limitations](06_limitations.md), and [full review](08_literature_review.md).

## 1. Fundamental observability and gauge ambiguity

**Mechanism.** Six-axis data supply gravity orientation only when acceleration of the reference point is negligible or inferable; gyroscope integrates relative angular motion but adds bias. Global yaw has a continuous SO(2) gauge. Under translational acceleration an identical accelerometer vector can be explained by different gravity directions and accelerations. No model architecture, transformer context or confidence calibration can universally identify arbitrary full pose from unassisted six-axis measurements.

**Evidence.** Contemporary direct methods [Wei 2025](https://doi.org/10.1016/j.measurement.2024.116097) model dynamic acceleration rather than claiming universal observability; [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) reports quaternion predictions on selected datasets but does not establish globally identifiable absolute heading. [DPI-LSTM](https://doi.org/10.3390/s26134287) estimates position with dataset-provided initial trajectory reference and short-duration assumptions, and should not be interpreted as proof of unassisted drift-free global position.

**Experiment.** Generate pairs of motions with the same six-axis observations but different a_W and tilt under a physical acceleration model; measure worst-case ambiguity as a lower bound. Repeat with a yaw-rotated world coordinate frame. **Acceptance:** the algorithm declares low information when constraints cannot distinguish cases; do not score an impossible output as necessarily identifiable.

## 2. Dynamic acceleration: translation and rotational lever arms

**Mechanism.** An accelerometer measures f=Rᵀ(a-g), so normalized acceleration points toward gravity only under restricted a. For rigid offset r, alpha×r and omega×(omega×r) appear even with zero reference-point translation. A confidence gate using only norm, jerk and angular rate cannot distinguish every acceleration causing an erroneous tilt, including constant-magnitude false-gravity cases. See the exact 30-degree counterexample in [mathematics](03_mathematics.md#8-a-constructive-non-identifiability-counterexample).

**Research.** [Brändle et al. 2025](https://arxiv.org/abs/2502.03681) show how angular acceleration changes the closed-loop transfer zeros of Mahony/Madgwick-like orientation correction; decreasing feedback can suppress interference at the expense of tracking bandwidth. [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784) explicitly calibrate and compensate the rotational motion terms when sensor-to-pivot geometry and calibration are known. Their [C++ source](https://github.com/fallow24/ros_imu_calib/tree/918a19236cecd6a367921e44ac73958ab9fbb9c8) contains calibration/residual optimization utilities based on Eigen/Ceres and static-interval methods; inspecting this code does **not** establish every paper result has been reproduced.

**Experiment.** Same angular motion around a rotation center with r=0, known nonzero r and intentionally miscalibrated r; plus translational acceleration pulses and long constant accelerations with nearly zero jerk. Report accepted gravity fraction, bias, peak tilt, reconvergence and p95 by regime.

## 3. Gyroscope bias, stochastic noise, drift and temperature

**Mechanism.** A gyro bias error delta b produces an integrated orientation error approximately proportional to time under suitable motions, compounded by noncommutative rotations and sampling artifacts. Bias and slow platform dynamics can be partially confounded. Aggressive bias augmentation can create training targets incompatible with physically plausible temporal drift.

**Research.** [Ye et al. 2025](https://doi.org/10.1002/rob.70082) train a sliding-window transformer gyro corrector with SO(3) integration loss, [Altawaitan et al. 2025](https://arxiv.org/abs/2505.06748) predict IMU bias as a nuisance state for an invariant visual-inertial filter, [Park et al. 2025](https://doi.org/10.46670/JSST.2025.34.2.130) compare bias sensitivity of rule-based and learned approaches, and [Park et al. 2026](https://doi.org/10.1109/ACCESS.2026.3691717) study recurrent bias augmentation. The VIO paper uses visual measurements and is not evidence of absolute heading recovery from an IMU alone.

**Experiment.** Device-held-out bias distribution, piecewise constant bias, random-walk bias, bias jumps, long rest/rotation intervals and temperature proxies. Train equal-budget implicit-GRU, explicit bias-filter and learned bias-residual heads. Report angular drift versus time as well as sequence tilt metrics.

## 4. Reference integrity: synchronization, coordinate frames and missing data

**Mechanism.** Microsecond/millisecond clock mismatch, body-to-world inversion, quaternion wxyz/xyzw mismatch, ENU/NED gravity-sign errors, incorrect IMU-to-mocap extrinsics, and SLERP across invalid truth gaps can dominate differences much larger than reported incremental neural gains.

**Evidence from target implementation.**
- [src/tg_dcm/so3.py](../../src/tg_dcm/so3.py) compresses finite reference entries in interpolate_quat before SLERP, allowing interpolation across original NaN gaps; the exact final valid timestamp is rejected by the present right-index test.
- [src/tg_dcm/prep/_common.py](../../src/tg_dcm/prep/_common.py) invokes that interpolation for dataset preparation.
- [src/tg_dcm/losses.py](../../src/tg_dcm/losses.py) clamps dt to at least 1e-6 before checking dynamic pair validity; zero and negative finite intervals can be treated as tiny positive steps.
- [src/tg_dcm/so3.py](../../src/tg_dcm/so3.py) also searches a target time offset using gyro/GT consistency; if selected with test truth, document that alignment as part of the evaluation rather than independent sensor inference.

**Experiment.** Synthetic identity quaternion streams with a missing middle sample, exact final endpoint, inverted gyro axes, shifted timestamps, time drift, split NaN runs and irregular dt. Assert behavior of each data adapter, then calculate changes in gate support and per-sequence error for repaired data. No model ranking should be considered final before this.

## 5. Dataset domain shift and unbalanced motion regimes

**Mechanism.** Device mounting, human/robot carrier, motion spectrum, frequency, scale, temperature, gyro bias, magnetometer absence and reference hardware vary by dataset. Average performance under one natural distribution is not a guarantee on heavy-tailed or rare rapid motion. Segmenting the same raw motion into many near-duplicates can bias aggregate means and bootstrap intervals.

**Research.** [KISS-IMU 2026](https://arxiv.org/html/2603.06205) explicitly clusters motions and balances training examples; [Tartan IMU 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) uses cross-platform pretraining/adaptation; [MosaicIMU 2026](https://arxiv.org/html/2606.09355) uses four carrier experts, but admits performance failures with unseen carriers, unfamiliar mount and severe bias; its quadruped ATE row can be worse than TLIO in the paper's table. Domain generalization is empirically conditional, not solved by “foundation model” naming.

**TG-DCM evidence.** The released standard split has 137 sequences with mean tilt RMSE 1.521° and paired mean -0.091° vs RIANN, 95% CI [-0.147,-0.036]. DIODEM is 422 segments from 86 source motions; mean 10.41°, median 2.70°, p95 44.20°, max 62.38°. Its grouped paired CI vs RIANN [-0.440,+0.043] crosses zero; no resolved mean superiority on that domain at 95%. This matters more than a small pooled gain.

**Experiment.** Leave-one-device, leave-one-carrier, raw-motion-level separation, high-angular-acceleration holdout, rare low-frequency carriers and independent recording sessions. Report mean and p95 by **original raw motion** and by dataset; never treat five segments from one raw DIODEM motion as five independent experiments.

## 6. Geometric consistency versus coordinate-system covariance

**Mechanism.** A learned model can be invariant to artificial rotations in a specific input format yet fail under a physical IMU remount with changed sensor axes, lever arm, bias and body-frame attitude. A vector network may be SO(3)-equivariant internally while an extra scalar, quaternion log or covariance layer violates the group law.

**Research.** [EqNIO](https://arxiv.org/html/2408.06321) designs under gravity-preserving O(2) transformations and explicitly considers axial gyro parity under reflections; [GINIO](https://arxiv.org/abs/2609.25338) builds vector-neuron SO(3) processing. The [GINIO author-side issues](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) identify a with-attitude Log(R) branch that is not equivariant under the physical remount law and incorrectly comparable table entries.

**Experiment.** Numerical Q in SO(3), compare f(Qx) to Qf(x); transform full attitude R->RQᵀ, biases and covariance, verify mean and covariance congruence to numerical tolerance. Repeat with a real remount and different lever arms and assess failure without pretending algebraic equivalence covers physical changes.

## 7. Uncertainty calibration and degenerate confidence gates

**Mechanism.** Networks trained on Gaussian NLL can increase predicted variance on hard cases without improving mean error. A confidence floor and hard mask can remove every difficult motion from the training loss. A soft confidence weight divided by gate population rather than total weight yields a changing effective regularization coefficient.

**Research.** [EqNIO](https://arxiv.org/html/2408.06321) transforms covariance through a learned frame; [GINIO](https://arxiv.org/abs/2609.25338) explores spectral covariance but has documented disagreements in reported NEES/NLL; [MosaicIMU's variance head](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) uses softplus of learned logits, yet confidence calibration still requires measurement. [KISS-IMU's covariance controller](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/covweight.py) adapts multiple uncertainty weights.

**Experiment.** Predictive interval coverage, NLL, NEES for the relevant two-dimensional tilt residual, calibration versus motion severity, gate support fraction, reliability diagrams and OOD selective-risk curves. Reject confidence models that improve accepted-only accuracy by excluding all hard frames.

## 8. Physics constraints can be self-inconsistent

**Mechanism.** A learned physics penalty is useful only when the model's state, measurement assumptions, and target units are consistent. Penalizing wrong-sign gravity or stale gyro motion can make predictions *more physically consistent with a wrong model* and less accurate.

**Research.** [DPI-LSTM 2026](https://doi.org/10.3390/s26134287) combines LSTM with translational and quaternion propagation penalties, but uses dataset-provided reference initialization. [MoRPI-PINN 2026](https://doi.org/10.1038/s41598-026-50630-y) constrains 2-D navigation under platform-specific motion. Their successes are conditional on assumed physical dynamics and state observability.

**Experiment.** Perturb dt, gravity sign, IMU-mount transform, reference timestamps and calibration; compare supervision-only, physics-only and joint loss with the same model and training budget. Report the sign and magnitude of per-term gradients and the fraction of valid physics pairs.

## 9. Online adaptation, distribution shift and inference latency

**Mechanism.** A cross-platform pretrained encoder may learn a carrier signature that misroutes novel hardware. Online fine-tuning requires supervision and may catastrophically forget earlier domains; a bidirectional architecture can appear realtime in throughput while requiring future frames.

**Research.** [Tartan IMU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) explores LoRA adaptation; the released Transformer builder is absent. [MosaicIMU](https://arxiv.org/html/2606.09355) adds a new expert and router adapter. Its actual velocity/variance heads are BiLSTM; its 500-sample/200-Hz training windows contain 2.5 s of context, but **true estimation delay depends on how the window is aligned**. [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) explicitly uses a future-sample budget.

**Experiment.** Equalize lookahead to 0, 10, 25, 50 and 100 ms and benchmark first-output latency, steady-state runtime, memory, filter warm-up and performance after state resets. Freeze a domain-adaptation rollback criterion before seeing test labels.

## 10. Experimental evidence and missing artifact integrity

**Mechanism.** Mislabelled training domain, missing checkpoint files, unreported checkpoint epoch, inconsistent metric definitions, reused dataset segments and code only partially released prevent paper claims from being reproduced. Statistical confidence intervals from one selected seed do not prove architectural attribution.

**Research counterexamples.** [GINIO KNOWN_ISSUES](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) disputes covariance NLL/NEES, LFA config, mixed trained/zero-shot baselines and checkpoint epoch inconsistencies. [TartanIMU Transformer code](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py) raises NotImplementedError, so it cannot reproduce a corresponding full architecture. [Ye 2025](https://doi.org/10.1002/rob.70082) abstract describes a 1.56° orientation result as better than a 1.49° Gyro-Net figure, an apparent numerical ordering error if lower degrees means better.

**Experiment.** For every table cell, emit immutable source SHA, checkpoint hash, train config, dataset manifest/sha, metric code, per-sequence table and exact command. Use multi-seed factorial ablations and raw-motion clustered paired bootstrap. Publish **failed** as well as successful hypotheses.

## 11. Challenge prioritization

| Priority | Blocker | Rationale |
|---|---|---|
| P0 | Invalid GT interpolation and dt propagation | Potentially contaminates all subsequent experiment interpretation |
| P0 | Common sensor/latency contract for new baselines | Prevents invalid comparisons |
| P1 | DIODEM heavy-tail and motion-domain failure | Demonstrated deployment-relevant weakness |
| P1 | Confidence collapse/non-gravity acceleration | Fundamental model misspecification |
| P1 | Robust gyro bias and frame remount | Dominant sensor/domain transfer issue |
| P2 | Calibrated tangent covariance and filter update | Enables principled runtime trust |
| P2 | SO(3) equivariant heads and multi-carrier routing | Higher complexity, value must be shown with held-out devices |
| P2 | Full reproducibility artifact mapping | Necessary for defensible public novelty claim |

**Status:** priority classification is analysis, not a completed experimental outcome. Work identified as source-level defect remains to be reproduced and repaired in a separate implementation/test phase.
