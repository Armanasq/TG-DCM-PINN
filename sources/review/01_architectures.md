# 01 — Architectures: 2024–2026 inertial attitude and navigation

**Target:** six-axis accelerometer + gyroscope + timestep to gravity-referenced tilt; **cutoff:** 2026-11-13. **Evidence:** primary works and pinned implementation excerpts; no external benchmark has been re-executed. See [mathematics](03_mathematics.md) and [complete review](08_literature_review.md).

## 1. Architecture taxonomy and comparison

The correct unit of comparison is an **information budget**: available sensor channels, observation window, initial state, reference frame, training teacher, allowed future frames and evaluation target. A learned velocity update in an EKF is not a direct tilt estimator, and a six-axis estimator cannot be evaluated on the same footing as a method receiving barometer, magnetometer, rotor speed, externally derived attitude or LiDAR teacher data.

| Family | Source | Architecture and actual target | Strength | Scope restriction |
|---|---|---|---|---|
| Classical geometric filter | [Wei 2025 RAESKF](https://doi.org/10.1016/j.measurement.2024.116097) | Nominal DCM + error-state EKF, acceleration compensation/covariance switching → tilt | Dynamic acceleration robustness, causal, interpretable | Publisher abstract; full implementation not inspected |
| Adaptive SO(3) observer | [Oveissi 2024 RCAE](https://arxiv.org/abs/2401.13092) | Multiplicative retrospective-cost orientation correction | Avoids MEKF correction Jacobian and explicit gyro-bias state | Receives **orientation measurements**; not raw-six-axis-only |
| Direct causal network | [TG-DCM-PINN](../../README.md) | GRU 2×200, seven inputs, four quaternion outputs, yaw-free tilt supervision | Compact streaming inference | Physics gates only in training, not inference |
| Direct windowed network | [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) | Axis-separated residual CNN + frequency encoding + BiLSTM, quaternion geodesic objective | Learns local and global temporal structure | Variable future lookahead, full method/code inaccessible |
| Learned gyro correction | [Ye 2025](https://doi.org/10.1002/rob.70082) | Sliding-window transformer + SO(3) preintegration objective | Bias/noise correction before integration | Orientation still requires integration; inconsistent comparison number in abstract |
| Neural pseudo-measurement | [DIVE 2024](https://doi.org/10.1109/LRA.2024.3370006), [AirIO 2025](https://doi.org/10.1109/LRA.2025.3581130) | IMU history → velocity/uncertainty → on-manifold filter | Combines motion prior and state dynamics | ATE/velocity is not tilt |
| O(2) canonicalization | [EqNIO 2024](https://arxiv.org/html/2408.06321) | Gravity-aligned IMU + equivariant yaw/reflection frame → displacement/covariance | Reduced orientation redundancy | Gravity estimate supplied by an EKF |
| Full SO(3) equivariance | [GINIO 2026](https://arxiv.org/abs/2609.25338) | Lie-/vector-neuron recurrent representations, covariance transport | Exact symmetry for defined tensor transform | Published artifacts have documented inconsistencies |
| Self-supervised correction | [KISS-IMU 2026](https://arxiv.org/html/2603.06205) | Learned inertial corrections + covariance, GMM motion balancing, PGO | Rare motion and label-scarcity handling | LiDAR ICP and PGO are used for training targets |
| Multi-carrier MoE | [MosaicIMU 2026](https://arxiv.org/html/2606.09355) | CNN encoder, prototype router, four experts, BiLSTM velocity/variance, MSCKF | Domain-conditioned motion features | Windowed, cross-domain failures remain |
| Adaptable foundation encoder | [Tartan IMU CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) | Shared ResNet-LSTM and carrier heads; paper studies LoRA adaptation | Transfer over robot types | Released Transformer builder is a placeholder |
| Additional-sensor observers | [Tchonkeu 2026](https://arxiv.org/abs/2607.13254), [Zhai 2026](https://doi.org/10.1016/j.measurement.2025.119824) | Barometric nonlinear SO(3) observer; learned covariance with angular accelerometer | Additional observability or higher-grade angular measurements | Not pure six-axis sensing |

**Historical antecedents:** Mahony, Madgwick, RIANN, VQF6D and [2023 six-axis CNN–BiLSTM attitude regression](https://doi.org/10.1016/j.measurement.2023.113105). Existing CNN–BiLSTM + quaternion prediction is prior art, irrespective of a 2026 publication date.

## 2. Direct architectural tradeoffs

**RAESKF.** Separates rotation on SO(3) from a small local perturbation. Error-state measurements retain interpretable innovation and covariance and can suppress accelerometer feedback under dynamics. Weakness: disturbance model and switching thresholds can fail under unfamiliar translation, shock or sensor location. Use matched 6-axis inputs to reproduce.

**TG-DCM-PINN.** The checked [train_impl.py](../../src/tg_dcm/train_impl.py) constructs a 2-layer 200-unit GRU; [losses.py](../../src/tg_dcm/losses.py) uses yaw-invariant supervised quaternion relative tilt, a norm regularizer, target-gated gyro dynamics, and norm/jerk/rate gravity confidence. There is no explicit filter state covariance at runtime. The full quaternion output is a representative of an unobservable yaw-equivalence class; the stated task is **tilt**, not absolute full attitude.

**Ding.** Independent axis-aware residual CNN kernels, a frequency branch and BiLSTM are plausible tools for dealing with variable sampling and dynamic motion. They introduce a larger context and hidden **lookahead**. The publisher states quaternion shortest-geodesic loss; before treating that as appropriate for raw-six-axis tilt, check whether reference yaw was aligned or regularized. Do not compare its median reported angle directly with TG-DCM's sequence tilt RMSE.

**Gyroscope denoisers.** Learned correction of measured angular rate preserves a physical integrator and makes gyro-bias tests explicit. [Ye 2025](https://doi.org/10.1002/rob.70082) provides a transformer + geometric-loss example. The longer context increases memory and latency; learned denoisers do not turn yaw into an observable absolute quantity.

## 3. Geometric and equivariant networks

[EqNIO](https://arxiv.org/html/2408.06321) canonicalizes a **gravity-aligned** inertial window with a learned horizontal frame F. The predicted displacement mean and covariance are mapped back by F and F Σ Fᵀ. Its O(2) reflection treatment requires angular velocity (an axial vector) to transform differently from acceleration (polar vector); the paper decomposes angular rates to construct compatible equivariant features. Because the gravity frame comes from the current filter estimate, it should not be imported into six-axis direct tilt with a hidden reference dependency.

[GINIO](https://arxiv.org/abs/2609.25338) extends symmetry to full SO(3) vector-valued features. In [airio/model/vn_gru.py](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/vn_gru.py), projected vector norms supply **invariant scalar GRU gates**, whereas the latent vector state transforms equivariantly. This is a hard architectural property rather than a penalty. The with-attitude log-map input violates the expected physical remount law under some conventions, as documented in the authors' [known issues](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md). Proving equivariance of a module is not sufficient for the composed estimator.

**Promising experiment:** a causal, calibrated S² tangent-correction network whose input vectors, filter state, bias and covariance all obey one stated sensor-coordinate transformation law. Its geometric components are not novel by themselves.

## 4. Neural priors integrated into filters

[DIVE](https://doi.org/10.1109/LRA.2024.3370006) regresses 3D velocity from gyroscope/accelerometer history, using the result as an EKF update when normal visual feedback is degraded. [AirIO](https://doi.org/10.1109/LRA.2025.3581130) retains body-frame dynamics, and the real [CodeNetMotionwithRot](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/model/code.py) consumes attitude features. Comparisons must use the attitude-free variant if six-axis is compulsory.

[KISS-IMU](https://arxiv.org/html/2603.06205) predicts IMU error corrections/uncertainty, propagates by preintegration, trains on LiDAR ICP/PGO pseudo-targets and balances rare motion types via GMM. Verified implementation evidence: [gmm.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/models/gmm.py) (GaussianMixture/motion weights) and [covweight.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/covweight.py) (bounded EMA weight controller). It establishes motion imbalance and uncertainty-control ideas, not a raw-IMU-only training protocol.

[MosaicIMU](https://arxiv.org/html/2606.09355) composes platform-specialist experts by a cosine-prototype router. The actual [moe_net.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) uses Gumbel softmax in its router, a shared sequence decoder, dilated convolution and **bidirectional LSTMs for both velocity and variance**. The filter in [msckf.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/filter/msckf.py) makes a cloned-state velocity update. [train_config.yaml](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/configs/train_config.yaml) uses a 500-sample sequence and nominal 200 Hz. This means 2.5 s of signal per window; the actual output delay depends on window positioning and the streaming policy, not solely on window length. The paper explicitly reports limitations under unseen carriers, mounting and severe bias changes, and requires supervision during early online adaptation.

## 5. Pretraining and temporal representation

[Tartan IMU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) evaluates pretraining, LoRA and deployment adaptation across devices. The inspected [LSTM builder](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/lstm.py) creates shared ResNet-LSTM trunk and dog/human/car/drone heads. The [Transformer builder](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py) raises NotImplementedError. Do not attribute an unreleased Transformer to an executable public benchmark. [iMoT 2025](https://doi.org/10.1609/aaai.v39i6.32664) uses progressive sequence decoupling, cross-modal positional encoding and query motion particles for velocity. [IONext](https://arxiv.org/abs/2507.17089) combines global/local feature mixing and temporal gating. Their relevance to tilt is a *controlled hypothesis*, not demonstrated accuracy transfer.

## 6. Architecture selection: prioritized falsifiable studies

| Rank | Change | Mechanism | Discriminating test |
|---|---|---|---|
| 1 | Preprocessing integrity + current GRU | Removes false label/timestamp transitions | Re-evaluate all test sequences; gate coverage and error tails |
| 2 | Causal gyro-bias head | Limits slow gyro residual drift | Injected constant/random-walk bias, temperature proxy, long motion |
| 3 | S² propagation + tangent update | Learns only observable tilt corrections | Same params/latency and yaw-gauge checks |
| 4 | Calibrated tangent covariance | Replaces heuristic trust with bounded uncertainty | NLL, NEES, empirical coverage and OOD |
| 5 | Lever-arm compensation | Recovers physical gravity in rotations | Known/perturbed r, sharp angular acceleration |
| 6 | Equivariant head | Reduces sensor-mount dependence | Algebraic SO(3) remount + real physical remount |
| 7 | Multi-carrier expert/temporal mixer | Handles different device dynamics | Leave-one-device, low-shot adaptation, router collapse |

Each candidate must outperform a matched-sensor, matched-lookahead comparator across 137 standard sequences and 422 DIODEM segments, with clustered confidence intervals and p95 guards. See [goals](05_goals.md), [limitations](06_limitations.md), [future work](07_future_works.md).
