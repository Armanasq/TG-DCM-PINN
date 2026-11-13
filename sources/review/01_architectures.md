# 01 · Architectures — what each estimator actually learns

**Question:** Which architecture introduces information, which merely changes the inductive bias, and which assumes information the six-axis sensor does not provide? **Cutoff:** 2026-11-13. **Evidence:** [full comparison](08_literature_review.md), [paper registry](../papers/index.json), [pinned code](../repositories/index.json). Reported accuracy is author-reported unless a reproduction is explicitly stated.

## 1. Classify the estimator by its correction mechanism, not its backbone

| Mechanism | Representative work | State learned or constrained | Hidden dependency | Relevant falsification |
|---|---|---|---|---|
| Direct orientation regression | [TG-DCM](../../src/tg_dcm/train_impl.py), [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) | Quaternion representative from a raw IMU window | Training distribution supplies a gravity/motion prior; BiLSTM may use future data | Same-sensor, same-lookahead tilt error |
| Physical propagation + adaptive accelerometer update | [RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097) | Nominal DCM and local error/covariance | Dynamics model and choice of valid gravity observation | Sustained acceleration with \(\|f\|=g\) |
| Learned bias dynamics + integration | [Liu et al., RSS 2025](https://arxiv.org/html/2504.09495), [TransIMU 2025](https://doi.org/10.1002/rob.70082) | Corrected rate or explicit time-varying bias, then integrated SO(3) | Initial bias/pose and pose-label derivatives; residual noise remains | Bias steps and long-horizon holdout |
| Learned pseudo-measurement + geometric filter | [DIVE 2024](https://doi.org/10.1109/LRA.2024.3370006), [AirIO 2025](https://doi.org/10.1109/LRA.2025.3581130) | Velocity and covariance used in EKF update | Platform motion prior; some variants consume attitude | Out-of-carrier velocity innovation consistency |
| Geometrically equivariant motion prior | [EqNIO 2024](https://arxiv.org/html/2408.06321), [GINIO 2026](https://arxiv.org/abs/2609.25338) | Vector mean and SPD covariance with group action | EqNIO needs gravity alignment; extra attitude channels may violate physical remount | Full-input, full-state remount commutation |
| Rare-motion or carrier-specific mixture | [KISS-IMU 2026](https://arxiv.org/html/2603.06205), [MosaicIMU 2026](https://arxiv.org/html/2606.09355) | Mode-weighted motion correction or prototype-routed experts | LiDAR-derived training labels for KISS; carrier specialization for Mosaic | New carrier / unseen motion / collapsed router |
| Physics-informed joint navigation | [DPI-LSTM 2026](https://doi.org/10.3390/s26134287), [MoRPI-PINN 2026](https://doi.org/10.1038/s41598-026-50630-y) | Rotational/translational state consistency | Initialized motion model, platform priors or short trajectories | Long unassisted trajectory with wrong motion prior |

**Interpretation:** a larger backbone cannot resolve a rank-deficient observation model. A covariance head quantifies predicted uncertainty; it does not supply a missing observation. A group-equivariant network handles coordinate transformations; it does not automatically handle a moved sensor with different lever arm or bias.

## 2. Direct tilt architectures: the decisive distinction

**Released TG-DCM.** [train_impl.py](../../src/tg_dcm/train_impl.py) uses a 2-layer, 200-hidden-unit GRU and a 4D quaternion head. The loss supervises *tilt* and applies gyro/gravity consistency **during training**. At inference the model emits a quaternion without a physical update, a measurement innovation, or calibrated state covariance. Its heading is a gauge representative, not a measured absolute yaw.

**[Ding et al. 2026](https://doi.org/10.1016/j.measurement.2026.122485).** Axis-separated residual convolutions encode local rate/force structure; a frequency branch addresses varying sampling rate; a BiLSTM uses both directions of its supplied window; a shortest-quaternion-geodesic objective fits the reference attitude. This may improve responsiveness at bounded delay, but it also changes the available temporal information. Its median angular error cannot be ranked against sequence-mean *tilt RMSE*. The publisher exposes architecture discussion; a complete independently runnable implementation was not found in this review. Full-SO(3) supervision additionally requires a declared initial-heading/reference convention for a six-axis estimator.

**[Wei et al. RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097).** Unlike direct regression, an error-state filter distinguishes gyro **process uncertainty** from accelerometer **observation uncertainty** and can increase the latter under dynamic motion. The unverified assumption is that disturbance detection remains valid when translational acceleration mimics gravity magnitude. Its published dynamic gains are scenario-specific, not matched TG-DCM evidence.

**Non-obvious difference:** compare an ESKF, a causal GRU and a bounded-lookahead CNN–BiLSTM only after specifying (i) the same initial tilt information, (ii) the same 6 IMU axes, (iii) the same reference-frame conventions, (iv) the same delay budget, and (v) the same per-sequence tilt statistic. Otherwise the experiment measures information access and protocol choices as much as algorithm quality.

## 3. Explicit bias models: the strongest architecture-level omission

[Liu et al., RSS 2025](https://arxiv.org/html/2504.09495) learn continuous gyro and accelerometer bias **dynamics**, not a direct quaternion or generic “denoised IMU” signal. Their controlled Neural ODE uses a causal backward-difference cubic-Hermite interpolation, integrates rotation on a Lie-group chart, trains the gyro bias first from orientation error, then freezes it and trains accelerometer bias from position/velocity error. Ground-truth **bias** labels are unnecessary; **pose** truth and trajectory-derived initial bias approximations still supervise training, and initial conditions matter. The paper explicitly treats remaining stochastic measurement noise as outside the core debiasing model. [Source code](https://github.com/UMich-CURLY/Debias_IMU/tree/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy) implements separate bias networks, spline and Lie integration.

This is a sharper alternative to indiscriminately increasing model capacity: distinguish (a) slowly varying systematic bias, (b) stochastic measurement noise, and (c) carrier motion. Bias correction can be separately tested and integrated with any attitude observer. It is **not** a claim that bias is identifiable from arbitrary motion without known initialization or sufficient excitation.

A second [2026 empirical preprint](https://doi.org/10.31224/7144) reports 2,811 gyroscope-bias experiments over EuRoC and TUM-VI. In its variance decomposition dataset choice explains approximately 97% and model family under 1%; an SO(3) auxiliary term brought no improvement in its tested configurations. **Interpretation:** these percentages describe its two-dataset bias experiment, not all tilt research. Their useful insight is to prioritize reference-label quality, grouped splits and clean nuisance-state targets before CNN/Transformer architecture search.

## 4. Symmetry by design: differentiate coordinate change from physical remount

[EqNIO](https://arxiv.org/html/2408.06321) canonicalizes a **gravity-aligned** window under an O(2) subgroup (yaw and reflections), predicts displacement/covariance and transports the result out of the canonical frame. It handles gyroscope angular rate as an **axial** vector under reflections; applying the same O(2) transform to acceleration and angular rate is wrong. Its gravity direction comes from an estimated filter attitude—an important assumption if applied to a tilt estimator, where gravity is the unknown.

[GINIO](https://arxiv.org/abs/2609.25338) uses SO(3)-equivariant vector-neuron and related layers. Its [VNGRUCell](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/vn_gru.py) computes **invariant scalar gates from projected vector norms**, then updates equivariant vector hidden states. But [known issues](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) describe a with-attitude log-map channel that does not satisfy the physical-remount transformation, as well as mislabelled checkpoint/evidence rows. **Insight:** an equivariant layer is not an equivariant *estimation system* unless the input map, recurrent state, correction, covariances and runtime attitude all transform consistently.

## 5. Mixtures, uncertainty, and platform priors

[AirIO](https://doi.org/10.1109/LRA.2025.3581130) retains body-frame motion context; its [implementation](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/model/code.py) contains separate forward variants with and without attitude input. Check that the evaluation does not silently use the attitude-assisted variant.

[KISS-IMU](https://arxiv.org/html/2603.06205) learns corrected inertial increments and uncertainty from **LiDAR ICP/pose-graph pseudo-supervision** and uses a motion GMM to reduce frequent-motion dominance. [KISS GMM](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/models/gmm.py) is real code, not evidence of IMU-only training labels.

[MosaicIMU](https://arxiv.org/html/2606.09355) uses four carrier-conditioned experts with a learned cosine-prototype router; its [mean and variance heads](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) both contain **bidirectional** LSTMs, and the [train config](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/configs/train_config.yaml) uses 500 samples at a nominal 200 Hz. A 2.5-second observation window does not prove a 2.5-second output delay, but it invalidates an unqualified zero-future-context comparison. The paper reports carrier-specific failures despite favorable average ATE.

[TartanIMU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) demonstrates cross-platform shared representations. The public [Transformer constructor](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py) raises NotImplementedError; the released LSTM path is distinct from paper-level future architecture claims.

## 6. Architecture decision under a constrained experiment budget

1. **Prioritize physical/label integrity, not model family.** It can invalidate every reported delta; see [source audit](../implementation/target-audit.md).
2. **Establish a same-input adaptive ESKF and matched-delay six-axis CNN–BiLSTM.** They test whether the present GRU is genuinely competitive.
3. **Explicit gyro bias state before MoE.** RSS 2025 supplies a concrete learned-bias counterpoint; a small causal bias head is a lower-complexity ablation than four platform experts.
4. **S² tilt correction and bounded uncertainty.** Its benefit is a *testable observable-state representation*, not the invention of Lie-group filtering.
5. **Conditional rotational-acceleration compensation.** Only for measured/identified lever arms; arbitrary translation remains unresolved.
6. **End-to-end coordinate-equivariance tests.** Add only after a strong augmentation baseline and exact SO(3) property tests; physical remount is a separate OOD test.

**Stop criterion:** if a candidate's gain vanishes when delay, truth alignment, sensor channels and data splitting are equalized—or if DIODEM p95 regresses—report a trade-off or null result, not a new best method.

## Second-pass architecture attribution and geometric prior art

**The RIANN ancestry matters.** [Weber's RIANN training reimplementation](https://github.com/daniel-om-weber/riann_dev/tree/435274141c08ab318c01be904de9754943b4feed) has the same broad 2-layer GRU, Myon/TUM-VI split, bias/noise augmentation, rate resampling and Ranger design. It is not the originally released RIANN training run—the author explicitly labels it a reimplementation. It substantially narrows the claim of novelty: TG-DCM must isolate what **target-gated loss** changes beyond a matched re-trained RIANN, not claim the GRU, training policy or split.

**The S² gravity state is also prior art.** [Bouazza et al. 2025](https://arxiv.org/abs/2508.21163), [Abi et al. 2026](https://arxiv.org/abs/2609.26612) and [Boughellaba et al. 2026](https://arxiv.org/abs/2601.07156) construct geometric observers with body gravity and velocity. Their strong convergence conditions depend on visual or optical-flow measurements; do not transfer those guarantees to an unassisted six-axis estimator. A future contribution must be in its measurable update, uncertainty/error model, an additional proven property or held-out performance—not the gravity-state representation alone.

**Correct direct baseline priority:** official RIANN ONNX plus independently reproduced RIANN training reimplementation, six-axis ESKF, VQF without future backfill, and TG-DCM at the **same** data, initialization and delay. See [second-pass findings](09_second_pass_verification.md).
