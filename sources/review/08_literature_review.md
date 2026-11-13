# 08 — Comprehensive critical literature review (2024–2026)

## Deep learning, geometric filtering, physical constraints, and uncertainty for six-axis inertial attitude estimation

**Review frozen:** 13 November 2026. **Reference system:** [TG-DCM-PINN](../../README.md), target 6-axis IMU to yaw-invariant tilt. **Coverage:** 2024–2026 primary literature supplemented by clearly labeled pre-2024 antecedents. **Research standard:** distinguish (i) paper claims, (ii) readable method and equations, (iii) inspected implementation, and (iv) independently reproduced experiments. This research has inspected primary web/HTML text and selected code from pinned GitHub revisions; it has not reproduced experimental metrics. Bibliographic metadata and narrower paper notes are in [../papers/index.json](../papers/index.json).

### Abstract

The dominant difficulty in six-axis attitude estimation is not choosing a neural backbone but respecting the information supplied by gravity, angular velocity, and platform motion. Global yaw is structurally ambiguous without external reference, while the gravity direction becomes conditionally observable only under restrictions on translational and rotational accelerations. Deep learning can exploit motion statistics and bias patterns but does not remove these identifiability limits. Recent research divides into (1) adaptive geometric filters compensating or down-weighting accelerometer disturbances; (2) direct CNN/recurrent quaternion regressors; (3) learned gyro-error compensation and physics-constrained objectives; (4) hybrid inertial odometry with learned pseudo-measurements and uncertainty; (5) equivariant architectures with correct group actions; and (6) multi-platform representation learning and adaptation. The strongest practical opportunities for TG-DCM are improved source data integrity, gate-coverage/uncertainty calibration, mechanism-aware rotational acceleration compensation, a bias-conditioned tangent-space tilt observer, and controlled motion-domain transfer. The evidence does **not** yet establish that an additional physics loss or geometric layer is necessarily a unique innovation or improves high-error tails.

## 1. Review framing, search and comparison rules

This synthesis evaluates research first disseminated or published in 2024–2026, through the review cutoff. Prior models (Mahony/Madgwick filters, VQF, RIANN, INS preintegration, 2023 CNN–BiLSTM attitude estimation and AirIMU) are *historical antecedents* and not counted as new 2024+ papers. Selection favored primary task relevance, formal geometric contributions, reproducible source code and challenging generalization regimes. Primary discovery used arXiv/structured HTML, scholarly indexes, publisher research pages and GitHub source-level inspection. Where a publisher article is paywalled, claims are labeled abstract/preview-level and no inaccessible internal algorithm is invented. This is a **curated critical technical survey**, not an exhaustive PRISMA systematic review.

The primary target is yaw-invariant tilt: the geodesic angle between world vertical directions expressed in the body frame. Many inertial odometry papers report translation ATE, relative translation error, velocity RMSE or navigation position RMSE; these are useful for transferable architectures and learned uncertainty but cannot be directly ranked against TG-DCM's sequence-level tilt RMSE. Inference modalities and causal delay must also match: a LiDAR-based teacher available only during training, a live rotor-speed input or a bidirectional future window changes what the network can know.

**Evidence labels:** [P] readable primary paper methodology; [A] abstract/publisher preview only; [C] inspected source module at exact Git SHA; [E] evaluation independently executed. No external work in the current corpus qualifies for [E]. Some mathematical counterexamples are newly derived and verified independently of the claimed experiments.

## 2. Physical problem and scientific assumptions

Let R_WB(t) map vectors from the inertial sensor body frame into the world frame. The gyroscope measures angular velocity plus bias and noise; the accelerometer measures body-frame specific force \(R^T(a_W-g_W)\) plus bias and noise. If the sensor is displaced by a lever arm r from the platform reference point, angular acceleration and centripetal terms enter. Near-stationary accelerometer direction can therefore reveal gravity, whereas dynamic motion confounds gravity with external acceleration.

Define h(t)=R_WB(t)^T e_z. This unit vector is the observable gravity/up direction when the accelerometer reading is reliable. For constant world yaw Y=R_z(phi), YR has the same h and body gyro observations as R. Absolute heading is therefore a gauge, while tilt is represented on S². The direct 6-axis task should use h or a yaw-free quaternion relative tilt metric; external yaw reference cannot be learned in a universally identifiable sense. A deep model can learn useful distributional priors for yaw, but priors are not new physical observations.

The angle between predicted h and true h is

    theta = atan2(||h_pred x h_true||, h_pred dot h_true).

In the released TG-DCM code, using qerr=qhat tensor conjugate(q_true), this is implemented as 2 atan2(sqrt(xerr²+yerr²),sqrt(werr²+zerr²)). Propagation by measured gyro follows R_{k+1}=R_k Exp_SO3([omega]x dt), or h_{k+1}=Exp_SO3(-[omega]x dt)h_k. See [03 mathematics](03_mathematics.md) for exact derivations, coordinate actions and covariance.

**Nonidentifiability counterexample:** a constant accelerometer vector tilted 30° from true gravity but with norm exactly 9.81 m/s² gives a gravity-confidence magnitude factor 1. It also produces zero jerk and zero angular speed when sustained at fixed orientation, so all three current heuristic gate factors are maximal, even though true gravity is 30° away. This is an analytical limit on the gate information, not a measured model failure on any particular dataset.

## 3. 2024 research: the shift from regression to geometry and adaptive observers

### 3.1 Retrospective-cost adaptive SO(3) observer

[Oveissi, Mirtaba and Goel, 2024 — Learning-based Attitude Estimation with Noisy Measurements and Unknown Gyro Bias](https://arxiv.org/abs/2401.13092) [P].

**Goal:** improve estimation when angular-rate measurements have unknown bias and direct orientation measurements are noisy. **Method:** propagate attitude on SO(3) using gyro increments, then apply a multiplicative retrospective-cost adaptive correction rather than a conventional additive quaternion update. The adaptive scheme avoids deriving the explicit MEKF correction Jacobian and does not require an explicit gyro-bias state. **Mathematical role:** the correction lives in rotation space and composes with the nominal attitude; it is not a direct four-component unconstrained quaternion fit.

**Why it matters:** the geometry protects rotation validity while data-adaptive correction can adjust to error without a fixed, fully known noise process. **Critical limitation:** the estimator consumes noisy *attitude observations*, not merely a six-axis IMU. It therefore cannot be ranked as a strict same-input competitor to TG-DCM. The exact performance on raw six-axis tilt is unmeasured in this review. **Transferable design:** multiplicative/tangent correction and adaptation, provided a credible gravity-only observation model is derived.

### 3.2 DIVE: neural velocity as a virtual inertial measurement

[Bajwa, Cossette, Shalaby and Forbes, 2024 — DIVE](https://doi.org/10.1109/LRA.2024.3370006) [P/A].

**Goal:** retain navigation estimates during visual failure or without external velocity while pure integration drifts. **Architecture:** use an IMU-history neural predictor to produce 3-D velocity and insert that prediction into an on-manifold EKF. **Research contribution:** learned motion priors act as an actual filter measurement rather than replacing physical propagation entirely. The reported in-distribution/OOD/navigation-outage gains are trajectory/localization results.

**Assumptions and limits:** the learned velocity prior depends on platform dynamics and training distribution; filter covariance must reflect virtual measurement error. Its velocity or position ATE cannot establish improved tilt. **Transfer:** a small learned gravity-innovation head inside a physical tilt observer is more relevant than copying the displacement network.

### 3.3 EqNIO: subgroup symmetry and learned canonicalization

[Jayanth et al., 2024 — EqNIO: Subequivariant Neural Inertial Odometry](https://arxiv.org/html/2408.06321) [P+C], with inspected [implementation](https://github.com/RoyinaJayanth/EqNIO/tree/97b7a60a5e5cc8f0132bba343c83de114211f827).

**Problem:** inertial odometry networks may memorize redundant motion examples at every yaw alignment instead of learning an equivariant displacement prior. **Method:** given gravity-aligned IMU vectors, an O(2)-aware network estimates a canonical horizontal frame from invariant scalars and equivariant vectors. It rotates the input into the frame, uses an ordinary odometry network to predict displacement/covariance, then transforms mean and covariance back. Under transformation Q preserving gravity,

    m(QX)=Qm(X),    Sigma(QX)=Q Sigma(X) Q^T.

Reflections require special care because acceleration is polar and angular velocity is axial: \(\omega'\) transforms with det(Q) parity. This is explicitly addressed in EqNIO's feature construction; pretending all six axes transform as identical vectors under reflection would be wrong.

**Results:** authors demonstrate changes in TLIO/RoNIN-style odometry performance across TLIO, Aria, RIDI and OxIOD. **Important dependency:** EqNIO's TLIO gravity alignment uses the current filter's attitude estimate. The framework cannot be directly inserted into a standalone tilt estimator as if the gravity direction were already known. The direct output is displacement/covariance, so there is no published TG-style tilt RMSE equivalence. **Opportunity:** an exactly covariant tilt correction, tested at both frame and hardware-remount levels.

## 4. 2025 research: direct networks, filter compensation, bias and cross-platform training

### 4.1 Adaptive dynamic-acceleration error-state filtering

[Wei et al., 2025 — A robust adaptive error state Kalman filter for MEMS IMU attitude estimation under dynamic acceleration](https://doi.org/10.1016/j.measurement.2024.116097) [A].

**Goal:** improve roll/pitch accuracy when measured acceleration contains motion rather than gravity alone. **Method:** nominal direction-cosine matrix and local error-state Kalman filter; select between compensated acceleration and increased measurement uncertainty using motion-state thresholds. **Rationale:** adapt the measurement model and gain instead of trusting all accelerometer samples equally. The publisher summary reports pitch/roll gains in selected dynamic experiments, including at least 42.76% pitch and 67.21% roll for a cited dynamic setting. These are author-specific relative comparisons and not broadly generalizable percentages.

**Limitations:** threshold tuning, unknown translation, gravity/accel ambiguities and full paper/implementation availability. **Why it matters:** among newer direct task peers, a reliable same-six-axis adaptive ESKF is a critical baseline missing from the released TG-DCM table. **Experiment:** implement with explicit calibration, identical initial states, rate and tilt metric, then compare robustly under known external acceleration.

### 4.2 Angular acceleration as a closed-loop dynamics problem

[Brändle et al., 2025 — On the Effects of Angular Acceleration in Orientation Estimation Using IMUs](https://arxiv.org/abs/2502.03681) [P].

**Question:** why do complementary attitude filters fail even when the attitude dynamics are correctly integrated from a gyro? **Derivation:** for an offset accelerometer, the measured force includes tangential alpha×r and centripetal omega×(omega×r) terms. These enter the correction feedback, affecting its transfer-function zeros. In certain operating regimes the linearized system can be nonminimum phase; making gravity feedback weaker filters out these effects but reduces the response bandwidth. This formalizes a real tradeoff that is easy to hide in a neural gravity gate.

**Limitations:** linearization, known platform/sensor geometry and dynamic-regime dependence. **Importance to TG-DCM:** the gravity confidence multiplier uses rate/jerk and downweights fast motion, which may protect some samples but can discard recoverable physics and cannot detect all constant acceleration errors. **Hypothesis:** compensate known rigid-body terms before confidence gating and report both worst-case tilt and recovery time.

### 4.3 Gyroscope compensation with attention and geometric integration

[Ye et al., 2025 — A Robust Transformer-Based Error Compensation Method for Gyroscope of IMUs](https://doi.org/10.1002/rob.70082) [P/preview].

**Motivation:** time-correlated gyro errors and biases accumulate during open-loop orientation integration. **Design:** learn a corrected angular rate from a history window using a reported six-layer transformer encoder/decoder, then integrate the corrected motion and train with an SO(3)-aware preintegration objective. The architecture separates a nuisance-sensor-correction problem from the physical rotation update, unlike direct orientation regression.

**Research caveat:** the published abstract says the proposed orientation error is 1.56° versus Gyro-Net 1.49° and describes this as an improvement. Assuming lower angle means better and the two numbers are the same metric, that ordering is inconsistent. Without the underlying per-sequence records the correct comparison must remain unresolved. **Transfer:** compare gyro-bias correction explicitly against implicit GRU memory, with bias/time stress; retain prior-art awareness that a learned gyro denoiser does not identify absolute yaw.

### 4.4 Invariant visual-inertial bias learning

[Altawaitan et al., 2025 — Learned IMU Bias Prediction for Invariant Visual Inertial Odometry](https://arxiv.org/abs/2505.06748) [P].

**Goal:** avoid drift while maintaining geometric filter consistency. **Method:** learn accelerometer and gyro biases from historical inertial data and correct inertial propagation inside an invariant filter; treat bias as an estimated nuisance process with time structure. **Evidence:** visual-inertial experiments, including visual degradation; not IMU-only full attitude. **Limit:** the model's bias estimate can be confounded with motion when inertial excitation is weak, and the visual update helps stabilize otherwise ambiguous states. **Relevance:** an explicit slow-time-scale bias head could improve TG-DCM tilt without replacing a compact recurrent encoder.

### 4.5 AirIO and body-frame feature observability

[Qiu et al., 2025 — AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability](https://doi.org/10.1109/LRA.2025.3581130) [P+C].

**Problem:** naive global-frame preprocessing may obscure useful body-motion relationships, especially for aerial robots. **Architecture:** temporal CNN encoder with velocity/uncertainty outputs; separate paths with and without explicit attitude features; learned motion goes into downstream inertial integration/filtering. Inspected [model/code.py](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/model/code.py) uses Conv1d + BatchNorm + GELU and CodeNetMotion / CodeNetMotionwithRot. In [train_motion.py](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/train_motion.py), the attitude-assisted path feeds gt_rot log-map during training. **Limitation:** if an implementation requires available attitude at runtime, the origin of that attitude must be declared; omitting it is not a clean six-axis comparison. **Relevance:** carefully compare body and gravity-aligned temporal features as the measured task changes.

### 4.6 Tartan IMU: shared representations and adaptation

[Zhao et al., CVPR 2025 — Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) [P+C].

**Goal:** learn motion features transferable among humans, robots and vehicles rather than re-training from scratch. **Method:** shared temporal backbone with platform-specific heads, cross-platform pretraining and efficient LoRA/online adaptation. Paper reports >100 h of multiple-platform IMU data and faster adaptation/inference. **Actual inspected code:** [lstm.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/lstm.py) builds ResNet-LSTM trunk with dog/human/car/drone heads. In [transformer.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py), the registered Transformer option explicitly raises NotImplementedError because its implementation is not shipped in the public LSTM-only release.

**Interpretation:** multi-carrier learning is valuable prior art, but an unavailable backbone cannot be treated as runnable from this revision, and inertial positioning is not gravity-tilt estimation. **Relevance:** a pretrain+adapter study is justified only after simpler bias and motion-alignment failures are controlled.

### 4.7 iMoT and IONext: alternative temporal inductive biases

[Nguyen, Le and Havinga, AAAI 2025 — iMoT: Inertial Motion Transformer](https://doi.org/10.1609/aaai.v39i6.32664) [P/A] uses a progressive series decoupler, modality-specific positional encoding and query motion particles for velocity/trajectory. The point is to model different temporal motion hypotheses rather than a single recurrent state. [IONext 2025](https://arxiv.org/abs/2507.17089) uses a dual-wing feature mixer and spatiotemporal gating to capture local and global motion details.

**Limits:** neither demonstrates a matched-latency six-axis tilt benchmark, and attention/context windows may incorporate future frames. These motivate a small causal TCN/multirate ablation, not automatic replacement of the 2×200 GRU.

### 4.8 Time-frequency evaluation and sensor bias robustness

[Sultan and Greiser, Sensors 2025](https://doi.org/10.3390/s25165161) [P] compare time- and frequency-domain attitude filter behavior against robot arm orientation reference. This type of frequency response and phase-delay analysis is critical when evaluating an estimator's live dynamics: a model can improve angular RMSE while increasing transient delay. [Park, Lee and Lee, 2025](https://doi.org/10.46670/JSST.2025.34.2.130) compare rule-based and machine-learning approaches under gyro bias; [Park et al. 2026](https://doi.org/10.1109/ACCESS.2026.3691717) extend bias augmentation for recurrent models. Paper-specific hyperparameters and quantitative outcomes were not reproduced here.

## 5. 2026 research: conditioned physics, self-supervision, generalization and direct attitude competition

### 5.1 Direct six-axis CNN–BiLSTM under dynamic acceleration

[Ding, Hu, Sun and Yue, Measurement 2026](https://www.sciencedirect.com/science/article/pii/S0263224126021949) [P preview/A] is the closest directly comparable new learned attitude architecture. The paper explicitly contributes axis-separated residual convolutional feature extraction, BiLSTM long temporal context, a sampling-frequency encoding branch, a shortest-quaternion-geodesic loss and a sliding-window policy that trades future frames for accuracy. It evaluates publicly available OxIOD and EuRoC data and reports median attitude estimation error within 3° with >10% gains against selected baselines.

**Why these choices:** local convolution captures high-frequency sensor signatures, BiLSTM aggregates temporal motion context, and frequency encoding reduces sensitivity to sample rate. Residual connections help optimization and shortest-geodesic loss respects quaternion geometry. **Unresolved:** exact split and target heading, latency against current GRU, per-sequence tilt RMSE, remaining paper tables and trainable implementation. BiLSTM can use future observations, so a claimed “real-time” model is not zero-delay by default. **Critical antecedent:** a [2023 six-axis CNN–BiLSTM quaternion estimator](https://doi.org/10.1016/j.measurement.2023.113105) already reported sampling-rate generalization. Consequently these modules alone are not a new principle, and direct 2026 novelty must be in the specific feature/latency arrangement or demonstrated robustness.

### 5.2 DPI-LSTM: physics-informed dual branches for UAV position and attitude

[A Physics-Informed Dual-Branch LSTM Network for UAV Position and Attitude Estimation, Sensors 2026](https://doi.org/10.3390/s26134287) [P].

**Goal:** couple translational and rotational UAV predictions to known inertial kinematics instead of using unconstrained trajectory regression. **Architecture:** shared LSTM temporal encoder feeding separate position and quaternion attitude branches; dynamic losses impose quaternion propagation, coordinate transformation and velocity-position integration consistency. **Evidence:** UZH-FPV drone racing data with authors' position RMSE example 0.0654 m and ablations.

**Unstated implication often missed:** the source specifies a locally gravity-aligned frame, flat-Earth approximation, dataset-provided initial reference trajectory and no explicit online gyro drift compensation; the position error is not evidence that absolute 3D displacement can be indefinitely reconstructed from arbitrary IMU-only motion. A gain in UZH-FPV position RMSE does not prove better TG-style tilt. **Transfer:** when a separate velocity/position state is truly modeled and initialized, branch-specific physics regularizers can improve internal consistency, but check that each penalty is valid for the available sensors.

### 5.3 MoRPI-PINN: physical constraints under particular robot dynamics

[Sahoo and Klein, 2026 — MoRPI-PINN](https://doi.org/10.1038/s41598-026-50630-y) [P].

**Goal:** reduce 2-D mobile robot inertial integration drift when explicit ground-truth trajectory supervision is scarce. **Method:** physics-informed learning for planar inertial kinematics with a motion-specific robot prior and boundary/initial conditions. **Evidence:** the published real-robot setup reports a 4.2-m position error versus 68.6-m 2-D INS and 22.1-m MoRPINet for its study. **Limit:** snake-like controlled motion assumptions, planar state and chosen trajectories make this a specialized physical inference problem. A PINN constraint itself is not proof of generic 3-D attitude novelty.

### 5.4 KISS-IMU: LiDAR-derived self-supervision, motion balance and uncertainty

[Choi et al., 2026 — KISS-IMU](https://arxiv.org/html/2603.06205) [P+C].

**Problem:** inertial data are cheap but accurate training trajectories are expensive, and common motion types dominate learning. **Training:** use a LiDAR iterative closest point (ICP) and pose graph to supply pseudo-targets, train corrections to gyro/accelerometer and learned uncertainty, balance samples using Gaussian mixture motion clusters. **Runtime:** integrate corrected IMU motion and adapt relative measurement weights in a pose-graph / uncertainty-aware inference system.

**Implementation:** inspected [src/models/gmm.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/models/gmm.py), which constructs scikit-learn GaussianMixture motion component probabilities and weights; [src/training/covweight.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/covweight.py), with an EMA-based bounded controller; and [src/training/integrator.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/integrator.py) using PyPose integration.

**Critical distinction:** no *human-generated ground-truth trajectory* at training does not mean no *other sensor*: LiDAR supplies the teacher. ICP/PGO failure and motion priors can bias labels. The paper is primarily about odometry/trajectory and cannot replace direct tilt benchmarks. **Transfer:** motion-balanced training and uncertainty controls can be applied to already valid six-axis tilt targets if cluster identities and pseudo-label provenance are preserved.

### 5.5 AI-IO: aircraft aerodynamics and the value of additional measurements

[Cui et al., 2026 — AI-IO](https://arxiv.org/abs/2603.00597) [P+C].

**Motivation:** quadrotor velocity is poorly constrained by generic inertial history without exploiting rotor thrust and aerodynamic drag. **Method:** a temporal network with transformer modalities includes rotor angular speed and learned velocity/uncertainty updates integrated into an EKF. [src/learning/network/model.py](https://github.com/SJTU-ViSYS-team/AI-IO/blob/5866134d5e9e20c082a32f06a59e99abb9049a15/src/learning/network/model.py) contains temporal convolutions and IMUTransformerWithModality; [losses.py](https://github.com/SJTU-ViSYS-team/AI-IO/blob/5866134d5e9e20c082a32f06a59e99abb9049a15/src/learning/network/losses.py) implements log-standard-deviation Gaussian likelihood. **Assumptions:** extra rotor telemetry and simplified aerodynamic model; cross-airframe transfer is conditional. **Lesson:** observability improves when additional variables are measured, not just by increasing model capacity.

### 5.6 MosaicIMU: carrier-conditioned composition and adaptation

[Zou et al., 2026 — MosaicIMU](https://arxiv.org/html/2606.09355) [P+C].

**Motivation:** platform-specific networks do poorly when moved among pedestrians, vehicles, quadrupeds and drones. **Architecture:** deep residual/depthwise temporal CNN, prototype-based cosine-similarity router, four carrier experts, a shared U-Net-like decoder, separate temporal velocity and positive variance heads, and a history-aware EKF; on new platforms freeze the base and train a small new residual expert/router adapter.

**Actual implementation:** inspected [moe_net.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) and [finetune_net.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/finetune_net.py). The router computes cosine similarity between a context vector and learned prototypes and uses Gumbel softmax during training; the velocity/variance heads use dilated Conv1d and **bidirectional LSTM**; the [filter](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/filter/msckf.py) performs a cloned-state Kalman velocity update. Its [train config](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/configs/train_config.yaml) uses 500-sample windows at 200 Hz nominal sampling.

**Evidence and boundaries:** authors report 40% mean ATE and 34% RTE-10s reductions across their compared settings, but some carrier-specific rows (including a quadruped case relative to TLIO) do not consistently win. Paper limitations include unseen carriers, unusual sensor mounts, severe gyro bias changes and the need for supervision early in online adaptation. Model throughput is not the same as zero-lookahead when each head has a BiLSTM. For TG-DCM, this is an argument for evaluating held-out devices and rare motion separately, not adopting a four-expert network without comparative evidence.

### 5.7 GINIO: equivariance with unresolved paper-to-code consistency

[Kim et al., 2026 — GINIO](https://arxiv.org/abs/2609.25338) [P+C].

**Motivation:** change of sensor coordinates should transform predicted velocity and uncertainty through group laws rather than change them arbitrarily. **Method:** SO(3)-equivariant vector-neuron CNN and GRU variants, equivariant feature updates and covariance congruence; variants condition on additional attitude/context. **Inspected mathematical architecture:** [VNGRUCell](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/vn_gru.py) computes reset/update gates from invariant norms of projected vector features, while all candidate/state features are rotated 3-vectors; this supports a proper SO(3) transformation contract in isolation.

**Serious publication artifact caveat:** [GINIO docs/KNOWN_ISSUES.md](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) is the authors' own worklist of mismatches. It says printed spectral NLL/NEES values do not have matching measured files (printed -1.875/2.974 versus recorded -0.8747/6.774); an LFA-labeled checkpoint actually omitted LFA; a NanoBench result used TLIO pretraining without the comparator being labeled zero-shot; a with-attitude log-map variant fails the physical-remount transformation; some rotated-test cells lack regenerating evidence or mix checkpoint epochs. **Interpretation:** the geometric vector-neuron mechanism remains interesting, but printed performance and stronger “end-to-end equivariance” claims must be conditioned on exact model version, input transformation and checkpoint. This is an observed author-side discrepancy record, **not our own rerun of their server archives**.

### 5.8 Motion acceleration calibration and compensation

[Arzberger and Nüchter, 2026 — Motion-Acceleration Calibration and Compensation in IMUs without External Equipment](https://arxiv.org/html/2607.25784) [P+C partly].

**Problem:** displaced IMUs produce motion-induced acceleration corrupting a gravity-based filter even when base translational acceleration is small. **Model:** calibrate intrinsic accelerometer/gyro bias/axis/scale and estimate base-to-IMU displacement r, then compensate centripetal/tangential terms based on gyroscope motion. The [public calibration source](https://github.com/fallow24/ros_imu_calib/tree/918a19236cecd6a367921e44ac73958ab9fbb9c8) contains multi-position C++ calibration/residual optimization and IMU geometry handling; the reviewed files include src/calibration.cpp, include/imu_tk/calibration.h and src/filters.cpp. **Evidence:** paper simulations and real spherical mobile mapping system. **Limitations:** fixed rigid geometry, sufficiently exciting calibration motion, reliable gyro derivatives and inability to remove arbitrary unknown translational accelerations. This is particularly relevant to TG-DCM's high-rate gravity gate.

### 5.9 Attitude-aided accelerometer calibration and additional-sensor observers

[Yu, Huang and Yang 2026 — ALAC](https://arxiv.org/abs/2606.06308) [P] studies linear triaxial accelerometer scale/misalignment/bias calibration using five arbitrarily oriented measurements **with known attitudes**, an efficient calibration problem but not an attitude-free six-axis online estimator.

[Tchonkeu, Berkane and Hamel 2026](https://arxiv.org/abs/2607.13254) [P] develop nonlinear SO(3) attitude observers using IMU plus barometer and optional magnetic information, deriving convergence properties for the specified sensor setup. These guarantees cannot be transferred to an accelerometer/gyro-only system after removing pressure or magnetic observations.

[Zhai et al. 2026](https://doi.org/10.1016/j.measurement.2025.119824) [A] combine a gyroscope with an **angular accelerometer**, using learned covariance/noise adaptation in a filter; the authors report >17% gains against their selected baselines. Extra angular sensing changes the instrument set.

### 5.10 IMU self-supervision by physics and graph consistency

[Qiu et al., 2026 — Self-Supervised Inertial Odometry](https://arxiv.org/abs/2610.25988) [P/A] uses short-window inertial preintegration as a noisy teacher, a more context-rich student, and a pose-graph distillation stage. This reduces manual labels under repeated motion priors but cannot manufacture an absolute yaw reference. Student and teacher may share slowly drifting bias. The proper relevance is a *testable pseudo-label mechanism* for relative state increments, not evidence of arbitrary long-term fully inertial observability.

## 6. Cross-study technical comparison: what is really new?

| Design axis | Historical baseline | 2024–2026 advance | Demonstrated evidence | Unsatisfied question |
|---|---|---|---|---|
| Geometry | Quaternion / SO(3) EKF | RCAE multipicative adaptive correction; EqNIO O(2); GINIO SO(3) equivariance | Proof/experiments for defined observation groups | Robust physical remount and whole-system state action |
| Accelerometer dynamics | Gravity normalization, fixed confidence | RAESKF compensation/covariance switch; lever-arm calibration; nonminimum-phase analysis | Dynamic experiments and mechanistic derivations | Unknown translation, calibrated uncertainty, general motion |
| Supervision | Full quaternion or tilt regression | Target-gated physics, dual-branch kinematics, physics-constrained 2-D odometry | Selected benchmarks and losses | Bias in accepted training distribution, causal attribution |
| Temporal model | RNN/TCN, 2023 CNN–BiLSTM | Frequency-aware CNN–BiLSTM, gyro transformer, iMoT/IONext mixers | Dataset-specific outcomes | Fair future-lookahead and compute budget |
| Uncertainty | Fixed filter covariance | EqNIO/GINIO covariance transformation, KISS/Mosaic learned variance | Some calibration/odometry results | Reliable OOD coverage; correct tangent covariance |
| Training data | Supervised GT | KISS LiDAR pseudo-label, SSIO preintegration teacher | Motion/trajectory outcomes | Teacher bias, truly sensor-only training |
| Domain transfer | Augmentation | Tartan cross-platform, MosaicIMU prototype experts | Multi-platform experiments | Severe new-device bias, mount changes, adaptation supervision |
| Reproducibility | Papers + checkpoint links | Source-to-table audits (GINIO issues), pinned code | Identifiable failures of released evidence | End-to-end auditable dataset and trained artifact |

A physically valid formulation is not automatically empirically better; a strong benchmark result is not automatically a novel mathematical mechanism. Both should be assessed separately.

## 7. TG-DCM-PINN: exact released method and critical analysis

The [source implementation](../../src/tg_dcm/losses.py) first trains a wxyz quaternion model through a yaw-invariant Charbonnier tilt penalty. It regularizes raw quaternion norm. The target-gated dynamics term propagates target quaternion by measured gyro and accepts a target pair if its tilt consistency error is within 2 degrees; for accepted pairs it penalizes predicted tilt propagation inconsistency with a 0.15-degree deadzone and robust 0.5-degree scale. The gravity term multiplies three Gaussian confidences for acceleration magnitude deviation from 9.81, adjacent acceleration jerk and gyroscope norm; it penalizes misalignment of predicted h with normalized accelerometer after 0.20-degree deadzone. Released weights: tilt 1.0, norm 0.01, dynamics 0.10, gravity 0.05; accel sigma0.25 m/s², jerk sigma45m/s³, gyro sigma1.4rad/s. Full target derivation is in [docs/reference/mathematics.md](../../docs/reference/mathematics.md).

**Why it is plausible:** supervised tilt respects the yaw ambiguity; gyro consistency guards against reference time errors; gravity trust prevents some high-acceleration corruptions; inference is a compact GRU. **What is not established:** that each gate independently improves accuracy, that any gate is novel on its own, that the learned quaternion output is physically consistent on OOD trajectories, or that the confidence score measures a calibrated probability.

**Implementation issues:** source [so3.py](../../src/tg_dcm/so3.py) can interpolate across missing reference quaternion gaps and rejects exact last endpoints; [losses.py](../../src/tg_dcm/losses.py) clamps nonpositive dt before finite-pair checking; target pair gating can selectively remove hard high-dynamics transitions; gravity weighted loss divides by number of accepted frames, not total confidence mass. The first two are concrete code-level correctness hazards; the last two are design choices that need explicit measurement. No package tests or checkpoint inference were rerun here.

**Released evidence:** 137 standard sequences: mean tilt RMSE 1.521°, paired mean difference -0.091° vs RIANN, 95% bootstrap [-0.147,-0.036]. DIODEM 422 segments from 86 raw motions: TG-DCM mean 10.41°, median2.70°, p95 44.20°, max62.38°. Cluster-bootstrap RIANN difference [-0.440,+0.043] includes zero. A pooled mean across 559 sequences is dominated by the DIODEM segmentation distribution, not equal-weight datasets. No multi-seed controlled loss-component ablations were released. Consequently the claims supported are **limited standard-domain aggregate gain** and *conditional, not resolved*, DIODEM improvement.

## 8. Priority research gaps with defensible hypotheses

### Gap A — Observable-only, bias-aware state

A causal **S² tilt observer** with explicit bias and a two-dimensional tangent correction may improve physical consistency and OOD drift without arbitrary absolute yaw supervision. The new contribution would need a precise state/measurement model and identifiable bias assumption. Compare against supervised quaternion GRU, a calibrated 6D ESKF, VQF and RIANN at the same latency; test tangent chart singularities, frame equivariance and drift.

### Gap B — Mechanistic reliability versus statistical motion confidence

Replace some heuristic acceleration rejection with **calibrated lever-arm compensation** when geometry is known, and a fallback uncertainty model when translation remains ambiguous. Compare this against norm/jerk/rate confidence and an adaptive ESKF on controlled nonzero-r and constant-norm translation cases. A general-purpose IMU cannot recover arbitrary translation, so the contribution must state its operating domain.

### Gap C — Label validity and gate-induced selection bias

Decompose numerical time validity, GT integrity, gyro measurement consistency and gravity reliability. Keep a strict mask for genuinely invalid reference data but compare hard gate, soft robust loss and no gate on **unconditionally** evaluated test sequences. Publish rejected gate coverage by motion, raw experiment, angular acceleration and dataset. A reduction in loss on an accepted subset is not a generalization guarantee.

### Gap D — Exact coordinate remount plus physical remount

Build an equivariant estimator whose sensor-axis transform consistently acts on gyro, accelerometer, bias, hidden vector states, attitude and covariance. Algebraically test the full model; separately test a sensor *physically* moved to a new location and mount. This avoids the failure observed in a GINIO with-attitude variant where Log(R) was treated as an ordinary equivariant input vector.

### Gap E — Motion-domain and tail-risk optimization

Optimize high-error motion families using training-only motion clustering and balanced sampling inspired by KISS, while retaining strict raw-motion disjointness. Report p95, maximum and worst device, not just mean. Apply a dataset-balanced estimand in addition to the released segment mean to avoid DIODEM dominance or inflated effective sample size.

### Gap F — Reproducible novel claims

For any candidate, keep full source-to-table provenance: original paper and DOI, reviewed arXiv/release version, pinned code commit, exact split/data/calibration, 5+ independent training seeds, checkpoint SHA256, prediction artifacts and grouped paired statistics. The public GINIO issues demonstrate that even sophisticated theoretical arguments can be undermined by mislabeled experiments. A strong negative result or clear failed-assumption regime is more scientifically valuable than an unsupported novelty narrative.

## 9. Experimental program and falsification criteria

| Stage | Hypothesis tested | Minimum comparators | Evidence required |
|---|---|---|---|
| 0 — Source and data repair | Pipeline integrity changes observable labels and dynamics masks | Released TG-DCM; baseline data recipe | Gap/dt tests, regenerated source hashes, new per-sequence tables |
| 1 — Direct modern baseline | A new direct six-axis model offers true gain at the same delay | RIANN, VQF6D, 6D RAESKF, GRU and bounded-lookahead CNN–BiLSTM | Identical splits, initialization, future-frame budget and tilt RMSE |
| 2 — Loss attribution | Each physics branch has an independent effect | Tilt only, tilt+gyro, tilt+gravity, full | 5+ seeds, gate occupancy, paired group CI, p95 |
| 3 — Geometry/uncertainty | S² + bias + tangent covariance improves generalization | Strong matched-parameter GRU/ESKF | Gauge, remount, SPD, NEES/NLL/coverage, OOD p95 |
| 4 — Mechanistic acceleration | Compensation recovers valid gravity information | Original gate, adaptive covariance, correct/incorrect lever arm | Fast rotation and translation stress, calibration sensitivity |
| 5 — Domain transfer | Group symmetry or balanced sampling reduces mount/motion shift | Augmentation baseline, shared GRU, KISS-like weighted sampler | Held-out carrier, raw-motion clusters and real remounts |
| 6 — Scientific synthesis | Improvements are attributable and reproducible | All frozen finalists | Exact provenance, negative results, one-to-one table regenerators |

At each stage, reject an algorithm if the claimed benefit disappears under equal sensor/latency input, after correcting data, across independent seeds, or when confidence intervals and worst-case tails are considered. Do not use held-out test outcomes to select hyperparameters.

## 10. Conclusion and claims boundary

**Established:** six-axis gravity-referenced tilt is a gauge-invariant S² estimation problem; yaw is not globally observable, dynamic acceleration and bias are central, and modern architecture choices must be evaluated under equal sensing and temporal context. The 2024–2026 literature supplies adaptive filters, learned noise/bias correction, exact equivariance constructions, uncertainty-aware inertial measurements, self-supervision and cross-platform adaptation. These ingredients are not independently novel in TG-DCM merely because combined.

**Evidence from this code audit:** TG-DCM correctly uses yaw-invariant tilt supervision and a compact recurrent inference graph, but has source-level reference/time integrity hazards and lacks component attribution. The released evidence supports a small standard-domain mean benefit against RIANN; it does not establish a resolved DIODEM benefit or strong tail robustness.

**Most defensible next research track:** repair data integrity; run matched modern six-axis and adaptive-filter baselines; perform seeded factorial attribution; then test a causal bias-conditioned S² correction with calibrated uncertainty, optional physical lever-arm compensation and full remount invariance under clearly identified assumptions. A new architecture becomes a scientific contribution only when its distinct mechanism and reproducible empirical benefits survive these controls.

## Primary-source reading list and provenance

**2024:** [RCAE](https://arxiv.org/abs/2401.13092); [DIVE](https://doi.org/10.1109/LRA.2024.3370006); [EqNIO](https://arxiv.org/html/2408.06321).

**2025:** [RAESKF](https://doi.org/10.1016/j.measurement.2024.116097); [Brändle angular acceleration](https://arxiv.org/abs/2502.03681); [AirIO](https://doi.org/10.1109/LRA.2025.3581130); [Tartan IMU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html); [iMoT](https://doi.org/10.1609/aaai.v39i6.32664); [Learned bias](https://arxiv.org/abs/2505.06748); [IONext](https://arxiv.org/abs/2507.17089); [Gyro transformer](https://doi.org/10.1002/rob.70082); [frequency response study](https://doi.org/10.3390/s25165161); [gyro bias comparison](https://doi.org/10.46670/JSST.2025.34.2.130).

**2026:** [Ding six-axis CNN–BiLSTM](https://doi.org/10.1016/j.measurement.2026.122485); [DPI-LSTM](https://doi.org/10.3390/s26134287); [MoRPI-PINN](https://doi.org/10.1038/s41598-026-50630-y); [KISS-IMU](https://arxiv.org/html/2603.06205); [AI-IO](https://arxiv.org/abs/2603.00597); [MosaicIMU](https://arxiv.org/html/2606.09355); [ALAC calibration](https://arxiv.org/abs/2606.06308); [Barometric SO(3) observer](https://arxiv.org/abs/2607.13254); [Motion compensation](https://arxiv.org/html/2607.25784); [GINIO](https://arxiv.org/abs/2609.25338); [SSIO](https://arxiv.org/abs/2610.25988); [Angular-accelerometer KF](https://doi.org/10.1016/j.measurement.2025.119824); [Bias augmentation](https://doi.org/10.1109/ACCESS.2026.3691717).

**Historical context (pre-2024):** [2023 CNN–BiLSTM six-axis framework](https://doi.org/10.1016/j.measurement.2023.113105); [RIANN](https://doi.org/10.1109/LRA.2021.3096167) (verify bibliographic version before citing this DOI in publication); VQF, complementary orientation filters and classical inertial preintegration.

**Implementation provenance:** [original code review](../implementation/target-audit.md), [cross-repository audit](../implementation/external-code-audit.md), [repo pin index](../repositories/index.json) and [paper index](../papers/index.json). Inspecting code does not mean cloning locally, building dependencies or reproducing paper metrics. Paywalled and unavailable originals remain explicitly marked.

**Remaining literature gaps:** a full citation-chained search of new November 2026 works, paywalled appendices of direct attitude papers, faithful reimplementation of RAESKF/Ding at equal delay, and independent device-level experiments. Future expansions should update this review only after new primary-source and artifact verification.
