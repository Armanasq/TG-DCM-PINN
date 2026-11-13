# External implementation audit: code versus paper

All revisions and files are pinned in ../repositories/index.json. This document identifies the *actual computation*, its theoretical motivation, and implementation gaps. No repository was locally cloned or executed, so static code observations are not benchmark replication.

## GINIO (CoRL 2026): important paper-to-artifact discrepancies

The SO(3)-equivariant vector-neuron GRU constructs invariant gates from norms of projected input/hidden vector fields, then multiplies equivariant 3-vector candidates by scalar gates. In airio/model/vn_gru.py, VNGRUCell uses projected VNNorm features to form gate_mlp_r and gate_mlp_z; candidate is VNLinear(x)+r*VNLinear(h) followed by VNLeakyReLU/VNLayerNorm, and h_next=(1-z)h+z*c. This is a concrete architectural symmetry constraint rather than only rotational augmentation. airio/model/vn_codenet.py follows vector-valued convolution and predicts body-frame velocity; test_a2_spectral_equivariance.py tests physical remount, mean/covariance transformation, SPD and negative-log-likelihood invariance.

The authors' docs/KNOWN_ISSUES.md is an unusually useful **critical evidence source**. It states:
- Printed spectral-NLL row: NLL -1.875 and NEES 2.974 lack measured source; stored measurement is NLL -0.8747, NEES 6.774.
- GINIO-RS row labeled body+LFA actually uses a checkpoint trained without LFA.
- GINIO-E result 0.785 uses TLIO-pretrained weights evaluated zero-shot on NanoBench; comparator in the same table was NanoBench-trained.
- Several ID/SO(3) numbers have no generating artifact or consistent protocol.
- With-attitude Log(R_WS) channel fails to transform as a pure vector under a physical remount R_WS -> R_WS Q^T, and therefore fails exact equivariance under that remount convention.
- Some ID/rotated entries are derived from different checkpoint epochs, creating a paired-comparison flaw.

These are **repository-maintained claims about surviving evidence**, not independently rerun results. A robust TG-DCM study should avoid analogous mistakes by publishing a source-to-cell manifest and testing the full transformation of all auxiliary channels.

## EqNIO: what is really learned

The TLIO-master/src/network/model_eq_VNCnn_frame_TLIO_v2_2vectors.py implementation has vector/scalar layers, O(2)-aware orthogonal transforms, norm-based nonlinearities and temporal modules. The separately named model_eq_o2.py is largely test-style cross-product assertions; it must not be mistaken for the full trained architecture. TLIO-master/src/network/losses.py implements MSE and Gaussian log-std / full covariance likelihood; covariance is a learned part of the prediction, not a geometrically justified confidence without calibration tests. Because EqNIO regresses displacement, its translation gains do not imply smaller six-axis tilt error.

## AirIO: body-frame state observability

Actual model/code.py constructs temporal CNN features (Conv1d + BatchNorm + GELU + Dropout) and two main forward variants. CodeNetMotionwithRot explicitly consumes orientation features; train_motion.py supplies label['gt_rot'].Log().tensor() in the relevant branch. A project should distinguish train-time GT orientation availability from runtime orientation initialization, and not label the attitude-assisted path as raw-six-axis-only by omission.

## KISS-IMU: motion balance and teacher dependency

The src/models/gmm.py module uses sklearn GaussianMixture over motion-summary features to identify underrepresented modes and compute weights. src/training/covweight.py has an EMA-smoothed CovWeightController with bounded adjustments. src/training/integrator.py wraps PyPose preintegration. Paper's ground truth independence refers to **manual trajectory labels**; ICP and pose-graph LiDAR remain the training supervisory signal. It is false to call the training modality IMU-only.

## AI-IO: physics requires extra inputs

The src/learning/network/model.py defines IMUTransformerWithModality alongside temporal convolutional layers; losses.py uses a diagonal log-standard-deviation Gaussian likelihood. The paper explicitly adds motor rotor speed and assumes simplified aerodynamics/windlessness. It is a good example of recovering observability by measuring more variables, not evidence that a six-axis-only estimator can learn those missing variables in every motion regime.

## TartanIMU: released backbone differs from implied availability

The src code exposes a ResNet-LSTM shared trunk and separate outputs for dog, human, car and drone. In tartan_imu/model/backbones/transformer.py, the Transformer builder raises NotImplementedError with a clear statement that the Transformer core is absent from the public LSTM-only release. Do not claim independent Transformer experiments from these public files. The inspected architecture is a heterogeneous *positioning* model, not a direct quaternion estimator.

## AirIMU: valuable legacy reference, outside the primary date filter

The model/net.py creates a PyPose IMUPreintegrator with covariance propagation; model/losses.py computes SO(3) logarithm rotation residuals, velocity and position losses. Its seminal preprint predates 2024 and is therefore treated as foundational method lineage. The code was inspected to understand geometric integration, not counted as a new post-2024 contribution.

## Code audit is distinct from execution

For every external repository inspected we captured full revision SHA and individual Python file paths. We did not resolve dependencies, execute training, download weights, hash raw datasets, run test commands, or reproduce experimental figures. No numerical metric above is labeled independently reproduced.

## RSS 2025 explicit bias dynamics — deeper source check

[Debiasing 6-DOF IMU via Hierarchical Learning of Continuous Bias Dynamics](https://arxiv.org/html/2504.09495) ([peer-reviewed RSS DOI](https://doi.org/10.15607/RSS.2025.XXI.009)), source [UMich-CURLY/Debias_IMU](https://github.com/UMich-CURLY/Debias_IMU/tree/7eb2afcaba71d27c9c340850c19158f33c8b2f69), MIT code license.

Inspected modules:
- BiasDy/network_for_BiadDy.py: separate gyro and accelerometer neural vector fields; SO(3) kinematics integrated with predicted bias.
- BiasDy/Interpolation.py: backward-difference cubic-Hermite interpolation and derivatives; causal if only past samples used.
- BiasDy/lie_algebra.py: exponential/logarithm/Jacobians and Lie-state integration helpers.
- BiasDy/learning.py: gyro stage optimized against GT orientation and accelerometer stage against GT velocity/position, with ODE solver.
- BiasDy/dataset.py: reference/initial-state construction.

**Important input distinction:** the network does not require ground-truth *bias labels*, but pose GT and approximate bias initial conditions derived from GT pose differences remain during training. Initial bias is assumed observable from a stationary phase at deployment. Paper tests use same-sequence chronological train/validation partitions but separate unseen test sequences. The model does not fully model stochastic measurement noise. No code run or metric reproduced.

## 2026 bias architecture large-scale preprint — negative-result evidence

[A Large-Scale Empirical Study of Deep Learning for MEMS Gyroscope Bias Estimation](https://doi.org/10.31224/7144), **not peer reviewed**, reports 2,811 study-specific experiments and a null result for its tested SO(3) auxiliary loss. The authors report that EuRoC bias labels come from batch optimization and TUMVI angular-rate references from differentiated/smoothed pose, complicating architecture comparisons. Dataset variance contribution approximately 97% and architecture <1% are **conditional on this two-dataset design**. Full text was read through scholarly indexing; no reproducible public code artifact was verified or executed. This is a method-quality warning, not a general theorem.

