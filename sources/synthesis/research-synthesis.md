# 2024–2026 critical synthesis for TG-DCM-PINN

## 1. Observable state and mathematical objective

For a six-axis IMU the gyroscope observes angular velocity corrupted by bias/noise and the accelerometer observes specific force, not gravity alone:

    omega_m = omega + b_g + eta_g
    f_m = R_WB^T (a_W - g_W) + b_a + eta_a

The gravity direction in body coordinates h=R_WB^T e_z belongs to S^2. Without a magnetometer, external heading, or a sufficiently restrictive motion prior, global yaw is a gauge freedom. The correct primary geometric target is tilt on S^2, not an apparently absolute SO(3) heading. A source-model-invariant tilt error is:

    e_tilt = atan2(||h_pred x h_gt||, h_pred dot h_gt).

Gyro provides relative propagation (time increments), not a global heading anchor. A purported method removing absolute yaw drift without extra information must state an identifying assumption and experiment.

## 2. Direct methodological families

**Robust classical filters (Wei et al. 2025 RAESKF):** error-state DCM and adaptive dynamic-acceleration compensation/covariance inflation. The strength is interpretable innovation handling and limited compute; weakness is threshold/model tuning under unknown accelerations. Evidence: publisher abstract and dynamic experiments only; must be rerun on the same tilt sequences.

**Adaptive SO(3) corrections (Oveissi et al. 2024 RCAE):** multiplicative retrospective-cost correction avoids a fixed MEKF Jacobian/bias estimate. Strength is manifold preservation and adaptation; critical mismatch is the method assumes noisy orientation observations, so not a drop-in six-axis-only baseline.

**Deep six-axis attitude regression (Ding et al. 2026):** temporal CNN plus bidirectional LSTM learns quaternion dynamics under disturbance; reported median under 3 degrees on the paper's protocol. Its future lookahead is part of the model's observation budget. One may compare only at the same latency and tilt metric.

**Physics-regularized joint navigation (DPI-LSTM, MoRPI-PINN):** differential kinematics constrain a shared encoder and separate state heads. Strong for physically coupled tasks, but their initialization, degrees of freedom, trajectories and positional RMSE do not establish fully observable absolute inertial pose.

**Mechanistic compensation (Brändle 2025; Arzberger–Nüchter 2026):** an IMU away from the rotation center adds alpha x r and omega x (omega x r) to specific force. Reducing accelerometer gain suppresses false gravity at a bandwidth cost; compensating known/estimated lever-arm force can retain the valid gravity observation. This is especially relevant to TG-DCM's motion-dependent gravity gating.

## 3. Transferable innovation, with strict task mismatch

**EqNIO 2024/2025:** gravity-preserving O(2) equivariance reduces remount/yaw duplication in learned odometry; canonical frames and O(2) covariance transform must be adapted with correct angular-velocity parity.

**GINIO 2026:** full SO(3) vector-neuron equivariance and covariance congruence (mean Q mu, covariance Q Sigma Q^T) offer more principled mount robustness. Important: code repository includes numerous specific disputes between published table cells and surviving checkpoints. A geometric proof for an idealized feature transformation is not proof of physical remount robustness under sensor translation, bias, timestamp or untransformed attitude context.

**AirIO 2025; AI-IO 2026:** body-frame features retain motion-dynamics information; AI-IO additionally shows that rotor speeds are relevant for aircraft velocity observability. Extra attitude/rotor inputs cannot be silently transferred into a six-axis method.

**Learned Bias 2025:** slowly varying bias can be handled as a nuisance process while preserving an invariant filter structure. But bias identifiability with IMU-only data remains conditional on excitation and motion assumptions.

**KISS-IMU 2026; SSIO 2026:** motion imbalance causes failure on unusual maneuvers and pseudo-supervision reduces annotation dependency. KISS uses LiDAR during training; SSIO uses short-window IMU preintegration plus pose-graph distillation. Their successful velocity/position objectives do not imply arbitrary linear acceleration is distinguishable from gravity using a single accelerometer.

**TartanIMU 2025; iMoT 2025; IONext 2025:** heterogeneous pretraining, attention/particle hypotheses and multi-scale convolution are viable architecture hypotheses. More layers or parameters alone are not a research novelty. Public TartanIMU Transformer core is not released in inspected revision.

## 4. Most defensible open research problems

1. **Bias-conditioned, covariance-calibrated tilt innovation on S^2:** propagate a unit gravity vector by bias-corrected gyro, then predict only a two-dimensional tangent correction and its 2x2 SPD covariance. Compare rigid-body equivariance and calibrated innovation statistics. Novelty conditional on a 2024–2026 prior-art search of exact S^2 observer/uncertainty combinations; not yet an established original contribution.
2. **Physics-residual-conditioned trust with mechanical compensation:** distinguish (i) measured/compensated gravity, (ii) unverifiable dynamic acceleration and (iii) discontinuous or missing reference targets. Replace norm-only acceleration confidence with learned residual consistency + a heteroscedastic head, with safeguards against degenerate gating (ignoring every difficult frame). The novelty is *combination plus proof/empirical results*, not the individual components (already described in published literature).
3. **Gauge-consistent sensor-remount equivariance:** make predictions commute with arbitrary sensor coordinate changes, while separately testing *physical* remounts that change r, bias and time alignment. Use GINIO-specific error evidence to prevent a wrong transformation contract.
4. **OOD and rare-motion targeted optimization:** cluster motion using magnitude, jerk, angular-rate, frequency and label-reliability features; weight rare high-dynamics motions without overweighting target-corrupt sequences. KISS's GMM approach is prior art, not a new generic principle.
5. **Causal online correction with finite-state memory and explicit latency:** compare 2-layer GRU, lightweight temporal CNN, S^2 geometric predictor and fixed-lookahead BiLSTM at matched delay, parameter count and compute. Distinguish predictor architectural improvements from higher observation budget.
6. **First-class label quality:** quantify gyro/GT disagreement coverage, timestamp warping, quaternion sign/gap interpolation, frame handness and repeat-segment leakage. Evidence quality may dominate a 0.091-degree mean gain.

## 5. Avoid unsupported novelty claims

TG-DCM's physics-informed loss, quaternion/vertical geometry, gating and recurrent prediction are combinations of established ingredients. Novelty must be attributed to a precise **previously unreported combination or formal property**, and then demonstrated using ablation, statistical significance, cross-platform transfer and code-level reproducibility. Current results support a small standard-domain aggregate improvement, but do not prove that the gates or each component caused it. The DIODEM mean gain over RIANN is not resolved at 95%, and DIODEM tail reliability remains a major challenge.

## 6. Decision order

First fix and regression-test preprocessing/temporal integrity. Next add reliable direct RAESKF and Ding-style matched-latency baselines. Then run factorial gate/physics ablations with OOD and clustered confidence intervals. Only after those checks evaluate the more expensive S^2-geometry/equivariant-uncertainty architecture; report counterexamples and failure regimes.


## Second-pass corrected synthesis: attribution and source independence

The standard 137-file set duplicates 18 Caruso physical sensor streams across two target-processing variants. Their published paired error changes contradict *pure antipodal sign correction* under the current converter. Removing the orig variants changes TG-minus-RIANN mean from -0.090626° to -0.043945° and paired-bootstrap CI to approximately [-0.1004°, +0.0105°]. This overturns an unqualified claim that the standard-domain mean advantage is statistically resolved independently of how raw recordings are weighted.

The EuRoC preparer uses externally estimated gyro bias from the reference-state CSV; DIODEM exports lack sufficiently documented 11-session/rigid-foam calibration lineage; validator selects reset-window live weights while release uses streaming EMA. These are **method/protocol provenance defects or gaps**, not a proof the physical model is wrong. The [2025 DIODEM original paper](https://doi.org/10.1038/s41597-025-05468-w), [RIANN 2026 training reimplementation](https://github.com/daniel-om-weber/riann_dev), [2025 optical-flow gravity observer](https://arxiv.org/abs/2508.21163) and [2026 reduced-order gravity observer](https://arxiv.org/abs/2609.26612) close important primary-source omissions.

**Revised contribution boundary:** TG-DCM currently has a differentiated target-gated **training formulation** and descriptive numerical reports, but the architecture and body-gravity representation are prior art; component causality, robust cross-device gains, DIODEM superiority and standard significance under independent physical-recording weighting are **not yet established**. [All findings, formulas and tests](../review/09_second_pass_verification.md).


## Newly investigated method and precise prior-art boundary

[Closure-Constrained Tilt Observer](../proposals/01_closure_constrained_tilt.md) exploits a different conditional gravity observation: if the *sensor point* has bounded change in world velocity over a finite interval, integrating its relative-body-rotated specific force yields a spherical feasible set for initial tilt despite large instantaneous acceleration. This is an exact preintegration consequence, not a new law. The provided deterministic synthetic check illustrates both ideal cancellation and an adversarial false-closure case.

[Brás et al. 2013](https://doi.org/10.1016/j.sysconle.2013.06.008) developed set-valued attitude observers; [Forster](https://doi.org/10.1109/TRO.2016.2597321), [Delama 2024/25](https://doi.org/10.1109/LRA.2024.3511424), classical ZUPT and learned event detectors supply close prior art. Novelty could only arise from a demonstrably distinct *causal event-admission plus correct conditional bound-aware update* that improves beyond an ESKF using the **same closure event and preintegrated measurement** on nonstationary dynamic intervals. Predictions from raw IMU alone do not certify closure under arbitrary translation. [Prior-art and experiment program](../proposals/README.md).
