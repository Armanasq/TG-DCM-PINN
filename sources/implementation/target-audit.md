# TG-DCM-PINN implementation and methods audit

**Reference revision:** e27a43a1d3b6b4d3a82355414523e25bc97ca370 (main before research-only additions). Code inspected via GitHub file API, not locally executed.

## Architecture and released evidence

The network processes [acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,dt] and emits scalar-first Hamilton quaternions, body-to-world, using a 2-layer GRU with 200 hidden units and a 4-channel head. The estimator predicts a full quaternion representation but **claims/evaluates only yaw-invariant tilt**. Physics losses are applied during training, not as inference-time filter corrections. Training uses supervised Charbonnier inclination, raw-output quaternion norm penalty, gyro target-gated vertical dynamics, input-derived gravity confidence, Ranger, quaternion augmentation and EMA. The selected 2026 checkpoint is epoch 491 out of 512, with one seed documented.

Published evaluation: standard test 137 sequences mean tilt RMSE 1.521 degrees; paired mean difference vs RIANN -0.091 degrees, bootstrap 95% [-0.147,-0.036]. DIODEM has 422 segments from 86 raw motions, TG-DCM 10.41-degree mean / 2.70-degree median / 44.20-degree p95 / 62.38-degree max; clustered paired CI vs RIANN [-0.440,+0.043] crosses zero. Therefore **no resolved DIODEM improvement against RIANN at 95%**. Means pool sequences, not equal-weight datasets. No controlled multi-seed component ablations are released.

## Checked mathematical contracts

Let R_WB be predicted body-to-world attitude and e_z=(0,0,1). The observed vertical is h(R)=R_WB^T e_z in the sensor frame. For wxyz unit q, code evaluates h=[2(xz-wy),2(yz+wx),1-2(x^2+y^2)]. A valid tilt angle is acos(clip(hhat dot hgt,-1,1)); equivalent quaternion relative tilt uses 2 atan2(sqrt(qerr_x^2+qerr_y^2),sqrt(qerr_w^2+qerr_z^2)) when qerr=qhat tensor conj(qgt).

Gyro step code: q_{t+1}=normalize(q_t tensor exp(omega_t Delta t)), where quaternion exp uses cos(theta/2) and sin(theta/2)*omega/theta. Auxiliary gyro supervision compares h of propagated prediction to next predicted h, *only if* propagating consecutive reference quaternions by measured gyro agrees in tilt within 2 degrees. This is a label-quality gate that may systematically exclude the hardest maneuvers, not proof of robust consistency across them.

Gravity branch: w_g=exp(-(norm(a)-g)^2/(2 sigma_a^2))*exp(-jerk^2/(2 sigma_j^2))*exp(-norm(omega)^2/(2 sigma_w^2)), gated above 1e-4. Code uses sigma_a=.25m/s^2, sigma_j=45m/s^3, sigma_w=1.4rad/s, and branch weight .05, with .2-degree deadzone. If gravity acceleration and motion acceleration have equal total magnitude, the norm gate alone cannot discriminate. A motion-compensation model would require additional assumptions; it cannot recover arbitrary acceleration from an IMU alone.

## Code-inspection findings requiring tests

1. **High — quaternion interpolation bridges invalid gaps.** In src/tg_dcm/so3.py, interpolate_quat collects only finite sample indices, searchsorts that compressed array and SLERPs between adjacent **valid entries**, without requiring original indices to be adjacent. For e.g. a valid quaternion at sample 0, invalid at 1, valid at 2, query at sample 1 is assigned an interpolated orientation even across missing ground truth. The function's "Gap-aware SLERP" contract appears violated. Also right=searchsorted(..., side="right") excludes the exact final valid timestamp. This is a source-level finding, not a test result. Test gaps/last endpoint, fix by respecting original segment boundaries and exact endpoints, then re-evaluate effects on all prepared datasets.
2. **Medium — invalid/nonpositive dt masking.** In TargetDCM3Loss.forward, dt is clamp_min(1e-6) **before** the pair mask checks finiteness. Zero/negative finite time intervals can consequently enter the target-gated propagation and jerk calculations with artificial positive Delta t, instead of being rejected or separately flagged. Test timestamp collapse and reverse order; reject nonpositive or implausible dt before clamp, with explicit validity masks.
3. **Medium — confidence scaling depends on gate occupancy.** The gravity term computes masked_mean(gate_weight * penalty, gate), dividing by the **count of accepted frames**, not the sum of accepted continuous confidence weights. This means effective branch strength varies with confidence distribution. It may be intentional; compare count-normalization versus confidence-normalization in controlled ablations rather than declaring it a correctness bug.
4. **Research — ground-truth-informed target gating selection.** Accepted target pairs are defined using target gyro-propagation error, a sensible label-integrity check. But excluded high-motion data might be systematically difficult, and the observed dynamics loss is conditional on surviving reference pairs. Report gate coverage by dataset, angular velocity, acceleration, target discontinuities, and accepted/rejected error.
5. **Research — single run and DIODEM tail.** Reported small standard mean gain and heavy-tail outliers are not enough to establish that each architectural or loss component is responsible. Use at least 5 controlled seeds and clustered paired statistics, and retain DIODEM p95/worst-case as a safety-relevant guardrail.
6. **Documentation — duplicate repository identity.** pyproject.toml Homepage/Repository/Issues and README badges point to GinkgoQ/TG-DCM-PINN, while the working reviewed repository is Armanasq/TG-DCM-PINN. Treat this as an identity/provenance ambiguity before comparing tags/releases; no behavioral source files were changed here.

## Source paths actually inspected

- src/tg_dcm/losses.py: qnorm, inclination_error_rad, TargetDCM3Loss, RiannV2Loss.
- src/tg_dcm/so3.py: SLERP, gap interpolation, gyro-time-offset estimation and continuity handling.
- src/tg_dcm/train_impl.py: RNNLearner architecture, augmentations, optimizer, EMA.
- src/tg_dcm/evaluation.py and docs/evaluation.md: intended metrics and bootstrap protocol.
- configs/model.yaml and training/summary.json: released hyperparameters and selected checkpoint.
- README.md: claims and per-dataset reported results.

**Verification status:** static source inspection. Tests, model re-inference, and raw-data replay not executed. The above bugs are actionable hypotheses until a regression test reproduces them under the real package.

## Additional high-priority evaluation and masking audit (2026-11-13)

These extend the preceding source-level findings. No changed benchmark numbers are asserted before raw-file replay.

1. **P0 — NaN propagation despite masks.** In src/tg_dcm/losses.py, masked_mean computes sum(values * mask); a false mask does **not** prevent 0*NaN=NaN when target gyro or gravity residual is already invalid. Residuals/normalization are formed before masking. PyTorch scalar arithmetic verified separately. Before computing nonlinear residuals, substitute finite neutral values for invalid inputs and exclude the originals with explicit masks. Test forward **and backward** with NaN/Inf in gyro, acceleration and GT.
2. **P0 — invalid finite quaternions.** finite_target_quat checks only component finiteness. All-zero target quaternions pass even though their norm is zero; subsequent normalization is not a valid SO(3) attitude. Require finite norm above tolerance and record rejection counts.
3. **P0 — VQF noncausal initialization.** scripts/evaluate.py run_vqf uses warmup=min(500,len(q)//4), then q[:warmup]=q[warmup:warmup+1]. This backfills earlier timestamps from a **later estimate**. Compare raw VQF startup against a separate common fixed-time burn-in of all methods.
4. **P0 — secondary metric GT calibration mislabeled as heading-only.** scripts/evaluate.py align_heading averages R_GT*R_est^(-1) as a **full 3D rotation**, then applies it before roll/pitch RMSE. This can remove true roll/pitch bias. Primary yaw-invariant tilt is evaluated **without** this correction; distinguish its validity from the secondary GT-aided columns.
5. **P1 — input timestep parity.** run_riann_onnx supplies constant 1/fs; VQF is configured at 1/fs; TG-DCM run_model uses stored dt. Actual impact depends on sequence time irregularity, which is not measured here.
6. **P0 — wholly invalid GT can receive a score.** validate_sequence replaces invalid quat rows with identity; evaluate_sequence uses an all-true mask if no valid GT remains. Such sequences are unscorable, not identity-ground-truth tests. Stream length uses min() truncation, which can hide alignment faults.
7. **P1 — invalid DIODEM dt imputation.** load_h5 replaces nonfinite or nonpositive timestamp increments with median valid dt before subsequent checks. Keep original timestamps, report integrity, and reject impossible transitions instead of fabricating a valid physical interval.
8. **P1 — “static” condition is only near-g magnitude.** is_static uses |norm(acc)-9.81| threshold; translational accelerations can pass. Rename conditional metric or provide independent static-motion truth.
9. **P1 — runtime timings across unequal compute.** Baselines run on CPU; TG-DCM can run CUDA; execution may overlap. Report same-device sequential latency and actual future/lookahead cost before speed ranking.
10. **P1 — empirical effective diversity.** data.py creates correlated 50–500Hz resampling views of the same train motions. Split by original device/recording, distinguish held-out sampling-rate generalization from genuine motion/device OOD.

**Mathematical counterexample:** f=g(sin30deg,0,cos30deg) with omega=0 and constant f gives magnitude/jerk/rate confidence exactly 1 while gravity is 30deg away. See [mathematical audit](../review/03_mathematics.md). No claim is made that this exact motion appears in the released benchmark.

**Evidence standard:** static code-path finding != observed numerical bias. Required follow-up: regression tests, raw-data incidence counts and per-sequence before/after paired effect estimates. The previous single-seed aggregate result cannot attribute a gain to a particular component.

