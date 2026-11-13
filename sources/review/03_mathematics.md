# 03 · Mathematics — observable states, losses, and adversarial constructions

**Conventions:** \(R_{WB}\) rotates body to world; \(e_3=(0,0,1)^T\); \([v]_\times u=v\times u\); quaternions scalar-first Hamilton. Gravity/sign is dataset-dependent. These equations are derived from the observation model and [actual loss code](../../src/tg_dcm/losses.py), not copied from article prose.

## 1. The hidden state is a quotient, not a 4-vector

A six-axis accelerometer and gyroscope obey, under a rigid sensor mount,

\[
\omega_m=\omega+b_g+\eta_g,\quad
f_m=R^T(a_W-g_W)+b_a+\eta_a.
\]

When the IMU is displaced by a fixed body vector \(r\) from a reference point, add \(\alpha\times r+\omega\times(\omega\times r)\) to the specific-force model in the appropriate body frame. Gyro kinematics are \(\dot R=R[\omega]_\times\). Define \(h=R^Te_3\in S^2\). For any constant **world** yaw \(Y\) preserving \(e_3\), \(h(YR)=R^TY^Te_3=h(R)\). Thus a six-axis observer cannot identify the yaw gauge from gravity and gyro alone. Training a four-component quaternion can still be practical, but its yaw is a freely chosen representative unless another heading observation or identifying trajectory assumption exists.

For a predicted quaternion \(\hat q\) and target \(q\), define \(q_e=\hat q\otimes q^{-1}\) and \(q_e=(w,x,y,z)\). The implemented tilt is

\[
e_{\rm tilt}=2\operatorname{atan2}\left(\sqrt{x^2+y^2},\sqrt{w^2+z^2}\right),
\]

equal to \(\operatorname{atan2}(\|\hat h\times h\|,\hat h^Th)\) for valid normalized attitudes. The **left** relative rotation matters: reversing the multiplication order can change which frame's vertical is being compared. Quaternion antipodes \(q,-q\) must yield identical results.

## 2. Prediction dynamics and the runtime/training distinction

For piecewise constant corrected gyro \(\bar\omega=\omega_m-\hat b_g\),

\[
R^-_{k+1}=R_k\exp([\bar\omega_k]_\times\Delta t_k),
\quad h^-_{k+1}=\exp(-[\bar\omega_k]_\times\Delta t_k)h_k.
\]

The released [gyro_step](../../src/tg_dcm/losses.py) uses a quaternion exponential of \(\omega_m\Delta t\). It **does not** subtract an explicit learned bias. During training, its loss encourages the predicted sequence to be consistent with propagation; no runtime Kalman or Lie-group correction is executed. This distinction separates “physics-regularized regression” from a “physical observer.”

## 3. The exact optimized objective

Let \(\rho_\delta(e)=\sqrt{e^2+\delta^2}-\delta\), a smooth robust penalty. The supervised term is

\[
L_{\rm sup}=
\langle \rho_{0.001}(e_{\rm tilt}(\hat q_k,q_k))\rangle_{\rm valid}
+0.01\langle(\|\hat q^{\rm raw}_k\|-1)^2\rangle.
\]

The target-trusted gyro gate is

\[
m_k=\mathbb{1}\{q_k,q_{k+1},\omega_k,\Delta t_k\ {\rm valid}\}
\,\mathbb{1}\{e_{\rm tilt}(\operatorname{gyroStep}(q_k),q_{k+1})\le2^\circ\}.
\]

Its auxiliary loss penalizes the disagreement between the propagated \(\hat q_k\) **vertical** and the next predicted vertical, with a \(0.15^\circ\) deadzone and Charbonnier \(\delta=0.5^\circ\).

Gravity confidence is

\[
w_k=
\exp\!\left[-\frac{(\|f_k\|-g)^2}{2(0.25)^2}
-\frac{J_k^2}{2(45)^2}
-\frac{\|\omega_k\|^2}{2(1.4)^2}\right],
\]

where \(J_k\) is finite-difference acceleration jerk in m/s³. With \(s_k=\mathbb{1}[w_k>10^{-4}]\), the code computes

\[
L_g=\frac{\sum s_k\,w_k\,
\rho_{0.5^\circ}([e_{\rm angle}(\hat h_k,f_k/\|f_k\|)-0.2^\circ]_+)}
{\max(\sum s_k,1)}.
\]

The final term is \(L=L_{\rm sup}+0.10L_{\rm dyn}+0.05L_g\). All angular constants inside the executable loss are converted to **radians**. See [losses.py](../../src/tg_dcm/losses.py) and [reference derivation](../../docs/reference/mathematics.md).

### 3.1 A hidden change in regularization strength

Because the denominator is \(\sum s_k\), not \(\sum s_kw_k\), \(L_g=\bar w_{\rm accepted}\times\text{(confidence-weighted average penalty)}\). Therefore a device whose accelerometer norm deviates slightly more from 9.81 can receive a **weaker physics gradient**, even on comparable accepted errors. Hard-gate occupancy is not the same as effective training support. Measure the Kish effective sample size \(N_{\rm eff}=(\sum w)^2/\sum w^2\), conditional gate coverage, and branch gradient norm. Mass-normalization is a testable alternative—not automatically correct, because down-weighting the *total* branch under uncertainty may be intentional.

### 3.2 Gate selection creates a conditional physics claim

Target dynamics gate \(m_k\) is evaluated **using the reference transition**. It may remove poor synchronization, but also drop the high-angular-rate transitions most likely to fail. The reported dynamics loss is \(E[\ell\mid m=1]\), not \(E[\ell]\) on all motion. Decreasing conditional error can coexist with increasing unconditional tilt p95. The right control is a gate-on/off factorial ablation **and a plot of accepted probability versus motion severity**.

### 3.3 A physics consistency term is not an anchor

With \(\omega=0\), any constant predicted \(h\in S^2\) satisfies the dynamics loss exactly. With all gravity observations rejected, the auxiliary physics terms alone cannot select the correct tilt. More generally gyro propagation preserves an entire family of initial orientation states. Thus **temporal self-consistency cannot establish global correctness** without an external or trusted gravity observation and initial conditions.

## 4. Mathematical failure constructions

**F1 · Gravity-magnitude aliasing.** Let true \(h=e_3\), zero gyro and \(f=g(\sin30^\circ,0,\cos30^\circ)\) be a sustained accelerometer output. Then \(\|f\|=g\), jerk \(=0\), \(\|\omega\|=0\); therefore \(w=1\). Nevertheless the observation points \(30^\circ\) away from true gravity. In the consistent sign convention, the required additional acceleration is \(g(\sin30^\circ,0,\cos30^\circ-1)=(4.905,0,-1.3143)\,\mathrm{m/s^2}\). No threshold tuning of those three features can solve this indistinguishability.

**F2 · Missing-reference interpolation.** Consider source quaternions \(q_0,\mathrm{NaN},q_2\) at \(t_0<t_1<t_2\). [interpolate_quat](../../src/tg_dcm/so3.py) removes the invalid index and treats \(t_0,t_2\) as neighbors; it can return \(\operatorname{SLERP}(q_0,q_2,0.5)\) at \(t_1\). A valid mathematical interpolant is **not observed ground truth**. At the final valid timestamp its right-index test rejects an exact endpoint. These are direct consequences of indexing.

**F3 · Zero-mask does not remove NaNs.** [masked_mean](../../src/tg_dcm/losses.py) computes \(\sum m_ix_i\) after residuals \(x_i\) have been computed. In IEEE/PyTorch arithmetic, \(0\times\mathrm{NaN}=\mathrm{NaN}\); therefore a nonfinite gyro or accelerometer can poison the loss even if its mask is false. The gravity branch forms normalized acceleration and angle before multiplying by a zero gate, and dynamics propagate potentially nonfinite gyro before masking. This was verified as a PyTorch scalar arithmetic behavior, not as a reproduction of the full training pipeline. **Safer design:** replace invalid inputs by finite inert defaults *before* every nonlinear calculation and use a validity mask for reduction; ensure no NaN reaches gradients.

**F4 · Zero-norm “finite” quaternion.** [finite_target_quat](../../src/tg_dcm/losses.py) marks a target valid when its four entries are finite, without a nonzero norm threshold. A [0,0,0,0] target therefore passes the mask but cannot represent SO(3); subsequent normalization returns a zero vector. Training tilt and evaluation may then react inconsistently to the invalid target. Reject nonfinite **or non-unit/zero-norm** references before normalization and report affected counts.

**F5 · Subtle angular numerical floor.** [angle_between_unit](../../src/tg_dcm/losses.py) clamps dot to \(1-10^{-7}\) before \(\arctan2\); even identical unit directions produce \(\arccos(1-10^{-7})\approx0.0256^\circ\). Current \(0.15^\circ/0.2^\circ\) deadzones cancel most of this, but reuse as a precision metric would insert a false residual. Similarly the full geodesic function clamps its quaternion dot below 1. For direct metrics use \(\operatorname{atan2}(\|a\times b\|,a^Tb)\) with normalized nonzero inputs and suitable near-zero handling.

**F6 · Evaluation “heading correction” changes tilt.** [align_heading](../../scripts/evaluate.py) multiplies the whole estimated rotation by a GT-derived full 3D offset, not a world-yaw-only offset. Formally, if \(R_{\rm est}=R_x(10^\circ)R_{\rm GT}\), the correction can include \(R_x(-10^\circ)\), eliminating an actual roll bias. Primary raw tilt is unaffected; secondary GT-corrected roll/pitch is not an independent tilt metric.

## 5. Bias learning: what the 2025 NODE really assumes

[Liu et al., RSS 2025](https://arxiv.org/html/2504.09495) define explicit bias ODEs \(\dot b_g=f_\theta(b_g,\tilde\omega,\dot{\tilde\omega})\), and analogously for \(b_a\), integrate pose via \(SO(3)\) kinematics, and optimize orientation/position errors using **pose** reference rather than measured bias labels. A backward-difference cubic Hermite spline provides causal input interpolation. The solver changes Lie charts to avoid Jacobian singularities. The gyro/accelerometer hierarchy uses the causal dependence of translational integration on attitude.

The paper approximates **initial bias** using finite differences of GT pose for each short training interval and assumes a stationary calibration can provide initial bias at inference. It also explicitly neglects a learned stochastic-noise component. Thus “no bias GT labels” is true, while “no reference assumption, no initial bias, or no supervision” is false. [Implementation](https://github.com/UMich-CURLY/Debias_IMU/tree/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy) contains bias networks, backward Hermite spline and Lie integration; its split uses temporally adjacent 80/20 partitions of training recordings for validation but separate test recordings.

## 6. A principled but unvalidated S² correction candidate

Given propagated \(h^-\), choose a local orthonormal tangent basis \(B(h^-)\in\mathbb R^{3\times2}\). A network emits correction \(\xi\in\mathbb R^2\) and, optionally, a positive-definite tangent covariance \(S=LL^T+\epsilon I\). With \(v=B\xi\),

\[
\operatorname{Exp}_{h^-}(v)=\cos\|v\|h^-+\frac{\sin\|v\|}{\|v\|}v.
\]

A residual on the tangent plane can be scored by \(\tfrac12(r^TS^{-1}r+\log\det S+2\log 2\pi)\), with under-model calibration \(E[r^TS^{-1}r]\approx2\). NEES near 2 is **not guaranteed by positive definiteness** and requires a correct residual model plus withheld validation. Tangent bases require chart transition/transport; near the antipodal pole the log map becomes ambiguous. Bias identifiability still requires excitation and/or calibration. This is a research candidate drawing on established geometry, not an independently novel theory.

## 7. Correct SO(3)/O(2) remount laws

For a proper change of body coordinates \(Q\in SO(3)\), body vectors \(v'=Qv\), attitude \(R'=RQ^T\), vertical \(h'=Qh\) and covariance \(S'=QSQ^T\) in full 3D. A body bias transforms \(b'=Qb\). Under reflections with \(\det Q=-1\), angular rate is **axial**: \(\omega'=\det(Q)Q\omega\), unlike polar acceleration \(f'=Qf\). [EqNIO](https://arxiv.org/html/2408.06321) handles this subgroup parity; [GINIO](https://arxiv.org/abs/2609.25338) requires the full system to transform consistently.

An IMU *physically moved* on a lever arm is not just a coordinate change: new acceleration terms appear, so ideal geometric equivariance has a more limited guarantee than many remount robustness headlines imply.

## 8. Mathematical acceptance tests

| Property | Minimal counterexample/test | Evidence required |
|---|---|---|
| Gauge | Left multiply any q by world yaw | Same tilt and loss |
| Quaternion validity | q, -q, zero q, 1e-12 norm | Proper mask/error; no false labels |
| Propagation | Pure rotation, compound rotations, different dt | Quaternion and S² update agree |
| Time | dt=0, dt<0, duplicate, missing and reverse timestamp | Invalid masks before physics |
| Missing truth | finite–NaN–finite and exact last endpoint | No bridge; valid endpoint preserved |
| Masking | NaN gyro/acc at gated-out frame | Finite forward and backward loss |
| Gravity | \(\|f\|=g\) but 30° misdirected | Detector reports uncertainty; no claim of universal disambiguation |
| Equivariance | 1000 random Q, reflected axes and physical remount | Correct full-state law and separate physical stress |
| Covariance | Singular/large/small predicted variance, OOD residual | SPD, calibrated NLL/coverage/NEES |

**Conclusion:** the strongest mathematical opportunity is a **well-specified observable-state update with explicit nuisance-state and uncertainty assumptions**. More auxiliary losses or more quaternion outputs do not change the measurement rank.

## Second-pass mathematical invariant: antipodal labels cannot explain Caruso score changes

For any nonzero quaternion q, \(R(-q)=R(q)\). Consequently a pure quaternion sign-continuity transform and positive normalization **must preserve** \(h(q)\), \(e_{\rm tilt}\), and valid-only per-sequence RMSE, provided original sensor inputs and valid-frame selection are unchanged. [Caruso converter](../../src/tg_dcm/prep/caruso.py) intends only such a transform, but the released [paired scores](../../results/full_evaluation/tables/per_sequence.csv) change for every one of 18 physical recordings, and GT-valid frame counts sometimes differ. This is **not an allowable mathematical consequence** of antipodal correction alone; it is an unlocalized source/validity/artifact disagreement requiring a paired HDF5 diff.

For DIODEM timestamps, \(dt[k]=t_k-t_{k-1}\) but the current physics pair \(q_k\to q_{k+1}\) uses \(dt[k]\). The true forward increment is \(t_{k+1}-t_k=dt[k+1]\) for \(k\ge0\). This convention error becomes material only under irregular sample spacing. DIODEM is test-only in released TG training, so no past training effect is asserted. [Detailed audit](09_second_pass_verification.md).

**Statistical object:** a bootstrap CI over 137 named files is not automatically a CI over 137 independent physical recordings. When 18 files are reprocessed copies of the others, the correct sampling unit is the raw sensor recording. Dropping the orig duplicates reduces the published TG-minus-RIANN mean from \(-0.090626^\circ\) to \(-0.043945^\circ\), with a new approximate paired-bootstrap interval \([-0.1004,+0.0105]^\circ\). This is a sensitivity test on published data, not a newly trained model.
