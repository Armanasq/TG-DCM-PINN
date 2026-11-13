# Research candidate: closure-constrained tilt inference from six-axis IMUs

**Working name:** CCTO — Closure-Constrained Tilt Observer. **Status:** analytical proposal plus deterministic synthetic validation; *not demonstrated algorithmic novelty, accuracy or publishability*. **Freeze:** 2026-11-13. **Original target:** [TG-DCM-PINN](../../README.md); review of [limitations](../review/06_limitations.md), [methods](../review/03_mathematics.md), [future work](../review/07_future_works.md), [second-pass evidence](../review/09_second_pass_verification.md).

## 1. Problem chosen: reject incorrect instantaneous gravity, not just high acceleration

TG-DCM gravity trust is a product of norms of specific force, jerk and angular rate. Its gate is maximally confident for an incorrect, constant, norm-g acceleration vector with zero gyro and jerk. A larger GRU, TCN, more calibrated variance or a generic S² output **cannot remove** this ambiguity without introducing an additional physical motion assumption. The target's 137-file fixed-checkpoint mean advantage becomes inconclusive when duplicate Caruso recordings are excluded, and its DIODEM p95 remains large. Therefore pursue a method whose *new measurement equation* is explicit rather than a larger learned representation.

**Central hypothesis:** a motion interval with **bounded sensor-point velocity change**, even if acceleration is strongly nonzero inside the interval, yields a useful integrated gravity-direction observation. A causal filter can use this interval-derived observation selectively. Learning may predict which windows plausibly satisfy the closure assumption, but **a learned prediction is not a guaranteed physical certificate**.

The idea is deliberately narrower than "six-axis IMU always recovers dynamic gravity." It is intended for movements with intermittent reliable velocity closure, such as controlled periodic motion or return-to-rest trajectories; performance outside those domains is an explicit negative control.

## 2. Derive the gravity observation from the measurement model

World frame W uses gravity \(g_W=-g e_3\); \(R(t)=R_{WB}(t)\) maps body to world; \(h_i=R_i^Te_3\) is the body-frame vertical at window start \(i\). The calibrated accelerometer reads specific force at its own sensor point:

\[
f_m(t)-b_a(t)-n_a(t)=R(t)^T\big(a_{s,W}(t)-g_W\big).
\]

With relative rotation \(\Gamma_{it}=R_i^TR(t)\) from integrated, bias-corrected gyro, define

\[
u_{ij}=\int_{t_i}^{t_j}\Gamma_{it}\big(f_m(t)-\hat b_a(t)\big)\,dt,\quad T=t_j-t_i.
\]

For perfect calibration and propagation,

\[
u_{ij}=R_i^T(v_{s,j}-v_{s,i})+gTh_i.
\tag{1}
\]

This follows by multiplying measured specific force by \(\Gamma_{it}\), integrating, using the fundamental theorem of calculus for sensor-point acceleration and \(R_i^Te_3=h_i\). No assumption is made that \(\|f_m(t)\|\approx g\) at any instantaneous time. **The closure condition is on the sensor point's *world velocity*, not on position, angular velocity, world-frame body velocity, or a label inferred from acceleration norm.**

When \(\Delta v_s=0\), (1) gives

\[
h_i=\frac{u_{ij}}{gT},\quad \|u_{ij}\|=gT.
\tag{2}
\]

Numerically normalize \(u\) only if it is nonzero; separately check magnitude consistency. A sustained tilt error due to dynamic acceleration can cancel in this *interval integral*, while single-sample acceleration directions remain misleading.

### 2.1 Bounded closure gives a spherical feasible region, not arbitrary confidence

If \(\|\Delta v_s\|\le\epsilon_v\) and a **conservatively calibrated** total preintegration error satisfies \(\|u_{\rm observed}-u_{\rm true}\|\le\epsilon_u\), let \(\epsilon=\epsilon_v+\epsilon_u\). Equation (1) implies

\[
\mathcal C_{ij}
=\left\{h\in S^2:\|u_{ij}-gTh\|\le\epsilon\right\}.
\tag{3}
\]

For \(\rho=\|u_{ij}\|>0\), its spherical cap has center \(c=u_{ij}/\rho\) and half-angle

\[
\alpha=\arccos\!\operatorname{clip}\!\left(
\frac{\rho^2+(gT)^2-\epsilon^2}{2\rho gT},-1,1\right).
\tag{4}
\]

First check feasibility: \(|\rho-gT|\le\epsilon\). If \(\epsilon\ge\rho+gT\), the cap is the **whole sphere**, not a meaningful direction constraint. If the bounds are inconsistent, report a failed closure/bias/noise assumption rather than projecting onto a fabricated feasible set.

**Conditional set containment:** if the velocity-change and sensor-error bounds really hold, then \(h_i\in\mathcal C_{ij}\). This is a direct consequence of (1) and the triangle inequality. It is not a theorem about a learned closure detector, arbitrary motion, or an estimated covariance. Choosing the cap center gives a worst-case angular error at most \(\alpha\) *under those conditions*; choosing any learned point inside the cap gives a looser worst-case bound of \(2\alpha\) by the spherical triangle inequality.

### 2.2 System-level body-frame equivariance

For any proper redefinition of body coordinates by \(Q\in SO(3)\), gyro, force and bias transform as \(Q\) vectors; \(\Gamma'_{it}=Q\Gamma_{it}Q^T\); \(u'_{ij}=Qu_{ij}\); \(h'_i=Qh_i\); and \(\mathcal C'_{ij}=Q\mathcal C_{ij}\). Thus the **physical measurement construction** is exactly coordinate-equivariant (apart from numerical integration/error). The learned closure head must consume invariant features or satisfy an equivalent full-system group law for the complete estimator to be equivariant. Reflections are not included here: gyro is an axial vector under \(\det Q=-1\).

This law does *not* cover moving an IMU physically on a lever arm. Its velocity at each endpoint changes by \(\omega\times r\); a body/base return-to-rest flag cannot be substituted for sensor-point velocity closure without analyzing these terms.

## 3. Causal algorithm, inference contracts and computation

**State:** a unit tilt vector \(h_k\), optionally gyro bias \(\hat b_{g,k}\) and a 2D tangent uncertainty; a finite ring buffer of relative rotation and preintegrated specific force. World yaw remains arbitrary.

**Step A — physical propagation:** \(h^-_{k+1}=\operatorname{Exp}(-[\omega_{m,k}-\hat b_{g,k}]_\times\Delta t_k)h_k\). Reject invalid/negative dt rather than clamp it into apparent physics.

**Step B — motion-closure candidates:** for predeclared look-back lengths \(T\in[T_{\min},T_{\max}]\), predict a *probability/score* \(p_{\rm cl}(i,k)\) that sensor velocity change lies below \(\epsilon_v\). During training this can use reference velocities as **teacher labels only**. At runtime only IMU-derived features are allowed for a six-axis-inference claim. The detector score is not a deterministic guarantee; the false-closure error is a primary endpoint.

**Step C — conditional physical feasible cap:** compute \(u_{ik}\), \(\rho\), required bound \(\epsilon_v+\epsilon_u\), and feasible cap (3) only for candidates meeting the declared closure policy. A magnitude-consistency residual \(|\rho-gT|\) is a **necessary but not sufficient** test for real closure. Do not claim that a small residual verifies the endpoint velocities.

**Step D — update:** transport the cap to the current time through \(\Gamma_{ik}^T\). If closure was independently verified and the cap is narrow, use a conservative geodesic projection, or choose its center for a strict conditional angular guarantee. If closure is merely predicted by a learned model, use a bounded/probabilistic robust innovation, report no deterministic certificate, and preserve a no-update fallback. An empty intersection of separately *verified* feasible caps flags violated bounds/time/calibration, not an instruction to force a new tilt.

**Step E — robustness:** monitor physical model residual, cap radius, time since last accepted closure, closure false-positive risk and actual sensor delay. A model should become *less certain* when closure is missing; it must not hallucinate a new gravity vector from the old heuristic confidence.

### 3.1 Constant-time interval query

Maintain a cumulative relative orientation \(\Gamma_{0t}\) and vector preintegral \(P_t=\int_{0}^{t}\Gamma_{0\tau}(f_\tau-\hat b_a)d\tau\). Then, for an interval \(i\to j\),

\[
u_{ij}=\Gamma_{0i}^{T}(P_j-P_i).
\]

Each candidate interval query takes O(1) vector/matrix work; O(H) for H fixed look-back horizons, with an O(L) rolling buffer. This algebra is **standard preintegration**, not novel on its own. Changing estimated bias retrospectively invalidates naive prefix integrals; either re-integrate a bounded history or maintain bias Jacobians as in existing preintegration methods.

## 4. The counterexample that prevents a false paper claim

Let \(R=I\), gyro=0, and choose constant \(f=g(\sin30^\circ,0,\cos30^\circ)\). This force is exactly magnitude \(g\), so TG-DCM norm/jerk/rate trust is maximal. Its integrated vector satisfies \(\|u\|=gT\); hence **even a perfect integrated-magnitude closure check passes**. But the actual acceleration is \(f-ge_3\neq0\), and the sensor's velocity change is nonzero. Incorrectly asserting closure would return a 30-degree wrong tilt.

Therefore **no purely inertial closure detector can be physically certified for arbitrary motion from magnitude and temporal self-consistency alone**. Real certification requires independently justified endpoint velocity bounds (e.g., known verified contact/platform state); otherwise the closure hypothesis is a learned, domain-dependent prior and all coverage statements must be probabilistic/empirical rather than absolute.

## 5. Bias and uncertainty must not be conflated

Relative rotation and preintegration are biased; the integrated force error generally **increases with horizon length**. A rough conservative bound is

\[
\epsilon_u \le \int_i^j \left(
\epsilon_f(t)+\theta_{\Gamma}(t)\|f_{\rm corr}(t)\|
\right)\,dt+\epsilon_{\rm disc},
\]

where \(\theta_{\Gamma}(t)\) upper-bounds rotation error in radians, \(\epsilon_f\) bounds accelerometer calibration/noise, and \(\epsilon_{\rm disc}\) bounds integration/time discretization. A precise theorem must use validated time-varying bias/gyro error bounds and account for correlated errors; not treat learned NLL variance as a hard bound. If \(\epsilon_u\propto T^2\) while \(gT\propto T\), longer windows may *worsen* the cap. There is no universal monotonic improvement with integration time.

[RSS 2025 continuous bias NODE](https://doi.org/10.15607/RSS.2025.XXI.009) is prior art for explicit learned gyro bias; [Delama et al. 2024/2025](https://doi.org/10.1109/LRA.2024.3511424) for equivariant biased preintegration. The first CCTO experiment should use fixed independently calibrated bias. Introduce a learned bias head **only after** establishing whether closure alone adds information beyond a classical, carefully initialized filter.

## 6. What is and is not the possible contribution

**Established prior art, *not* to claim novel:** IMU preintegration; ZUPT/return-to-rest constraints; gravity-direction S² state; set-valued attitude observers; Lie-group equivariance; learned bias/covariance; use of a neural closure classifier. See [prior-art audit](02_prior_art_and_novelty.md).

**Potential contribution requiring proof and data:** a *causal, selectively admitted, multi-horizon endpoint-velocity closure observation* for six-axis **tilt** that remains useful under high instantaneous acceleration, equipped with truthful conditional feasible sets and an explicit false-closure fallback. Publication would require demonstrating that the proposed interaction improves the relevant operating regime **beyond ordinary ZUPT, standard preintegration and an equally informed ESKF**, not a novel acronym or a re-statement of (1).

**Claim boundary:** "six-axis at inference" is feasible with a learned closure hypothesis, but certified coverage then holds **only conditionally on true closure**. If no closure occurs, the method may reduce to gyro propagation and be *worse* than ordinary gravity-correcting filters; this is a mandatory negative control.

## 7. What has actually been tested

[Reproducible synthetic check](../tools/verify_motion_closure.py) uses 4,096 midpoint samples over 2.0 seconds and exact relative rotations, with zero-mean sinusoidal world acceleration. Output on this machine:

- True sensor-point velocity change: \(5.88\times10^{-16}\ \mathrm{m/s}\) (numerical zero).
- Largest apparent tilt from a single rotation-compensated specific-force direction: **41.8817 degrees**.
- Tilt recovered from the integrated closure direction: **\(1.63\times10^{-16}\) degrees**.
- Conditional cap half-angle when \(\epsilon_v=0.5\,\mathrm{m/s}\), with no other error: **1.4607 degrees**.
- Invalid constant-force closure: \(|\|u\|-gT|=0\) despite true \(\|\Delta v\|=10.1526\,\mathrm{m/s}\) and **30 degrees** apparent tilt.
- Proper-coordinate-axis rotation check: residual \(\|u'-Qu\|=3.67\times10^{-14}\) in floating-point units.

This is a **deterministic identity/counterexample check**, not a model evaluation, uncertainty calibration, real sensor test or evidence of superior accuracy.

## 8. Mandatory first experiments

The proposed model should be rejected or narrowed if it does not beat a raw-data-corrected RIANN/TG-DCM/VQF/ESKF baseline on *actual verified-closure intervals*, if learned closure false positives cause out-of-domain high-tail regressions, or if calibration and bias make the cap uninformative. The full sequence, exact baselines, group-disjoint splits and stop criteria are in [experimental program](03_experiment_and_falsification.md).
