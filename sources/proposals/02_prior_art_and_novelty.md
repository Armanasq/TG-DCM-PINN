# Prior-art and novelty audit — closure-constrained tilt

**Goal:** prevent rebranding standard preintegration or set-membership theory as a new scientific contribution. **Review date:** 2026-11-13. **Target:** [CCTO candidate](01_closure_constrained_tilt.md). **Evidence classes:** peer-reviewed full text, arXiv full text, publisher abstract only, inspected public code, and own analytical derivation. All performance reported for external work is author-reported.

## 1. The important predecessors, in order of novelty threat

| Reference | Primary mechanism/observations | Relation to CCTO | What we cannot claim |
|---|---|---|---|
| [Brás et al., 2013, Systems & Control Letters](https://doi.org/10.1016/j.sysconle.2013.06.008) | Set-valued SO(3) attitude and gyro bias observer with bounded vector-sensor disturbances; guarantees under stated bounds | Explicit feasible attitude sets and bounded uncertainty are not new | “First certified set-based attitude filter” |
| [Periodic/quasi-periodic inertial navigation, Sensors 2011](https://www.mdpi.com/1424-8220/11/6/5931) | Motion-cycle priors and drift reduction for recurring trajectories | Periodicity assumptions in inertial navigation precede learning | “First periodic inertial-closure idea” |
| [Forster et al., 2017, TRO](https://doi.org/10.1109/TRO.2016.2597321) | On-manifold IMU rotation, velocity and position preintegration between keyframes | CCTO's integrated measurement \(u_{ij}\) is the conventional preintegrated velocity factor | “New IMU integration equation” |
| [Mitikiri & Mohseni, 2021, Automatica](https://doi.org/10.1016/j.automatica.2021.109494) | Geometric attitude solutions for a single reference direction and angular rate; hard constraints | Single-gravity-vector geometric correction and constrained attitude estimation are established | “New geometric single-vector observer” |
| [ZUPT methods, established; Mohammadkarimi et al. 2025](https://doi.org/10.1038/s41598-025-25054-9) | Velocity equality to zero in stationary alignment; error-state observability and calibration limits | Endpoint constraints are classical; interval velocity closure is a less restrictive **assumption** than zero-velocity endpoints, not a newly invented principle | “First use of velocity closure for attitude” |
| [Delama et al., IEEE RA-L 2024/2025](https://doi.org/10.1109/LRA.2024.3511424) | Galilean-group equivariant preintegration coupling biased states and covariance | Exact equivariance and bias-aware preintegration are established | “First equivariant biased preintegrator” |
| [Huai 2026](https://arxiv.org/abs/2605.28279) | Propagation and preintegration are equivalent realizations with consistent Jacobians/covariances | Fast rolling/cumulative preintegration is engineering, not fundamental invention | “Preintegration-versus-propagation equivalence is new” |
| [Brändle 2025](https://arxiv.org/abs/2502.03681) and [Arzberger & Nüchter 2026](https://arxiv.org/abs/2607.25784) | Motion acceleration and lever-arm compensation in gravity attitude filters | Requires sensor-point closure or calibrated mechanical geometry | “Dynamic motion compensation from gyro is new” |
| [Liu et al., RSS 2025](https://doi.org/10.15607/RSS.2025.XXI.009) | Controlled neural ODE learns time-varying IMU bias using pose supervision | Explicit learned bias dynamics is established | “Learned gyro bias is a new contribution” |
| [EqNIO 2024](https://arxiv.org/abs/2408.06321), [GINIO 2026](https://arxiv.org/abs/2609.25338) | Gravity-aligned subequivariance and SO(3)-equivariant inertial pseudo-measurements | State/feature transformation laws are already established | “SO(3) equivariance alone is innovative” |
| [RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097), [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) | Adaptive gravity filter; axis-separated CNN–BiLSTM with frequency encoding and bounded lookahead | Direct comparison on raw six-axis tilt is required | “Pure GRU plus physics loss already dominates” |
| [Chen et al., Measurement 2026](https://doi.org/10.1016/j.measurement.2026.121394) | Single-foot IMU time-domain zero-velocity detector and EKF/ZUPT | Learning/detecting zero-velocity intervals is a mature subproblem | “First IMU-only closure detector” |
| [Liu et al., Measurement 2026](https://doi.org/10.1016/j.measurement.2025.118667) | GLRT and learned autoencoder foot-motion classification for ZUPT | Explicit precedent for learning a motion event from IMU and applying inertial correction | “First learned ZUPT gate” |
| [GaRLILEO 2025 preprint/2026 journal](https://arxiv.org/abs/2511.13216) | Radar + leg kinematics reconstruct velocity and soft S² gravity factors | Aided gravity/velocity constraints are strong competing prior art; different sensor budget | “S² gravity factor is a newly derived representation” |
| [DIODEM descriptor, 2025](https://doi.org/10.1038/s41597-025-05468-w) | 11 continuous experiments, rigid/foam IMUs, optical markers and segment transform | Necessary physical velocity/attachment truth to validate closure candidates | “422 independent dynamic tests” |

**Evidence caution:** Brás et al. 2013 and some publisher papers were available as publisher methodology/abstract, not complete independently re-derived equations. Forster/preintegration and Delama were independently cross-checked through peer-reviewed/open text; no external algorithm was executed.

## 2. Why the previous best idea is insufficiently distinctive

The prior [future-work list](../review/07_future_works.md) prioritized "S² tangent update + gyro bias + covariance". The 2025–2026 geometric-observer literature and earlier bias-aware filters already contain each mechanism. [Bouazza 2025](https://arxiv.org/abs/2508.21163), [Abi 2026](https://arxiv.org/abs/2609.26612), [Boughellaba 2026](https://arxiv.org/abs/2601.07156) model body-frame gravity/velocity on geometric manifolds; they rely on visual measurements, so their stability cannot be transferred to raw six-axis data. But their state choice is prior art.

Likewise, a set-valued S² cone with learned covariance alone is a weak novelty claim given the 2013 set-valued SO(3)/bias observer. The **differentiating problem** is not the representation; it is gaining a valid gravity observation from **bounded endpoint velocity change without demanding instantaneous near-gravity acceleration**, then deciding how to *select, abstain and evaluate* that event without fake physical certainty.

## 3. A narrow research contribution hypothesis

**Candidate novelty claim, subject to citation-chained verification:**

> A causal, multi-horizon six-axis tilt observer that converts intermittent bounded **sensor-point velocity-closure hypotheses** into conditional spherical gravity feasibility constraints, then couples these with a learned-but-explicitly-uncertified closure selector and a conservative no-update fallback; it is evaluated against standard ZUPT-aided ESKF, preintegration and learned attitude baselines under matched sensing, latency, sensor-mount and raw-recording protocols.

Potential distinct features if actually demonstrated:
- Closure windows need only \(\|\Delta v_s\|\le\epsilon_v\), **not** both endpoint velocities equal to *zero*, and need not have near-gravity single-frame acceleration.
- Each admitted window generates an explicit physically interpretable **cap width** from bounded closure and integration errors; a learned model is prevented from turning an invalid physical assumption into an unconditional “guarantee.”
- A causal multi-window policy can abstain during non-identifiable motion rather than hallucinate precise gravity with a norm/jerk gate.
- Proper SO(3) coordinate-frame consistency follows for the physical construction; full-model equivariance must be separately measured.
- A rigorous benchmark specifically includes false-closure, constant-magnitude acceleration, sensor-attachment and repeated-session confounders missing from the target's released study.

**This is not yet an established novel method.** The integrated velocity observation is a direct rearrangement of the classic preintegration equation. A reviewer could reasonably call the entire model a ZUPT-based observer with learned event detection. To clear that objection, show a genuinely distinct admission mechanism, falsifiable uncertainty behavior, **measurable improvement in nonstationary equal-velocity intervals where classical ZUPT does not activate**, and repeatable gains across devices. A theoretical claim beyond algebraic rewriting would require proving a new property under credible bounded bias/event assumptions.

## 4. Novelty defense by precise reviewer challenge

| Likely reviewer objection | Necessary answer | No-go conclusion |
|---|---|---|
| "This is just preintegration plus a zero-velocity update." | Test intervals where start/end sensor velocities are equal but **nonzero** and intermediate \(\|a_W\|\) is large; compare direct closure-factor ESKF using the *same* event labels. Show selective update adds more than an ordinary factor. | No improvement over equal-information classical factor |
| "Your neural closure detector uses an unobservable velocity condition." | Explicitly separate teacher-aided training from IMU-only inference; measure OOD false-closure precision and confidence calibration; no deterministic guarantee from the classifier. | False-closure mistakes dominate gains |
| "Hard angular bounds are arbitrary." | Independently calibrate sensor error, bias drift and endpoint velocity uncertainty; propagate worst-case bounds; report radius/coverage jointly. | Bounds so loose they cannot improve tilt |
| "You leak optical truth into test inputs." | Two tracks: independently verified physical closure (extra event information, aided), and learned six-axis closure (pure runtime IMU). Do not mix them or use EuRoC reference-estimated gyro bias in raw trials. | Source-assisted inputs used under an unaided label |
| "Your statistics are based on duplicated sequences." | Deduplicate Caruso, cluster DIODEM by motion and 11 continuous experiment sessions, hold out devices and complete acquisition sessions, repeat ≥5 seeds. | Gain disappears under valid independence assumptions |
| "It is only an offline smoother." | Use only data at timestamps <= output timestamp, measure lookback and compute delay separately, bound ring-buffer size. | Requires future labels or hindsight to select windows |
| "It is the standard manifold observer." | Prove something stronger about conditional set containment *and* show a distinct valid-event admission strategy with OOD benefit. | No new theorem or empirical benefit beyond known methods |

## 5. Alternative research tracks and ranking

| Candidate | Distinctiveness from literature | Direct impact on observed TG weaknesses | Principal risk | Priority |
|---|---|---|---|---|
| Source-verified benchmark repair + domain-failure analysis | Possible strong **benchmark/methodology** contribution; not inherently new estimator | High: duplicate recordings, reference bias and invalid alignment | Requires raw HDF5 and ground-truth reconstruction | **P0 prerequisite** |
| Conditional velocity-closure tilt observer with robust event admission | **Plausibly distinct combination**, not proven novel | High for repeatable/closed high-acceleration motions | Valid closure detection, device transfer and error-bound tightness | **P1 research hypothesis** |
| S² tangent observer + slow bias + SPD covariance | Core elements well established | Moderate | Incremental feature combination/weak novelty | P2 baseline/ablation |
| Set-valued pure attitude observer | Direct pre-2024 prior art | Moderate safety value | No new theorem or accepted-observation mechanism | P2 comparator |
| More recurrent layers, BiLSTM/Transformer, MoE | Mature 2024–2026 architectures | Uncertain on DIODEM tails | Future-lookahead and overfit motion priors | P3 only after rigorous baseline |
| Universal self-supervised gravity from arbitrary six-axis IMU | Contradicts no-restriction identifiability | None without extra assumptions | Scientifically impossible as stated | **Reject** |

## 6. What must be checked before stating novelty

A structured forward/backward citation expansion is still needed around velocity closure, *nonzero equal-velocity* constraints, sliding-window inertial gravity alignment, ZUPT event classifiers, reachable-set observers and model-risk coverage. Searches here located relevant classical and contemporary methods but do not establish absence of an identical algorithm in journals, patents, theses or supplementary code. The public novelty status must remain **"candidate, prior-art risk unresolved"** until that check and working experimental validation.

The smallest publishable claim is likely *operating-regime-specific*: robust dynamic tilt during known/reliably detected repetitive motion using only six-axis measurements at inference and accurate uncertainty about when the correction is admissible. Avoid "general attitude estimation without accelerometer assumptions" or "provably reliable learned closure detection" as unsupported claims.
