# 05 · Goals — falsifiable claims and decision-grade experiments

**Research objective:** estimate \(h_t=R_{WB,t}^{T}e_3\) from six-axis data with lower error and better failure behavior under unseen device/motion conditions. The task is **gravity-referenced tilt**, not unobservable absolute heading. Claims must survive data-protocol correction, same-input comparisons, training-seed variability and tail-risk reporting.

## 1. Define the estimand before optimization

For sequence \(s\), let \(\theta_{s,t}=\operatorname{atan2}(\|\hat h_{s,t}\times h_{s,t}\|,\hat h_{s,t}^{T}h_{s,t})\) in degrees and \(M_s\) its valid truth indices:

\[
r_s=\sqrt{\frac{1}{|M_s|}\sum_{t\in M_s}\theta_{s,t}^2}.
\]

A **sequence-mean** estimate is \(N^{-1}\sum_s r_s\). A **raw-motion mean** first aggregates segments of the same original recording. An **all-frame RMSE** weights long recordings and large segments more heavily. A dataset-balanced mean gives one top-level weight to each dataset. Report which one is estimated; do not interchange.

The released baseline reports standard-domain 137-sequence mean tilt RMSE 1.521° and \(\Delta_{\rm TG-RIANN}=-0.091^\circ\) (paired bootstrap interval [-0.147,-0.036]); DIODEM is 422 segments/86 original motions with mean 10.41°, median 2.70°, p95 44.20°, max 62.38°. Its grouped paired interval versus RIANN includes zero. These are **frozen reported results**, not independently recomputed in this work.

## 2. Priority claims and failure criteria

| ID | Hypothesis | Measurement | Necessary counterfactual | Fails when |
|---|---|---|---|---|
| H0 | Existing comparisons survive protocol correction | Paired tilt effect, raw startup, DIODEM tail | No VQF backfill, no full-SO(3) secondary alignment, valid-only GT and dt | Ranking or effect disappears, or incomplete data prevents reconstruction |
| H1 | Gyro/GT target gate helps for reasons other than excluding hard samples | Unconditional \(r_s\), gate coverage, p95 | Supervised-only, hard-gate, soft-gate, shuffled equal-occupancy mask | Benefit only on selected accepted frames |
| H2 | Gravity gate improves robustness beyond accidental rate/device cues | Error conditional on true motion, not \(\|f\|\) alone | Constant-norm false-gravity translation; known/unknown lever arm | False confidence or OOD tail worsens |
| H3 | Explicit bias state reduces tilt drift | Bias/time-conditioned tilt and confidence | Same GRU with realistic static/random-walk bias augmentation | No additional independent held-out benefit |
| H4 | S² correction beats quaternion regression at equal compute | Tilt p95/mean, convergence, manifold tests | Same parameters, same past/future context, same data | Geometry adds complexity without OOD advantage |
| H5 | Learned covariance is meaningful, not just broad | 2-D NEES, coverage/NLL/interval width, selective risk | Fixed robust covariance and calibrated ESKF | Cannot improve coverage/sharpness jointly or falsely trusts OOD |
| H6 | SO(3)-covariance improves *actual* remount robustness | Algebraic residual and new-device error | Rotation augmentation, full-state transform test | Internal layer passes but full estimator fails |
| H7 | Motion balancing improves the tail without label amplification | DIODEM raw-motion p95 and failed-motion rate | Raw-motion sampling, KISS-style motion GMM, randomized weights | Improvement vanishes when invalid GT and shared segments are controlled |

## 3. Stage 0 — repair experimental identifiability

**Deliverable:** immutable evaluation protocol, not a new accuracy number. Requirements:
- GT-invalid frames never score. All-GT-invalid recordings are excluded with reason; do not silently compare to identity.
- Raw source time retained; no fabricated positive dt in physics losses or evaluation. Clock calibration is frozen independently of model prediction.
- Quaternion interpolation only within uninterrupted valid GT segments; exact final/first valid timestamps supported.
- Equal initialization and **causal** startup. Remove VQF future backfill; retain separate common burn-in analysis.
- Secondary heading-aligned metrics use an SO(2) *yaw-only* correction or are explicitly relabeled full-orientation GT calibrated.
- RIANN, VQF and TG record exact dt; quantify fixed-dt versus per-sample-dt discrepancies before interpreting results.
- Per-sequence output includes original recording ID, sample rate, motion label, GT-valid fraction and source hash.

**Go/no-go:** after fixes, recompute the released checkpoint and baseline outputs. If published comparisons cannot be reconstructed, no architecture result should supersede them without first explaining the discrepancy.

## 4. Stage 1 — replace paper ranking with experimental comparability

Direct comparators: official RIANN, VQF6D, released TG GRU, supervised-only GRU and a faithfully implemented **six-axis** adaptive error-state Kalman filter such as [Wei 2025 RAESKF](https://doi.org/10.1016/j.measurement.2024.116097). Add [Ding 2026 CNN–BiLSTM](https://doi.org/10.1016/j.measurement.2026.122485) only with documented architecture, training source, window alignment and matched lookahead. [RCAE 2024](https://arxiv.org/abs/2401.13092) receives an external orientation measurement; odometry methods train on velocity/position. They are **not** direct headline baselines.

Every comparator gets identical available sensor channels, units, coordinate transform, first observation, valid GT mask, future-sample budget and per-frame prediction timing. If a filter has a legitimate warm-up requirement, report a separate fixed-time steady-state experiment for **all** methods.

## 5. Stage 2 — causal attribution rather than architectural storytelling

Use a factorial experiment with four otherwise identical GRU configurations:
1. Supervised tilt only.
2. Tilt + target-gated gyro.
3. Tilt + gravity confidence.
4. Full combined objective.

Then isolate hard versus soft gyro gate, each gravity factor (norm/jerk/rate), fixed vs mass-normalized confidence, explicit gyro bias augmentation and its temporal persistence. Run ≥5 independent seeds **per configuration**, with unchanged early stopping/hyperparameter search budget. Report branch gradient magnitude, accepted sample count, effective confidence mass and unconditional error by motion severity.

**Reason:** physics loss can improve without changing observable state error, especially when confidence gates suppress high-error dynamics. A positive paired effect of *the full method* cannot establish that every component is beneficial.

## 6. Stage 3 — mechanism-centered novelty candidate

Prioritize a small causal tilt observer with:
- explicit slow-time gyro bias estimate, with stationarity/excitation assumptions;
- SO(3) gyro propagation mapped into a 2-D tangent correction on S²;
- physically justified acceleration compensation only when lever arm and angular-acceleration quality are known;
- bounded SPD 2×2 tangent covariance evaluated for calibration.

Literature controls include [RSS continuous bias NODE 2025](https://arxiv.org/html/2504.09495), [Brändle 2025](https://arxiv.org/abs/2502.03681), [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784), [EqNIO](https://arxiv.org/html/2408.06321), and [GINIO](https://arxiv.org/abs/2609.25338). Those individual mechanisms are prior art; novelty must lie in a precisely new derivation, guarantee, interaction or experimentally verified operating regime.

## 7. Statistics: three levels of uncertainty, no single magic confidence interval

**Level A — test sample:** paired bootstrap over independent sequences or raw motions, fixed trained checkpoint. This is what released paired intervals primarily address.

**Level B — training process:** independent random seeds, checkpoints and hyperparameter selection. A fixed-checkpoint bootstrap cannot represent this variation. Treat seed as a separate random factor.

**Level C — deployment domain:** held-out devices, mounting configurations, session conditions and motion regimes. Even multiseed confidence does not imply external domain generalization if all runs share one sensor family.

Report all three where claims concern deployment. Use original raw motion as the resampling unit for DIODEM and include both segment-weighted and equal-raw-motion estimands. For 26 paper-derived hypotheses, avoid fishing among many models on the final test; use validation-only selection or a multiplicity-aware confirmatory plan.

## 8. Research gates and outputs

| Milestone | Required artifact | Scientific decision |
|---|---|---|
| M0 | Protocol diff, unit tests and new original-raw-motion data manifest | Are existing results still comparable? |
| M1 | Same-input benchmark with raw per-frame predictions and fixed delay | Is there a genuinely stronger direct competitor? |
| M2 | Seeded loss-component ablations and per-motion gate statistics | Does target-gating cause an unconditional gain? |
| M3 | S²+bias+covariance candidate, OOD and adversarial tests | Is improvement mechanism-specific and identifiable? |
| M4 | Physical remount/lever-arm sensitivity | Does symmetry hold beyond artificial coordinate rotation? |
| M5 | Source/checkpoint/data hashes and machine-reconstructed tables | Is the claim reproducible? |

**Success is not “best average.”** It is a **replicable improvement on a preregistered metric without an unacceptable DIODEM tail regression**, with known applicability conditions, calibrated uncertainty if claimed, and a coherent runtime sensor/latency contract.
