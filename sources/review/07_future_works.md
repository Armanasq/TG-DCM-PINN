# 07 · Future work — mechanisms worth testing, not feature accumulation

**Research criterion:** a candidate needs (a) an explicit information assumption, (b) a mathematical or computational mechanism distinct from prior art, (c) a failure regime it uniquely addresses, and (d) a controlled result that could refute it. The following are **hypotheses**, not demonstrated innovations.

## 0. First deliverable: an evidence-correct baseline

**Not new research until repaired:** invalid GT interpolation, nonpositive dt handling, 0×NaN masked losses, no-GT identity scores, VQF future backfill, full-SO(3) “heading” alignment and mixed per-sample/fixed dt baseline semantics. Source references and counterexamples: [design](02_designs.md), [math](03_mathematics.md), [limitations](06_limitations.md).

**Reproduction:** freeze raw data, checkpoint and existing result tables. Implement minimal regression tests; regenerate reference targets and per-sequence predictions; publish a protocol-difference table. Recompute standard 137 and DIODEM 422/86 comparisons. Do not describe a protocol correction as a model improvement.

## H1 · Conditional observability-aware S² filter

**Gap:** full quaternion regression spends output capacity on a heading gauge and does not provide an inference-time physically constrained update. A standard ESKF models rotation but can give too much accelerometer trust under motion.

**Proposed mechanism:** retain \(h\in S^2\) and a slowly varying \(b_g\). Propagate \(h^-=\exp(-[\omega_m-b_g]_\times\Delta t)h\); let a causal model predict tangent innovation \(\xi\in\mathbb R^2\) and SPD tangent covariance. Update by \(\operatorname{Exp}_{h^-}(B(h^-)\xi)\), with an **observability/confidence mode** that can withhold gravity correction when acceleration cannot be separated from gravity. Couple an explicit initial-tilt prior to a valid calibration protocol.

**Novelty risk:** S² geometry, learned corrections and covariance are established. A genuinely new claim requires a precise observability-aware fusion rule or a formal/calibrated gain guarantee not already provided by geometric filters/[EqNIO](https://arxiv.org/html/2408.06321)/[GINIO](https://arxiv.org/abs/2609.25338).

**Falsification:** matched-parameter same-input GRU and 6D ESKF, five seeds, yaw-gauge/coordinate tests, constant-norm false gravity, random bias, OOD device, DIODEM p95, calibrated 2-D NEES and runtime. Reject if covariance broadens without improving error or if the filter fails to recover after an unobservable interval.

## H2 · Learn bias as a physical nuisance process, not a motion label

**Gap:** gyro bias, sensor noise and motion-conditioned prediction error are entangled in recurrent hidden states. A network can appear to “debias” by recognizing the carrier rather than identifying the instrument.

**Prior art:** [Liu et al., RSS 2025](https://arxiv.org/html/2504.09495) learn deterministic continuous bias dynamics via a causal spline and Lie-ODE, using **pose** GT rather than bias GT, and assume initial bias; [Altawaitan 2025](https://arxiv.org/abs/2505.06748) predict nuisance bias inside invariant VIO. The [2026 empirical bias preprint](https://doi.org/10.31224/7144) shows strong label/dataset sensitivity in its own two-dataset study.

**Proposed distinction:** parameterize a low-bandwidth bias state with bounded derivative and learn it from a *device-invariant* sensor history, without using carrier ID, with explicitly controlled resets. Preserve a separate stochastic residual. Add an observability flag when stationary/rotation excitation does not support a bias update.

**Falsification:** constant and random-walk bias, changing calibration, rest/rotation and temperature proxy, leave-one-device-out. Compare ordinary bias augmentation, direct gyro denoiser and the RSS NODE's **initialization/inference assumptions**. A successful pose drift reduction alone does not prove unbiased bias-state identification; verify bias against calibrated bench data when available.

## H3 · Mechanistic motion-acceleration compensation with robust fallback

**Gap:** the present norm/jerk/rate gate may reject valid gravity during fast rotation and falsely accept constant-magnitude translation. [Brändle 2025](https://arxiv.org/abs/2502.03681) identifies rotational-acceleration feedback dynamics; [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784) compensate known/estimated lever-arm effects.

**Proposal:** if r and calibrated angular acceleration are available, compute \(f_{\rm corr}=f_m-\alpha\times r-\omega\times(\omega\times r)\), propagate a variance term reflecting uncertainty in r, omega and alpha, then update tilt only if corrected-force consistency is credible. If no lever-arm calibration exists, do not make an invented compensation; fall back to an explicitly lower-trust observation.

**Falsification:** fixed r=0/known r/perturbed r, pure rotation/translation and controlled angular acceleration, phase delay, worst-case error/recovery. A method that improves only on the nominal lever-arm calibration but fails on plausible miscalibration has not solved robust gravity use.

## H4 · Two separate confidence models

**Gap:** ground-truth consistency gates control **training label quality**, while gravity confidence controls **runtime sensor information**. Mixing them produces selection-biased objectives and misleading statements such as “the physics loss is small, therefore the estimator is physical.”

**Proposal:** use a hard *reference-validity* mask generated by source metadata and a differentiable *measurement-noise* model trained against held-out tilt innovations. Explicitly constrain coverage/uncertainty, and report accepted fraction, effective sample size \(N_{\rm eff}\), branch gradient mass and unconditional error over the full test distribution.

**Falsification:** if gating gains disappear at equal accepted-frame count, or if improved NLL is achieved solely by increasing covariance, the claim is unsupported. Examine whether a fixed classical covariance schedule or robust M-estimator achieves the same performance at lower complexity.

## H5 · Symmetry-constrained estimation at the **system** level

**Gap:** [GINIO](https://arxiv.org/abs/2609.25338) develops SO(3)-equivariant network modules, but its [author-side issues](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) show that an added attitude representation can violate full physical-remount equivariance. [EqNIO](https://arxiv.org/html/2408.06321) similarly requires an already aligned gravity frame.

**Proposed mechanism:** define one exact group action on raw gyro, force, bias, tilt, hidden features, tangent basis and covariance. Prove that propagation and learned correction commute with a **coordinate change** \(Q\in SO(3)\). If reflections are used, represent axial angular rate with det(Q) parity. Then test physical remount separately, accounting for sensor displacement and calibration changes.

**Falsification:** algebraic group residual beyond numerical tolerance at any pipeline boundary; poorer held-out remount performance than a strong rotation-augmented GRU at the same latency/parameters. Do not call physical relocation an exact symmetry.

## H6 · Long-tail motion risk without amplifying bad labels

**Gap:** DIODEM's p95 sequence RMSE is 44.20° while median is 2.70°. Ordinary global mean minimization emphasizes frequent easy motions; uncritical hard-example sampling may emphasize mislabeled or unobservable motions.

**Proposal:** cluster *training-only* motion features (angular speed/acceleration, force spectrum, motion family, sensor identity withheld from prediction) with a KISS-style GMM. Sample rare **valid** motions more, but use reference-quality masking independently. Predeclare a tail-risk objective, such as minimizing raw-motion p95 under a guardrail on standard mean and equal compute.

**Falsification:** per-raw-motion tail does not improve, worst device worsens, or benefits disappear when labels are audited. Distinguish segment-weighted and equal-motion error.

## H7 · Causal multi-rate temporal features instead of a larger Transformer

**Gap:** short impulses and sustained bias have different timescales. A two-layer GRU may conflate them; however greater window length and future information can masquerade as model superiority.

**Proposal:** a small strictly causal multirate TCN front end plus compact stateful filter/bias head, with the same sensor channel and future-lookahead budget as the GRU. Compare separate explicit time scales to [Ding's 2026 frequency-aware CNN–BiLSTM](https://doi.org/10.1016/j.measurement.2026.122485), [iMoT](https://doi.org/10.1609/aaai.v39i6.32664), [IONext](https://arxiv.org/abs/2507.17089) and [MosaicIMU](https://arxiv.org/html/2606.09355).

**Falsification:** gain disappears under a same-delay architecture control, 95th-percentile time-to-output worsens, or the improvement occurs only when future frames are supplied.

## Prioritized decision sequence

| Step | Experimental output | Stop condition |
|---|---|---|
| 0 | Corrected data/evaluation source and reproduced baseline | Cannot regenerate per-sequence published results |
| 1 | Same-input ESKF, RIANN, VQF, GRU, direct CNN–BiLSTM at fixed delay | No genuine baseline separation |
| 2 | Seeded loss and gate-coverage ablation | Components do not yield unconditional benefit |
| 3 | Bias + S² tangent observer, separately measured uncertainty | No stable OOD/tail or coverage gain |
| 4 | Conditional lever-arm compensation and group-law tests | Fragile to plausible calibration or remount |
| 5 | Raw-motion tail balancing and hardware-held-out study | Mean improves but worst motions/devices deteriorate |
| 6 | Exact paper-to-commit/data/checkpoint/metric reconstruction | Any headline result lacks a generating artifact |

**Scientific contribution threshold:** establish a mechanism with narrower assumptions than a generic end-to-end model, prove a property or demonstrate its controlled effect, publish a counterexample/failure boundary, and make every number independently regenerable. Combining existing physics, gating and equivariance blocks without these controls is not novelty.

## New prerequisite: falsify the published baseline before proposing another architecture

**F0 — Caruso identity audit:** compare exact frame-aligned arrays and SHA256 for 18 corrected/orig pairs. Since \(q\mapsto\pm q\) preserves tilt, any remaining difference requires an explicit change to target validity, frame conversion, clock, sensor input or artifact version. Recompute independent-recording CIs.

**F1 — canonical raw benchmark:** evaluate EuRoC twice (raw vs reference-bias-debiased gyro), as *two distinct measurement budgets*. Cross-dataset aggregation must never silently mix them when supporting unassisted deployment.

**F2 — DIODEM causal decomposition:** recover original 11-session identifiers and published rigid/nonrigid flags. Trace each 422-entry HDF5 to original IMU and optical marker records, sensor-to-body transform, sync calibration and excluded-questionable policy. Quantify how much p95 error is explained by attachment type versus actual estimator sensitivity.

**F3 — own-estimator selection test:** compare live vs EMA selected at the same full-sequence streaming metric, then measure resets, warm-up and near-unobservable motion.

**F4 — prior-art differentiation:** [RSS 2025 bias dynamics](https://doi.org/10.15607/RSS.2025.XXI.009), [2025 optical-flow body-gravity observer](https://arxiv.org/abs/2508.21163), [2026 reduced gravity observer](https://arxiv.org/abs/2609.26612), and [2026 nonlinear geometric VIO observer](https://arxiv.org/abs/2601.07156) already establish bias dynamics and body-gravity geometric state under their own sensor assumptions. A new six-axis S² observer needs a specific novel update/uncertainty/identifiability result and independent positive tests, not a new name.

Prioritize F0–F3 over adding Transformer, MoE or more loss terms. [Evidence and sensitivity analysis](09_second_pass_verification.md).


## Highest-priority new hypothesis: finite-window motion closure

[Candidate method](../proposals/01_closure_constrained_tilt.md) derives an S² gravity constraint from **bounded sensor-point velocity change** over a window, avoiding an instantaneous near-gravity assumption. A causal learned six-axis event head may propose windows but **cannot certify** true closure under unrestricted translation. The mathematical containment theorem is strictly conditional on an independently justified bound.

The key test regime is **high intermediate acceleration with equal but nonzero endpoint world velocity**, against an ESKF with exactly the same motion-closure event and preintegrated measurement. Standard preintegration, ZUPT and set-valued attitude observers are strong prior art: [novelty audit](../proposals/02_prior_art_and_novelty.md). A synthetic test confirms interval cancellation despite 41.88-degree peak instantaneous apparent gravity error, while an invalid-closure example returns 30-degree false tilt despite zero integrated norm residual. Neither is real model performance.

**Decision:** finish the P0 benchmark/data audit; measure genuine sensor-velocity closures in independently instrumented dynamic recordings; only then train a closure event detector. If true episodes are rare or out-of-domain false positives dominate, reject or narrow the method. [Experiments](../proposals/03_experiment_and_falsification.md).
