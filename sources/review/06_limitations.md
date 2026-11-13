# 06 — Limitations: empirical claims, identifiability and implementation evidence

**Purpose:** document limitations that must survive the synthesis even if they weaken proposed novelty. **Reviewed range:** 2024–2026, with earlier background methods marked historical. See [challenges](04_challenges.md), [goals](05_goals.md), [full review](08_literature_review.md).

## 1. Distinguish four evidence classes

A **theoretical property** may be proved from a correctly defined observation model; an **author-reported experiment** may be valid but not independently replicated; an **inspected code property** follows from source-level control flow; and a **reproduced result** requires execution with matched checkpoint, dataset and metric. Do not convert one evidence type into another.

A preprint DOI or public GitHub link does not guarantee accessible exact training data, numeric tables, full architecture or working checkpoints. The source corpus metadata labels access as primary sections, publisher abstract or code inspected. No 2024–2026 externally published accuracy result was independently rerun in this review.

## 2. Fundamental limits of inertial inference

**Absolute yaw is not identified** by gravity plus gyro readings without extra reference or restrictive motion assumptions. A full quaternion network may predict a convention-dependent heading learned from datasets, not globally observable yaw. Reference yaw-dependent geodesic loss can be unstable under unknown device mount/initial heading, even if it improves numbers on a synchronized benchmark. S² tilt loss avoids this *heading* nuisance but cannot resolve arbitrary translational acceleration.

**Arbitrary linear acceleration is unseparable from gravity** at one sample without a motion prior, known external state, or other measurements. A constant-magnitude vector tilted by 30° from gravity can have \(\|f\|=g\), jerk zero and gyro rate zero while dynamic acceleration produces all the difference. The current confidence gate's three inputs then all appear maximally reliable; the exact counterexample is in [math](03_mathematics.md#8-a-constructive-non-identifiability-counterexample).

**Known rigid-body compensation is conditional.** [Arzberger–Nüchter](https://arxiv.org/html/2607.25784) can subtract alpha×r and omega×(omega×r) if calibration and lever arm are known/estimated. It does not remove arbitrary reference-point translation or eliminate differentiating angular rate noise. [Brändle 2025](https://arxiv.org/abs/2502.03681) shows that angular acceleration changes the filter feedback dynamics, yielding nonminimum-phase issues under some lever-arm geometry; filter gain reduction is a bandwidth-versus-disturbance tradeoff, not a universal fix.

**Aids change the problem.** [RCAE 2024](https://arxiv.org/abs/2401.13092) requires measured attitude; [Tchonkeu 2026](https://arxiv.org/abs/2607.13254) requires barometric data for its stated convergence setting; [Zhai 2026](https://doi.org/10.1016/j.measurement.2025.119824) uses angular-accelerometer fusion; [AI-IO 2026](https://arxiv.org/abs/2603.00597) uses rotor speed; [KISS-IMU](https://arxiv.org/html/2603.06205) trains from LiDAR-generated labels. None is a strict raw-six-axis baseline by default.

## 3. Limitations of TG-DCM-PINN, established by its own released results

| Constraint | Evidence | Consequence |
|---|---|---|
| Only one selected training seed / run; no controlled component ablations | [training/summary.json](../../training/summary.json) and [README](../../README.md) | Cannot assign aggregate gain causally to specific gate, loss or optimizer |
| Standard split 137-sequence mean gain small | 1.521° tilt RMSE; mean paired -0.091° vs RIANN; CI [-0.147,-0.036] | Supports this sampled mean, not uniform superiority, device-invariance or a novel principle |
| DIODEM heavy tail | 422 segments/86 motions, mean 10.41°, median 2.70°, p95 44.20°, maximum 62.38° | Rare severe failures dominate deployment risk despite modest central tendency |
| DIODEM mean difference inconclusive | Paired clustered RIANN delta -0.195°, 95% [-0.440,+0.043] | 95% interval includes zero; do not claim resolved improvement on DIODEM |
| Correlated DIODEM segments | Five-ish segments derived from shared raw motions | Treat 86 raw motions as bootstrap clusters; do not claim 422 independent sequences |
| Domain-specific optimization and data split | Myon/TUM-VI train, external datasets test; multi-frequency resampling | Training diversity does not guarantee all mounts, rates or distributions |
| No runtime physics-constrained update | Inference GRU emits quaternion only | Training-time consistency may fail OOD at runtime |
| Static norm/jerk/rate gravity trust | [losses.py](../../src/tg_dcm/losses.py) | Cannot distinguish every gravity-ambiguous acceleration |
| Confidence loss denominator = count, not weight mass | [losses.py](../../src/tg_dcm/losses.py) | Effective branch gradient scale changes with confidence weights; not necessarily a bug |
| 2° GT target gate | [losses.py](../../src/tg_dcm/losses.py) | Excludes label-discrepant and potentially difficult fast transitions from dynamics supervision |
| No independently verified released raw-data re-execution | Dataset not redistributed | Published results are not yet reproduced in this review |

## 4. Target implementation defects that affect evidential integrity

**Interpolation across invalid GT.** The [interpolate_quat](../../src/tg_dcm/so3.py) implementation compacts finite rows, then selects adjacent entries of the compact array. If original data contain valid–NaN–valid, the middle query may receive a smooth SLERP orientation despite absent source truth. A nominal “gap-aware” description is not an adequate substitute for gap-segment tests. The exact last timestamp of a valid sequence is also excluded by the current right-bound check. These are direct control-flow findings; their effects on the released result tables are **unknown until datasets are regenerated and comparisons rerun**.

**Invalid dt accepted after clamp.** In [TargetDCM3Loss.forward](../../src/tg_dcm/losses.py), raw dt is clamped before the finite target pair filter. Finite nonpositive dt thus becomes 1e-6 and may enter apparent gyro dynamics and acceleration jerk calculations. Verify path for NaN/Inf explicitly, then reject nonpositive intervals without silently modifying physics.

**Ground-truth-dependent offset.** The [time-offset optimizer](../../src/tg_dcm/so3.py) selects a shift by minimizing GT-vs-gyro propagation disagreement. This can be an appropriate calibration method, but if used independently on test truth it must be disclosed and held identical for all baselines; otherwise the benchmark may become conditional on post hoc test-label alignment. An explicit contamination claim requires tracing the actual preparation/evaluation calls.

**Artifact identity ambiguity.** The currently accessible repository path is Armanasq/TG-DCM-PINN, while [README](../../README.md) badges and [pyproject.toml](../../pyproject.toml) homepage identify GinkgoQ/TG-DCM-PINN. Track which source, tag, branch, release and checkpoint SHA were actually evaluated.

**Test gaps.** Current [tests/test_losses.py](../../tests/test_losses.py) covers basic quaternion vertical, zero qexp, selected dynamics gate and gravity magnitude cases. It does not cover the failure modes above, counterexamples under \(\|a\|=g\), covariance behavior, group equivariance or raw-motion resampling. The absence of tests is not a demonstration that every edge case fails, but means the claims lack regression protection.

## 5. Limitations of direct contemporary attitude studies

**Wei et al. 2025 RAESKF:** the publisher accessible summary describes covariance switching/compensation and relative pitch/roll gains. Full equations, test devices, rate, computational cost and raw trajectories were not accessible in this review, so its exact comparability and reproducibility remain open. Claims of a fixed percent improvement cannot be transferred to different hardware.

**Ding et al. 2026 six-axis CNN–BiLSTM:** primary abstract/extended publisher preview describes three architecture features and median errors; detailed methods and code unavailable here. A BiLSTM may use future samples, and full-quaternion geodesic supervision depends on yaw treatment. Its median angle is **not** the same statistic as TG-DCM sequence-level tilt RMSE. Do not claim an exact winning model from the current evidence.

**Ye et al. 2025 gyro transformer:** public abstract reports a mean orientation error 1.56° for the proposed method versus 1.49° for Gyro-Net, while describing the proposed result as better. If both are lower-is-better degrees for the same metric, this is arithmetically inconsistent. It might reflect a typographical error or different evaluation context; without full per-sequence outputs and method definitions, neither interpretation may be silently selected. [Primary source](https://doi.org/10.1002/rob.70082).

**Oveissi et al. 2024 retrospective-cost:** orientation observations are a core input; treating it as pure accelerometer+gyro performance would be a sensor-budget error. [Source](https://arxiv.org/abs/2401.13092).

## 6. Limitations of transferable odometry and representation methods

**EqNIO:** its O(2) symmetry assumes a gravity-aligned input supplied from a filter; it is not a proof of general arbitrary remount robustness when gravity itself is inaccurate. Distinguish reflections from proper rotations: gyro is an axial vector and must receive det(Q) parity. [Paper](https://arxiv.org/html/2408.06321), [inspected implementation](https://github.com/RoyinaJayanth/EqNIO/tree/97b7a60a5e5cc8f0132bba343c83de114211f827).

**GINIO:** the original authors' [KNOWN_ISSUES.md](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) is a serious evidence warning:
- Spectral/NLL published NLL -1.875, NEES 2.974 do not correspond to the known measured source; audit says NLL -0.8747 and NEES 6.774.
- GINIO-RS row labeled body+LFA uses a checkpoint trained without LFA.
- GINIO-E NanoBench score is from a TLIO-trained model evaluated zero-shot while another comparison row was NanoBench-trained; the training protocol was not labeled accordingly.
- With-attitude Log(R) does not satisfy correct physical remount equivariance even when its vector-neuron core is equivariant.
- Some rotation-sweep cells have no regenerating script or mix checkpoints from different epochs.

These are **claims in the repository authors' own erratum worklist**, not independently verified by rerunning their historical servers. They invalidate treating the printed metrics as transparently reproducible, but do not invalidate all GINIO architectural results.

**TartanIMU:** the paper reports several model/transfer features. Public [transformer.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py) explicitly raises NotImplementedError because the Transformer core is not in the inspected LSTM-only release. Its architecture cannot be assumed replicated from this code.

**MosaicIMU:** in its [paper](https://arxiv.org/html/2606.09355), an average ATE gain coexists with a quadruped test case worse than TLIO; authors cite unseen carriers, severe bias changes, unusual mounting and unassisted early adaptation as residual challenges. [Code](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) contains BiLSTM heads; true streaming delay/latency remains to be measured with an explicit output policy. A zero-lookahead claim would be unsupported.

**KISS-IMU:** pseudo-targets use LiDAR ICP/PGO at training, so “self-supervised” is not “no external modality ever”; pose-graph and ICP failure transfer to the labels. [Paper](https://arxiv.org/html/2603.06205).

**DPI-LSTM and MoRPI-PINN:** authors report strong positional metrics on bounded UAV or specific mobile robot trajectories using initial conditions and motion assumptions. These are not demonstrations of fully unassisted global position observability nor of a superior yaw-free tilt estimator. [DPI-LSTM](https://doi.org/10.3390/s26134287), [MoRPI](https://doi.org/10.1038/s41598-026-50630-y).

## 7. Methodological and review-scope limitations

This corpus is a **deep curated technical review**, not a claim to have enumerated every relevant paper after 2024. Scholarly search indexes may omit preprints, append articles not matching DOI publication years, return stale versions or omit negative studies. Paywalled publisher pages may expose only abstract and metadata. arXiv HTML conversion can omit figures/appendices. GitHub connector inspection confirms code content at named revisions but not installed dependencies, runtime tests, checkpoint availability, full training invocation, or matching paper numbers. No external repository was cloned locally or trained during this phase; network-connected acquisition remains a separate repeatable command.

Additional uncertainties to resolve: confirm all author/DOI metadata, verify any “best” percentage against complete per-sequence records, match reviewer-independent real datasets, inspect paper appendices and licensing for copied full text, and benchmark direct 6D attitude competitors at common delay. Keep every unknown visible instead of replacing it with optimistic assumptions.

## 8. Claim boundary

The currently supported TG-DCM contribution claim is a **training objective with target-gated gyro consistency and confidence-weighted gravity alignment**, evaluated against RIANN/VQF on reported splits. A small standard-domain mean improvement is documented. Component attribution, universal physical consistency, accurate absolute yaw, all-domain dominance, confidence calibration, and novelty relative to every 2024–2026 related method **are not established**.

The scientifically useful result of the review is a prioritized falsification program, not an unconditional assertion of novelty. See [goals](05_goals.md) and [future research](07_future_works.md).
