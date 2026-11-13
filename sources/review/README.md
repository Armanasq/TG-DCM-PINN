# Deep technical research review: 2024–2026

**Project:** TG-DCM-PINN, gravity-referenced attitude from a six-axis IMU. **Evidence cutoff:** 2026-11-13. **Coverage:** **26 curated post-2023 publications** with individual paper notes, and **10 source repositories** with commit-pinned inspection logs (including the target). **Status:** literature/mathematics and static code review completed; external experiments have **not** been independently reproduced.

## Read in this order

| Chapter | Scope | Key deliverable |
|---|---|---|
| [01 — Architectures](01_architectures.md) | Current filter, direct GRU/CNN–BiLSTM, learned inertial measurement, SO(3) networks, MoE | Side-by-side method and input-budget matrix |
| [02 — Designs](02_designs.md) | Frame/time contracts, data lifecycle, gates and uncertainty, streaming and evaluation | Design rules and matched-baseline protocol |
| [03 — Mathematics](03_mathematics.md) | SO(3), S² gauge, quaternion losses, gyro kinematics, noise, covariance, equivariance | Actual TG-DCM objective and constructed identifiability counterexample |
| [04 — Challenges](04_challenges.md) | Ambiguity, acceleration, gyro bias, OOD, GT corruption, long-tail, publication artifact gaps | Failure mechanisms and adversarial tests |
| [05 — Goals](05_goals.md) | Research objectives, estimands, metrics and falsification gates | Testable hypotheses and milestone decisions |
| [06 — Limitations](06_limitations.md) | Fundamental, paper-specific and target source code limitations | Claim boundary and evidence grading |
| [07 — Future work](07_future_works.md) | Prioritized novel hypotheses and dependencies | Mechanism, assumptions, experiments and reject criteria |
| [08 — Comprehensive literature review](08_literature_review.md) | Chronological 2024/2025/2026 primary-method review and critical synthesis | Detailed comparative research chapter, citations and reproducibility program |

## Canonical provenance

- [Paper registry](../papers/index.json) — 26 deduplicated entries; each includes title, authors, DOI/arXiv, evidence-access status, research problem, mathematics, claims, limitations and its individual Markdown note in [papers/](../papers/).
- [Inspected implementation index](../repositories/index.json) — 10 repositories with immutable commit SHA, exact Python/C++ files examined, and licensing notes when available.
- [Target source inspection](../implementation/target-audit.md) — released TG-DCM computations and code-level tests to implement.
- [External source inspection](../implementation/external-code-audit.md) — code-versus-paper boundaries and author-reported errata.
- [Experimental design](../synthesis/experimental-design.md) — proposed reproducibility, grouped statistics, paired effects, latency and ablation controls.
- [Research methodology](../methodology.md) — inclusion, review evidence levels, cross-source verification and legal-access boundaries.
- [Acquisition tool](../tools/acquire.py) — reproducible retrieval plan for downloadable sources and checked-out revisions; not proof of files already acquired.

## Interpretation standard

The research objective is yaw-invariant tilt, not unobservable absolute yaw. **Do not compare** position/velocity trajectory ATE with sequence tilt RMSE, a bidirectional window with a causal single-sample estimator at different delay, or IMU-only inference with an aided training and runtime sensing setup without disclosure. The current evidence contains *inspected code*, *source-derived geometry* and *author-reported experimental metrics*; it does **not** contain independent confirmation of third-party paper results, cloned original repositories, new TG-DCM training runs or full licensed-paper mirrors.

## Most important conclusions

1. Sensor/frame/timing and label integrity must be fixed before interpreting small neural gains. The target [SLERP interpolation](../../src/tg_dcm/so3.py) can bridge invalid reference gaps and reject exact final timestamps; its [physics loss](../../src/tg_dcm/losses.py) clamps nonpositive dt before physically validating time intervals.
2. Accelerometer magnitude, jerk and angular-rate confidence **cannot guarantee** valid gravity under arbitrary translation. The [mathematics chapter](03_mathematics.md) provides an explicit constant-magnitude/zero-jerk false-gravity construction.
3. The strongest new direct baseline is the [2026 six-axis CNN–BiLSTM](https://doi.org/10.1016/j.measurement.2026.122485), but its future context and quaternion/yaw supervision require a matched evaluation. Adaptive 6D ESKF is a complementary classical comparator.
4. [EqNIO](https://arxiv.org/html/2408.06321), [GINIO](https://arxiv.org/abs/2609.25338), [KISS-IMU](https://arxiv.org/html/2603.06205) and [MosaicIMU](https://arxiv.org/html/2606.09355) contribute transferable symmetry, confidence, motion balance and domain-generalization mechanisms. Their published odometry accuracy is **not direct evidence** of tilt superiority; GINIO's own code repository also documents paper-to-checkpoint discrepancies.
5. The most defensible new research track is a **causal S² tilt observer with explicit gyro bias, calibrated two-dimensional uncertainty and optional physically supported rotational-acceleration compensation**, but this is a hypothesis requiring experiments and exhaustive close-prior-art checks.

## Maintenance

Keep the eight chapters synchronized by stable links into the canonical paper and repository registries. Expand evidence **only after** retrieving/inspecting the new primary source and annotating access status; never silently reinterpret a publisher abstract as full-text evidence. For every claimed improvement supply source commit, dataset hashes/partitions, model checkpoint SHA256, per-sequence outputs, and raw-motion-level confidence intervals. Preserve negative findings and quality-control counterexamples.
