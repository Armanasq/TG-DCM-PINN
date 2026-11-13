# Research review · 2024–2026

**Target:** TG-DCM-PINN six-axis, yaw-invariant tilt. **Cutoff:** 2026-11-13. **Corpus:** [28 papers](../papers/index.json), [11 pinned implementation reviews](../repositories/index.json). **Evidence class:** primary paper or abstract, static inspected code, analytical derivation, and separately marked reproduced results. Third-party metrics have **not** been reproduced.

| Chapter | Research question |
|---|---|
| [01 · Architectures](01_architectures.md) | Which designs add an observable measurement, which add only an inductive bias, and which silently use more sensors or future frames? |
| [02 · Design](02_designs.md) | Can a benchmark compare methods when startup, GT alignment, and timestamps differ? |
| [03 · Mathematics](03_mathematics.md) | What does six-axis physics identify, and where do the exact losses/masks fail? |
| [04 · Challenges](04_challenges.md) | Which failure mechanism dominates, and what experiment distinguishes causes? |
| [05 · Goals](05_goals.md) | What falsifiable improvements and statistical estimands justify research claims? |
| [06 · Limitations](06_limitations.md) | What is mathematically impossible, source-level hazardous, or not experimentally verified? |
| [07 · Future work](07_future_works.md) | Which mechanism-specific hypotheses deserve a controlled trial, and when should they be rejected? |
| [08 · Critical literature review](08_literature_review.md) | How do 2024–2026 papers actually work, what assumptions buy their reported gains, and what does the code release support? |

## Non-obvious findings

**Benchmark:** VQF startup is replaced by later orientation; “heading” alignment uses GT-derived **full SO(3)** correction in secondary roll/pitch; RIANN and TG use different timestep representations; all-invalid GT can be scored against identity. The primary raw yaw-invariant tilt metric is separate from the GT-aligned secondary metrics.

**Loss:** masked residuals can remain NaN because 0×NaN=NaN; finite zero-norm quaternion is not valid SO(3); target-consistency gating selects a conditional subset of physical transitions; confidence-weighted gravity loss is normalized by accepted-frame count, so its effective coefficient changes with mean confidence.

**Physics:** accelerometer magnitude equal to gravity and zero jerk do not establish a correct gravity direction. An S² estimator cannot eliminate arbitrary linear-acceleration ambiguity or global yaw gauge.

**Research evidence:** a 2025 RSS explicit-bias Neural ODE still uses pose GT and pose-derived bias initialization; a 2026 *preprint* reports label/data quality dominating architecture in its specific bias experiments. GINIO's own issue log identifies paper-to-artifact mismatches. None of these results should be generalized beyond its observation model.

## Audit references

[Target source findings](../implementation/target-audit.md) · [External code inspection](../implementation/external-code-audit.md) · [Research experiment protocol](../synthesis/experimental-design.md) · [Access and provenance rules](../methodology.md) · [Acquisition script](../tools/acquire.py).

**Workflow:** repair experiment integrity → run identical direct six-axis baselines → seeded ablations → device/motion-held-out stress → evaluate bounded S²/bias/uncertainty proposals. Do not equate a lower conditional loss, a stronger odometry ATE, or a more complex backbone with novel tilt estimation.
