# Research candidate: closure-constrained tilt observer

**Status:** candidate research question, not proven novel or publishable. **Reviewed through:** 2026-11-13. **Task:** six-axis IMU at inference, gravity-direction tilt, with a possible teacher for motion-event labels during training. No target model source code or released results are changed by these proposals.

| Artifact | Contents |
|---|---|
| [01 — Method and mathematics](01_closure_constrained_tilt.md) | Sensor-point velocity closure, interval gravity observation, S² feasible cap, conditional containment proof, two counterexamples, causal algorithm and limitations |
| [02 — Prior-art/novelty audit](02_prior_art_and_novelty.md) | 2013 set-valued SO(3), manifold/Galilean preintegration, ZUPT, learned event detection, existing bias/geometry: explicit boundaries and reviewer objections |
| [03 — Falsification and experiments](03_experiment_and_falsification.md) | Raw-data integrity, oracle-vs-learned event study, adversarial acceleration, same-information baselines, grouped statistical testing, go/no-go |
| [Numerical identity/counterexample check](../tools/verify_motion_closure.py) | Minimal deterministic NumPy experiment; checks high-acceleration closure, false-closure residual and proper body-frame covariance |

## Proposed contribution boundary

A velocity-closure observation can recover or bound body gravity direction over a **whole interval** without assuming each acceleration sample approximates gravity. Existing inertial preintegration and ZUPT already contain the algebra. The research opportunity, **not yet demonstrated**, is causal and reliable event admission, *honest uncertainty when the event cannot be verified*, and an improvement beyond the classical ESKF given **exactly the same event and preintegration input**.

The method must not claim certified performance from a learned closure detector alone: without trusted endpoint-velocity information or a validated physical bound, the false-closure case is indistinguishable from a valid gravity observation under unrestricted motion. Pure six-axis inference is distribution dependent in that regime.

**Prerequisite:** [second-pass TG benchmark audit](../review/09_second_pass_verification.md) must be repaired before an improvement claim. The original 137-file fixed-checkpoint effect is sensitive to Caruso duplicates, DIODEM segments are nested in 11 sessions, and EuRoC contains reference-estimated gyro bias preprocessing.

**Result of the included test:** idealized dynamic closure recovers vertical numerically despite 41.88-degree instantaneous apparent-force tilt; a false interval with zero norm residual produces 30-degree wrong tilt. These are analytical/synthetic demonstrations, **not real-data estimator results**.
