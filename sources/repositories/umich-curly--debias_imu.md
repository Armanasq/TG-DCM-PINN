# UMich-CURLY/Debias_IMU — code-level source evidence

**Pinned revision:** 7eb2afcaba71d27c9c340850c19158f33c8b2f69
**Original work:** [RSS 2025, Debiasing 6-DOF IMU via Hierarchical Learning of Continuous Bias Dynamics](https://doi.org/10.15607/RSS.2025.XXI.009)
**Code license:** root LICENSE is MIT. **Review level:** source modules and publication text, no local clone or executed training.

| File inspected | Implementation detail | Research implication |
|---|---|---|
| [network_for_BiadDy.py](https://github.com/UMich-CURLY/Debias_IMU/blob/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy/network_for_BiadDy.py) | Independent gyro/accel bias ODE vector-field networks; controlled SO(3) propagation state | Error correction is explicit bias dynamics, not attitude regression |
| [Interpolation.py](https://github.com/UMich-CURLY/Debias_IMU/blob/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy/Interpolation.py) | Backward-difference cubic Hermite coefficients and derivative/evaluation | Causal interpolation if only past control samples are used |
| [lie_algebra.py](https://github.com/UMich-CURLY/Debias_IMU/blob/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy/lie_algebra.py) | SO(3) exponential/logarithm, Jacobians, quaternion utilities | Rotation integration respects a local Lie chart; branch singularities require tests |
| [learning.py](https://github.com/UMich-CURLY/Debias_IMU/blob/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy/learning.py) | Gyro stage trained against pose rotation then accelerometer stage against pose-derived v/p, ODE integration | No direct bias GT necessary, but pose GT and window initial conditions remain |
| [dataset.py](https://github.com/UMich-CURLY/Debias_IMU/blob/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy/dataset.py) | Dataset and initial-state construction context | Validate bias initialization/clock source before transferring |

**Behind-the-scenes limitations:** the paper's short training windows approximate bias initial values by pose differences, introduces a teacher dependence despite “no bias labels,” and assumes plausible static initial calibration. The repository README says TUM-VI uses raw IMU values but timestamps derived from its calibrated bag. Reproducing cross-hardware claims requires separating these calibration and timing inputs. The recorded code revision is immutable; no claim of successful solver execution or author-number replication is made.
