# Implementation audit: sparolab/KISS-IMU

**Immutable revision:** 572bfbe29e90bcff55dc1ddf90205f071851f818
**Source:** https://github.com/sparolab/KISS-IMU/tree/572bfbe29e90bcff55dc1ddf90205f071851f818
**License:** BSD-3-Clause
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

GaussianMixture for motion components, sample weights, EMA covariance weight controller, PyPose IMU integration.

## Files actually inspected

- [src/models/gmm.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/models/gmm.py)
- [src/training/covweight.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/covweight.py)
- [src/training/integrator.py](https://github.com/sparolab/KISS-IMU/blob/572bfbe29e90bcff55dc1ddf90205f071851f818/src/training/integrator.py)

## Reproducibility / attribution caveats

LiDAR supplies training pseudo-labels; runtime performance and reported numeric ATE not rerun.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
