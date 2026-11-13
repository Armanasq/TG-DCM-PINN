# Implementation audit: haleqiu/AirIMU

**Immutable revision:** 119fe7ce229e321b3890282909a183aefad61dfb
**Source:** https://github.com/haleqiu/AirIMU/tree/119fe7ce229e321b3890282909a183aefad61dfb
**License:** BSD-3-Clause
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

PyPose IMUPreintegrator with covariance propagation; SO(3) logarithm residual loss and combined rotation/velocity/position objectives.

## Files actually inspected

- [model/net.py](https://github.com/haleqiu/AirIMU/blob/119fe7ce229e321b3890282909a183aefad61dfb/model/net.py)
- [model/losses.py](https://github.com/haleqiu/AirIMU/blob/119fe7ce229e321b3890282909a183aefad61dfb/model/losses.py)

## Reproducibility / attribution caveats

Original AirIMU arXiv research began in 2023, so indexed as historical implementation baseline, not a qualifying 2024+ new paper.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
