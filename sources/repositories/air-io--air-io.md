# Implementation audit: Air-IO/Air-IO

**Immutable revision:** 7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73
**Source:** https://github.com/Air-IO/Air-IO/tree/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73
**License:** BSD-3-Clause
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

Temporal CNNEncoder with Conv1d/BatchNorm/GELU/Dropout, variants with and without attitude features, body-frame motion and covariance predictions; training uses GT orientation in with-attitude branch.

## Files actually inspected

- [model/code.py](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/model/code.py)
- [train_motion.py](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/train_motion.py)

## Reproducibility / attribution caveats

Distinguish attitude-assisted experiments from six-axis-only operation; reproduction requires datasets and checkpoint matching.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
