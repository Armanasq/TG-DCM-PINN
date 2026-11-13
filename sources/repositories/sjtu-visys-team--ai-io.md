# Implementation audit: SJTU-ViSYS-team/AI-IO

**Immutable revision:** 5866134d5e9e20c082a32f06a59e99abb9049a15
**Source:** https://github.com/SJTU-ViSYS-team/AI-IO/tree/5866134d5e9e20c082a32f06a59e99abb9049a15
**License:** MIT
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

Temporal convolutions and IMUTransformerWithModality; diagonal log-standard-deviation probabilistic loss. Rotor speed is an extra input.

## Files actually inspected

- [src/learning/network/model.py](https://github.com/SJTU-ViSYS-team/AI-IO/blob/5866134d5e9e20c082a32f06a59e99abb9049a15/src/learning/network/model.py)
- [src/learning/network/losses.py](https://github.com/SJTU-ViSYS-team/AI-IO/blob/5866134d5e9e20c082a32f06a59e99abb9049a15/src/learning/network/losses.py)

## Reproducibility / attribution caveats

Not a six-axis-only comparator. Paper results and runtime not reproduced.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
