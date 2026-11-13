# Implementation audit: superxslam/TartanIMU

**Immutable revision:** 8a8025000f85d2c6b0692b88e15c2c91e761f7d7
**Source:** https://github.com/superxslam/TartanIMU/tree/8a8025000f85d2c6b0692b88e15c2c91e761f7d7
**License:** Apache-2.0
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

ResNetLSTMSeqNet and per-platform heads for dog/human/car/drone; Transformer builder is registered but raises NotImplementedError due intentionally unreleased core.

## Files actually inspected

- [tartan_imu/model/backbones/lstm.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/lstm.py)
- [tartan_imu/model/backbones/transformer.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py)

## Reproducibility / attribution caveats

Transformer results cannot be independently reproduced from the public release as inspected.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
