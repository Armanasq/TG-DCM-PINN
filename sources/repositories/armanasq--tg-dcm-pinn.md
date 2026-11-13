# Implementation audit: Armanasq/TG-DCM-PINN

**Immutable revision:** e27a43a1d3b6b4d3a82355414523e25bc97ca370
**Source:** https://github.com/Armanasq/TG-DCM-PINN/tree/e27a43a1d3b6b4d3a82355414523e25bc97ca370
**License:** Apache-2.0
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

GRU 2x200, quaternion wxyz body->world. Tilt-only Charbonnier supervision and norm penalty, two gated auxiliary physics branches, EMA checkpoint. Source-level audit performed; trained results not recomputed.

## Files actually inspected

- [src/tg_dcm/losses.py](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/src/tg_dcm/losses.py)
- [src/tg_dcm/so3.py](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/src/tg_dcm/so3.py)
- [src/tg_dcm/train_impl.py](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/src/tg_dcm/train_impl.py)
- [src/tg_dcm/evaluation.py](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/src/tg_dcm/evaluation.py)
- [configs/model.yaml](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/configs/model.yaml)
- [docs/evaluation.md](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/docs/evaluation.md)
- [README.md](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/README.md)
- [training/summary.json](https://github.com/Armanasq/TG-DCM-PINN/blob/e27a43a1d3b6b4d3a82355414523e25bc97ca370/training/summary.json)

## Reproducibility / attribution caveats

Data and checkpoint inference not executed. Regressions found by code inspection require targeted tests.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
