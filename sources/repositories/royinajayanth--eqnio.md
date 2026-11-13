# Implementation audit: RoyinaJayanth/EqNIO

**Immutable revision:** 97b7a60a5e5cc8f0132bba343c83de114211f827
**Source:** https://github.com/RoyinaJayanth/EqNIO/tree/97b7a60a5e5cc8f0132bba343c83de114211f827
**License:** not verified (no root license present in inspected tree)
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

O(2)-equivariant vector/scalar channels, canonical frame modules, ResNet/TLIO integration, covariance-based probabilistic displacement loss.

## Files actually inspected

- [TLIO-master/src/network/model_eq_VNCnn_frame_TLIO_v2_2vectors.py](https://github.com/RoyinaJayanth/EqNIO/blob/97b7a60a5e5cc8f0132bba343c83de114211f827/TLIO-master/src/network/model_eq_VNCnn_frame_TLIO_v2_2vectors.py)
- [TLIO-master/src/network/model_eq_o2.py](https://github.com/RoyinaJayanth/EqNIO/blob/97b7a60a5e5cc8f0132bba343c83de114211f827/TLIO-master/src/network/model_eq_o2.py)
- [TLIO-master/src/network/losses.py](https://github.com/RoyinaJayanth/EqNIO/blob/97b7a60a5e5cc8f0132bba343c83de114211f827/TLIO-master/src/network/losses.py)

## Reproducibility / attribution caveats

model_eq_o2.py is a transform assert demonstration, not the trainable estimator. TLIO-master/src/dataloader/LICENSE exists but does not establish the repository root license.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
