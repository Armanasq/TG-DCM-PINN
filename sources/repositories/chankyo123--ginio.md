# Implementation audit: chankyo123/ginio

**Immutable revision:** 6b97f178be1a253ef44e9a05ed17432b41c00287
**Source:** https://github.com/chankyo123/ginio/tree/6b97f178be1a253ef44e9a05ed17432b41c00287
**License:** MIT
**Method:** remote code read via connected GitHub repository file API; local clone and tests not executed.

## Code-level implementation

Vector-neuron linear/convolutional features and GRU; scalar gates from rotation-invariant feature norms; vector mean and spectral covariance output; numerical equivariance tests.

## Files actually inspected

- [airio/model/vn_gru.py](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/vn_gru.py)
- [airio/model/vn_codenet.py](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/vn_codenet.py)
- [airio/model/test_a2_spectral_equivariance.py](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/airio/model/test_a2_spectral_equivariance.py)
- [docs/KNOWN_ISSUES.md](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md)

## Reproducibility / attribution caveats

The repository's own KNOWN_ISSUES.md disputes several camera-ready table cells and configuration labels; treat as an author-side erratum, not independently rerun measurements.

Any file not listed has not been fully reviewed. An accessible tree does not establish a reproduced training pipeline, confirmed dependencies, complete experiment archive, or correct paper metric.
