# A deep learning framework for robust attitude estimation in dynamic environments from six-axis raw inertial measurements

**Reference:** Wei Ding; Yansen Hu; Wei Sun; Zhe Yue. Measurement 2026 (published October), 2026. DOI: https://doi.org/10.1016/j.measurement.2026.122485. Primary source: https://www.sciencedirect.com/science/article/pii/S0263224126021949. 
**Applicability:** Six-axis tilt estimation. **Evidence accessed:** publisher_abstract_extended_preview. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Fixed-gain filters and gyro integration perform poorly under dynamic acceleration and raw-sensor distribution shifts.

## Architecture / algorithm / rationale

Residual temporal CNN feature extractor joined to bidirectional LSTM context; normalized quaternion output from raw 3-axis accel + 3-axis gyro, with bounded lookahead.

## Mathematical or identifiability mechanism

The network maps a time window a_{t-w:t+l}, omega_{t-w:t+l} to qhat_t; l>0 incurs algorithmic lookahead and must be accounted for when comparing causal systems.

## Experimental evidence actually verified

Publisher abstract claims median attitude error under 3 degrees and over 10% improvement against selected baselines; lacks same-split TG-DCM comparison or accessible full results here.

## Limitations and critical interpretation

Full text inaccessible; detailed loss, dataset splits, runtime and available code not verified. Bidirectionality prohibits treating the output as zero-lookahead without an explicit buffer policy.

## Actionable connection to TG-DCM-PINN

Highest-priority direct modern comparator: reproduce on same six-axis datasets at matched latency and same yaw-invariant tilt RMSE.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
