# A robust adaptive error state Kalman filter for MEMS IMU attitude estimation under dynamic acceleration

**Reference:** Xiaofeng Wei; Shiwei Fan; Ya Zhang; Wei Gao; Feng Shen; Xie Ming; Jian Yang. Measurement 242 (2025); 2024 preprint, 2025. DOI: https://doi.org/10.1016/j.measurement.2024.116097. Primary source: https://www.sciencedirect.com/science/article/pii/S0263224124019821. 
**Applicability:** Six-axis tilt estimation. **Evidence accessed:** publisher_abstract_only. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Acceleration beyond gravity corrupts roll/pitch correction in conventional inertial attitude filters.

## Architecture / algorithm / rationale

Nominal direction-cosine matrix plus small error-state filter; adaptive disturbance handling switches between explicit dynamic-acceleration compensation and measurement-noise covariance adjustment according to monitored motion thresholds.

## Mathematical or identifiability mechanism

R_true approximately R_nom exp([delta theta]x). The EKF updates a local error state instead of adding nine independent DCM components; dynamic acceleration changes measurement residual reliability.

## Experimental evidence actually verified

Publisher abstract reports at least 42.76% pitch and 67.21% roll improvement in selected dynamic experiments, but these are authors' scenario-specific comparisons, not results on the TG-DCM benchmark.

## Limitations and critical interpretation

Full methods and data tables not acquired; threshold sensitivity, motion-distribution shift and covariance-consistency properties remain unverified; no implementation inspected.

## Actionable connection to TG-DCM-PINN

Implement as direct six-axis baseline with identical preprocessing, calibration, per-sequence tilt metrics and time-window latency.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
