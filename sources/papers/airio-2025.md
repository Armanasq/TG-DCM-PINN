# AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability

**Reference:** Yuheng Qiu and collaborators. IEEE Robotics and Automation Letters 2025, 2025. DOI: https://doi.org/10.1109/LRA.2025.3581130. Primary source: https://doi.org/10.1109/LRA.2025.3581130. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** paper_sections_plus_code. **Code inspected:** Air-IO/Air-IO.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Rotating IMU acceleration into a gravity/world frame can discard informative body-motion interactions in aerial dynamics.

## Architecture / algorithm / rationale

Learn body-frame motion using temporal CNN + recurrent features; compare versions with explicit attitude input, predict velocity and heteroscedastic uncertainty and fuse through an estimator.

## Mathematical or identifiability mechanism

Rather than collapsing the input to world-frame specific force, body-frame a and omega are retained, and a learned covariance conditions downstream filter updates.

## Experimental evidence actually verified

Authors' ablations and published aerial trajectory results; inspected actual CNNEncoder, CodeNetMotion/CodeNetMotionwithRot and training/evaluation entry points.

## Limitations and critical interpretation

An attitude-aided network is not six-axis standalone without explaining how attitude is obtained; reported translation metrics are not tilt metrics, and sensor remounting is not automatically equivariant.

## Actionable connection to TG-DCM-PINN

Compare body-frame temporal context with gravity-aligned processing and uncertainty-conditioned updates; avoid importing an unavailable attitude input.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/Air-IO/Air-IO, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
