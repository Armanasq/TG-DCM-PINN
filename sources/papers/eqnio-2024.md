# EqNIO: Subequivariant Neural Inertial Odometry

**Reference:** Royina Karegoudra Jayanth; Yinshuang Xu; Ziyun Wang; Evangelos Chatzipantazis; Daniel Gehrig; Kostas Daniilidis. 2024 arXiv; ICLR 2025, 2024. Primary source: https://arxiv.org/abs/2408.06321. arXiv: 2408.06321. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** primary_full_text_sections_plus_code. **Code inspected:** RoyinaJayanth/EqNIO.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

The same pedestrian motion expressed in different yaw-aligned sensor frames otherwise forces a learned odometry network to memorize redundant directions.

## Architecture / algorithm / rationale

Build gravity-aligned equivariant vector and invariant scalar representations; learn an O(2)-subequivariant canonical frame and rotate IMU features into it; regress displacement/covariance in canonical coordinates then transform predictions back.

## Mathematical or identifiability mechanism

For gravity-preserving Q, mu(QX)=Q mu(X) and Sigma(QX)=Q Sigma(X) Q^T. Under reflections, gyroscope angular velocity is axial rather than polar, requiring the appropriate parity handling.

## Experimental evidence actually verified

Paper evaluates TLIO, Aria, RIDI, OxIOD and a RoNIN-derived model; repository implements vector-neuron network primitives and probabilistic displacement loss. Odometry ATE is not attitude tilt RMSE.

## Limitations and critical interpretation

Requires a gravity-aligned reference; the choice of subgroup and canonicalization can hide sensitivity to inaccurate estimated vertical. Its specific O(2) symmetry is not automatically arbitrary SO(3) remount equivariance.

## Actionable connection to TG-DCM-PINN

Derive a causal SO(3)-covariant tilt correction on S^2 without needing an externally supplied attitude estimate; use exact remount tests.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/RoyinaJayanth/EqNIO, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
