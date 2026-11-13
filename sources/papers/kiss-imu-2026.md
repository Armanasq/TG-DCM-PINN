# KISS-IMU: Self-supervised Inertial Odometry with Motion-balanced Learning and Uncertainty-aware Inference

**Reference:** Jiwon Choi; Hogyun Kim; Geonmo Yang; Juhui Lee; Younggun Cho. ICRA 2026/preprint, 2026. Primary source: https://arxiv.org/abs/2603.06205. arXiv: 2603.06205. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** primary_full_text_sections_plus_code. **Code inspected:** sparolab/KISS-IMU.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Collecting exact navigation ground truth across motion types is costly; common slow motions dominate training and rare rapid turns fail.

## Architecture / algorithm / rationale

Use LiDAR ICP registration and pose-graph optimization as pseudo-supervision during training; motion feature Gaussian mixtures balance training and adaptive uncertainty weighting governs inference.

## Mathematical or identifiability mechanism

For motion-component k with empirical density p(k), increase relative weight of underrepresented k subject to clipping; correct the neural integration using uncertainty-based factor weights.

## Experimental evidence actually verified

Authors demonstrate quadruped and other motion datasets; inspected sklearn GaussianMixture implementation, mixture-based weighting, covariance weight controller, and PyPose integrator.

## Limitations and critical interpretation

Not sensor-only self-supervision during training: LiDAR supplies the target; teacher ICP failure can bias pseudo-labels. Translational error metrics do not establish tilt improvement.

## Actionable connection to TG-DCM-PINN

Use motion-component balancing and per-motion robustness assessments, with correct group-level split to avoid near-duplicate leakage.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/sparolab/KISS-IMU, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
