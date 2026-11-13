# On the effects of angular acceleration in orientation estimation using inertial measurement units

**Reference:** Felix Brändle; David Meister; Marc Seidel; Robin Strässer; Frank Allgöwer. IEEE CCTA 2025, 2025. Primary source: https://arxiv.org/abs/2502.03681. arXiv: 2502.03681. 
**Applicability:** Six-axis tilt estimation. **Evidence accessed:** primary_full_text_sections. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Low-cost IMU orientation estimation fails during rotational acceleration, even when gravity is predictable at the body origin.

## Architecture / algorithm / rationale

Derive transfer functions including accelerometer displacement from a rotation center, then analyze Mahony/Madgwick feedback and nonminimum-phase operating regimes.

## Mathematical or identifiability mechanism

Rigid-sensor acceleration contains alpha x r + omega x (omega x r). Linearized transfer zeros can enter the right half plane depending on lever arm and operating point; lowering feedback gain trades tracking bandwidth for disturbance attenuation.

## Experimental evidence actually verified

Analytical model, numerical/experimental study of tracking vs bandwidth, with explicit orientation-filter consequences.

## Limitations and critical interpretation

Linearization and platform geometry determine when predictions apply; does not provide a general learned attitude estimator or globally optimal filter.

## Actionable connection to TG-DCM-PINN

Include lever-arm-controlled dynamic acceleration and closed-loop bandwidth in evaluation; mere accelerometer norm gating is insufficient.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
