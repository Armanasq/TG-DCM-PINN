# MoRPI-PINN: a physics-informed framework for mobile robot pure inertial navigation

**Reference:** Arup Kumar Sahoo; Itzik Klein. Scientific Reports 16, 19827, 2026, 2026. DOI: https://doi.org/10.1038/s41598-026-50630-y. Primary source: https://doi.org/10.1038/s41598-026-50630-y. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** open_access_article_sections. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Pure 2D inertial integration is highly sensitive to noise and gyro/accelerometer bias.

## Architecture / algorithm / rationale

Encode 2D inertial navigation differential equations as training residuals alongside data and initial-condition constraints; exploit controlled snake-like robot motion.

## Mathematical or identifiability mechanism

L = lambda_data L_data + lambda_phys ||d xhat/dt - f_INS(xhat,u)||^2 + lambda_init L_init, with paper-specific term details and scaling to be checked in exact reproduction.

## Experimental evidence actually verified

Real robot study over six trajectories reports position error 4.2m vs 68.6m for 2D INS and 22.1m for MoRPINet, under the study's specific motion setup.

## Limitations and critical interpretation

Snake-like motion is a deliberate observability/denoising assumption; no direct evidence of general SO(3) attitude improvement and no public code evaluated here.

## Actionable connection to TG-DCM-PINN

A physically consistent, learnable update is preferable to unconstrained regression when physical state/assumptions are explicitly modeled.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
