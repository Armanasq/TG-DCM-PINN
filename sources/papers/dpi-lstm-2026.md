# A Physics-Informed Dual-Branch LSTM Network for UAV Position and Attitude Estimation

**Reference:** Authors as listed in journal DOI. Sensors 26(13), 2026, 2026. DOI: https://doi.org/10.3390/s26134287. Primary source: https://doi.org/10.3390/s26134287. 
**Applicability:** Inertial attitude and position. **Evidence accessed:** open_access_full_text_inspected_partly. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Pure regression of UAV position and orientation ignores rigid-body kinematic coupling and drift-inducing physical inconsistency.

## Architecture / algorithm / rationale

Shared temporal LSTM encoder with separate position and quaternion attitude branches; kinematic losses include attitude propagation, coordinate transform, velocity-position integration and initial-state constraints.

## Mathematical or identifiability mechanism

Use quaternion kinematics q_{t+1} approximately q_t tensor Exp(omega_t Delta t / 2) and p_{t+1}=p_t+v_t Delta t, with additional world/body acceleration transformation. Published loss specifics require separate reproduction.

## Experimental evidence actually verified

Open-access full text obtained through scholarly connector; UZH-FPV Drone Racing evaluations, position RMSE 0.0654 m in stated experiment, plus ablations. The paper explicitly assumes dataset-provided initial reference state and neglects explicit online gyro drift compensation.

## Limitations and critical interpretation

Absolute position cannot be maintained from an unconstrained six-axis IMU without additional assumptions/prior; short-sequence, initialized evaluation is not a proof of arbitrary-duration drift-free inertial navigation. Position RMSE is not tilt.

## Actionable connection to TG-DCM-PINN

Use branch-specific constrained losses only where auxiliary state variables and reference initialization are available and auditable.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
