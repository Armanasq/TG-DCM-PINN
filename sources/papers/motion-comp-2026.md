# Motion-Acceleration Calibration and Compensation in IMUs without External Equipment for Attitude Estimation Filters

**Reference:** Fabian Arzberger; Andreas Nüchter. 2026 arXiv preprint, 2026. Primary source: https://arxiv.org/abs/2607.25784. arXiv: 2607.25784. 
**Applicability:** Six-axis tilt estimation. **Evidence accessed:** primary_html_sections. **Code inspected:** fallow24/ros_imu_calib.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

When IMU is not at the rotation center, tangential and centripetal acceleration contaminate the apparent gravity direction and confidence gating alone discards potentially recoverable information.

## Architecture / algorithm / rationale

Calibrate accelerometer/gyro scale, bias, axis misalignment, IMU-to-pivot lever arm and optionally multi-IMU extrinsics from multiposition/static-and-rotation maneuvers; subtract modeled rotational accelerations online.

## Mathematical or identifiability mechanism

a_sensor = a_origin + alpha x r + omega x (omega x r). Given r and estimates of omega and alpha, remove the last two terms before gravitational tilt correction.

## Experimental evidence actually verified

Primary arXiv HTML contains derivations, simulation and real spherical-mobile-system experiments; authors link open-source code and calibration data at fallow24/ros_imu_calib. This particular code was not inspected.

## Limitations and critical interpretation

Needs fixed/known pivot model over calibration motion; differentiating gyro to obtain alpha amplifies noise; cannot remove arbitrary translational acceleration from IMU alone.

## Actionable connection to TG-DCM-PINN

Add calibrated deterministic rotational acceleration compensation ahead of a learned confidence gate, and report the assumptions and failure modes.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/fallow24/ros_imu_calib, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
