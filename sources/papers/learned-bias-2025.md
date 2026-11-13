# Learned IMU Bias Prediction for Invariant Visual Inertial Odometry

**Reference:** Abdullah Altawaitan; Jason Stanley; Sambaran Ghosal; Thai Duong; Nikolay Atanasov. 2025 arXiv preprint, 2025. Primary source: https://arxiv.org/abs/2505.06748. arXiv: 2505.06748. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** primary_full_text_sections. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Unknown, slowly drifting gyro/accelerometer bias contaminates integration and complicates invariant filter symmetry.

## Architecture / algorithm / rationale

Predict IMU bias from historical sensor windows using a learned model, then feed corrected measurements into an invariant MSCKF instead of adding bias as an ordinary group-state component.

## Mathematical or identifiability mechanism

omega_corr = omega_m - bhat_g(x_history), a_corr = a_m - bhat_a(x_history); geometric propagation remains separate from a learned nuisance-state estimate.

## Experimental evidence actually verified

Real visual-inertial tests include periods of visual outage; does not establish unassisted six-axis gravity-only yaw accuracy.

## Limitations and critical interpretation

VIO evaluation assumes images during some segments and a particular filter state/observability structure; bias estimates may entangle motion cues if not regularized.

## Actionable connection to TG-DCM-PINN

Explore slow-time-scale bias state and uncertainty without supervising spurious absolute yaw, with rest, thermal, and time-drift tests.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
