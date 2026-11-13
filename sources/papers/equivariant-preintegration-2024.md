# Equivariant IMU Preintegration With Biases: A Galilean Group Approach

**Year:** 2024. **Authors:** Giulio Delama; Alessandro Fornasier; Robert Mahony; Stephan Weiss. **Venue:** IEEE Robotics and Automation Letters 10(1), 2025; first arXiv 2024. **Primary source:** https://arxiv.org/abs/2411.05548. **DOI:** https://doi.org/10.1109/LRA.2024.3511424. **Access:** peer_reviewed_pdf_full_text_extracted_truncated. **Independent execution:** none.

## Research objective

Preintegration and bias propagation benefit from a consistent Galilean group structure rather than decoupling bias in Euclidean coordinates.

## Method, mathematical mechanism and assumptions

Discrete-time bias-aware equivariant preintegration on the Galilean tangent-group construction, with coupled state/bias errors and covariance/Jacobian propagation; tested against classical formulations.

Group-based preintegrated navigation state with bias-aware Lie error; bias Jacobian and covariance propagated within the chosen invariant error coordinates.

## What was actually verified

Full open-access manuscript text inspected via scholarly extraction; numerical comparisons are author-reported and not independently executed.

## Limitations and hidden observation dependencies

Relies on established inertial mechanization and often additional pose/velocity factors in downstream navigation. Our velocity-closure integral is not novel preintegration.

## Relevance to proposed original research

Adopt tested bias Jacobians/error propagation when changing a rolling preintegrator; do not claim group-consistent bias propagation as new.

[Novelty audit](../proposals/02_prior_art_and_novelty.md) · [Proposed method](../proposals/01_closure_constrained_tilt.md)
