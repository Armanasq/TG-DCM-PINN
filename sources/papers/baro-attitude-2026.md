# Attitude Estimation Using Inertial and Barometric Measurements

**Authors:** Melone Nyoba Tchonkeu; Soulaimane Berkane; Tarek Hamel. **Year:** 2026. **Venue:** arXiv 2026.
**Primary source:** https://arxiv.org/abs/2607.13254. **DOI:** https://doi.org/10.48550/arXiv.2607.13254.
**Classification:** transfer. **Access actually achieved:** primary_abstract_and_metadata. **Code inspected:** none. **Independent reproduction:** no. **Reviewed:** 2026-11-13.

## Research objective and original problem

Highly accelerated flight makes tilt from accelerometer and gyro insufficiently constrained when velocity measurements are unavailable.

## Architecture and algorithmic design

Deterministic Riccati observer cascaded with a complementary filter, analyzed for almost-global asymptotic stability under uniform observability; nonlinear SO(3)xR^2 observer combines IMU inputs with barometer and magnetometer outputs for local exponential stability under stated conditions.

## Mathematical or information-theoretic formulation

The observer evolves on SO(3) with auxiliary vertical-motion states and extra sensed pressure. Observability and convergence hold only under the particular stated measurement channels and excitation conditions.

## Evidence actually available

Primary arXiv abstract reports simulated and real-flight evaluation and formal AGAS/LES claims; no independent proof check.

## Limitations and alternative interpretation

Barometer and, in the second formulation, magnetometer are additional sensors. Neither can be used to assert absolute unassisted six-axis tilt observability.

## Relevance and falsifiable transfer to TG-DCM-PINN

Use as comparison of sensing assumptions and stability proof techniques; exclude from strict six-axis baseline.

## Provenance

See [full 2024–2026 synthesis](../review/08_literature_review.md) and [pinned repository index](../repositories/index.json). Publisher/HTML/abstract content was reviewed with the stated access constraint; no large original paper asset or external model checkpoint was installed locally.
