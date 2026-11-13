# Nonlinear Observer Design for Visual-Inertial Odometry

**Authors:** Mouaad Boughellaba; Abdelhamid Tayebi; James R. Forbes; Soulaimane Berkane. **Year:** 2026. **Venue:** arXiv 2026. **Primary:** https://arxiv.org/abs/2601.07156.  **Access:** primary_arxiv_pdf_text_truncated_at_50000_chars. **Reproduced:** no.

## Why this work exists

Provide consistent almost-global navigation observer geometry rather than local-only linearized attitude filtering while maintaining known gauge directions.

## Algorithm and observation model

Construct an extended matrix Lie group unifying pose, gravity, linear velocity and landmarks. Decouple almost-global rotation observer and exponential Riccati translational observer; combine in cascaded stability analysis.

Extended matrix Lie group SE_(3+n)(3) for gravity/velocity/landmarks with joint SO(3) geometry. Gauge remains global gravity-axis yaw and translation; global conclusions depend on visual-inertial observation model.

## Evidence and implementation boundary

Primary arXiv PDF body retrieved (indexed text truncated at 50k characters); author simulation and EuRoC experimental claims not independently recomputed.

## What the paper does not prove

Requires visual measurements; convergence cannot be claimed for arbitrary six-axis motion by removing bearing information. Full methods not individually rederived in this second pass.

## Consequences for TG-DCM research

Explicit prior art for geometric observer with gravity and bias state; focus potential novel work on six-axis observation constraints and calibrated confidence.

[Second-pass verification](../review/09_second_pass_verification.md) · [full review](../review/08_literature_review.md)
