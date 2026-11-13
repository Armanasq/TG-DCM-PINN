# Observer Design for Optical Flow-Based Visual-Inertial Odometry with Almost-Global Convergence

**Authors:** Tarek Bouazza; Soulaimane Berkane; Minh-Duc Hua; Tarek Hamel. **Year:** 2025. **Venue:** arXiv 2025; CDC 2025 submission. **Primary:** https://arxiv.org/abs/2508.21163.  **Access:** full_arxiv_pdf_text. **Reproduced:** no.

## Why this work exists

Estimate gravity direction and body velocity using optical-flow measurements with geometric convergence guarantees rather than a trajectory-regression prior.

## Algorithm and observation model

Cascaded observer: optical-flow velocity-direction constraints and inertial measurements feed a persistently exciting Riccati observer for body velocity/gravity; a complementary SO(3) observer recovers attitude using estimated gravity and optional magnetic observation.

Body vertical/gravity is an explicit S2 state; Riccati observer guarantees require persistently exciting visual output; SO(3) orientation is recovered up to its unobservable gravity-axis heading.

## Evidence and implementation boundary

Complete primary arXiv PDF text retrieved through scholarly connector; abstract and method reviewed. Stability and simulations are author claims, not independently proven/reproduced.

## What the paper does not prove

Uses optical flow; convergence theorem does not hold after removing visual inputs. No directly matched raw-six-axis-only tilt benchmark.

## Consequences for TG-DCM research

Treat explicit body-gravity-state observer as prior art and identify exactly what a new six-axis correction provides beyond established Riccati/geometry mechanisms.

[Second-pass verification](../review/09_second_pass_verification.md) · [full review](../review/08_literature_review.md)
