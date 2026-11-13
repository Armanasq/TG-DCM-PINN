# Reduced Order Observers for Monocular Visual Inertial Odometry

**Authors:** Amel Abi; Mouaad Boughellaba; Miaomiao Wang; Abdelhamid Tayebi. **Year:** 2026. **Venue:** arXiv 2026 version 2, 31 October 2026. **Primary:** https://arxiv.org/abs/2609.26612.  **Access:** full_arxiv_pdf_text. **Reproduced:** no.

## Why this work exists

Recover gravity direction and velocity without estimating every visual landmark, while preserving explicit yaw/translation gauge and geometric convergence conditions.

## Algorithm and observation model

Construct output from optical bearing measurements and temporal derivatives, yielding a reduced six-dimensional LTV observer for gravity and body velocity. Recover attitude up to constant yaw and translation up to a world offset. Add bias for 12-D LTV extension with local stability under stated conditions.

Coupled LTV observer dynamics for body velocity/gravity with gains from a Riccati equation; stability depends on visual bearing-derived output and uniform observability. Constant-yaw gauge remains.

## Evidence and implementation boundary

Primary arXiv PDF full text retrieved; claims include uniform global exponential stability for unbiased reduced observer and local bias-augmented stability under assumptions. No numerical replication.

## What the paper does not prove

Camera bearings, derivatives and excitation are essential; not direct six-axis-only attitude accuracy evidence. Modeling gravity as explicit state is not novel independently.

## Consequences for TG-DCM research

Use as controlled geometric prior art for S2 observer, and state precisely the loss of guarantees when removing visual observation.

[Second-pass verification](../review/09_second_pass_verification.md) · [full review](../review/08_literature_review.md)
