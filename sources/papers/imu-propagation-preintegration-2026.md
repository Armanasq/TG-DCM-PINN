# IMU Propagation as Preintegration

**Year:** 2026. **Authors:** Jianzhu Huai. **Venue:** arXiv 2026 technical note. **Primary source:** https://arxiv.org/abs/2605.28279.  **Access:** primary_arxiv_pdf_full_text. **Independent execution:** none.

## Research objective

IMU propagation and preintegration are often duplicated code paths despite encoding identical physical state updates.

## Method, mathematical mechanism and assumptions

Demonstrate convention-independent equivalence between propagation and preintegration with bias Jacobians and covariance, including tests against GTSAM.

Preintegrated relative rotation/velocity/position and propagated state share underlying discrete state-transition and noise-error Jacobians.

## What was actually verified

Full source preprint extracted and read; validation against GTSAM is author-reported.

## Limitations and hidden observation dependencies

A methodological/integration note, not a six-axis attitude benchmark or new observable measurement. Confirms rolling preintegration is prior art.

## Relevance to proposed original research

Use to validate numerical equality of prefix-integral and explicit relative-window implementations.

[Novelty audit](../proposals/02_prior_art_and_novelty.md) · [Proposed method](../proposals/01_closure_constrained_tilt.md)
