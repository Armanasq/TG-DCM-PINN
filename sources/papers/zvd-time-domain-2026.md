# Time-domain features based zero-velocity detection for foot-mounted pedestrian inertial navigation system

**Year:** 2026. **Authors:** Yanping Chen; Qiao Zhou; Jianpeng Li; Jun Zhou; et al.. **Venue:** Measurement 275, 121394 (2026). **Primary source:** https://www.sciencedirect.com/science/article/pii/S0263224126011036. **DOI:** https://doi.org/10.1016/j.measurement.2026.121394. **Access:** publisher_abstract_and_section_preview. **Independent execution:** none.

## Research objective

Traditional foot-stance ZUPT gates struggle with varying gait speed; reliably recognizing zero velocity can matter more than downstream filter architecture.

## Method, mathematical mechanism and assumptions

Time-domain motion features detect foot zero-velocity intervals for SINS/ESKF navigation; authors report performance in field pedestrian trials.

Zero velocity pseudo-observation v=0 constrains state via EKF update when stance detection is true; output depends on correct event classification.

## What was actually verified

Publisher abstract and section snippets; full article code/parameters and experiments not independently obtained.

## Limitations and hidden observation dependencies

Foot-mounted gait and exact zero velocity; not an equal nonzero sensor-velocity closure model across arbitrary carriers.

## Relevance to proposed original research

Match a normal ZUPT event detector and analyze false-closure risk on previously unseen motions.

[Novelty audit](../proposals/02_prior_art_and_novelty.md) · [Proposed method](../proposals/01_closure_constrained_tilt.md)
