# Intelligent zero-velocity detection with autoencoder fusion for enhanced foot-mounted human tracking

**Year:** 2026. **Authors:** Yu Liu; Jun Zhou; Ruimin Song; Yanping Chen; et al.. **Venue:** Measurement 2026. **Primary source:** https://www.sciencedirect.com/science/article/pii/S0263224125020263. **DOI:** https://doi.org/10.1016/j.measurement.2025.118667. **Access:** publisher_abstract_only. **Independent execution:** none.

## Research objective

Learned motion state classification can improve foot-mounted zero-velocity detection over fixed IMU thresholds but remains distribution dependent.

## Method, mathematical mechanism and assumptions

Time/frequency features, GLRT and convolutional autoencoder for foot stance / gait condition, then ZUPT correction in SINS.

Fusion of a GLRT event likelihood and learned reconstruction features selects zero-velocity pseudo-observation updates.

## What was actually verified

Publisher article abstract and highlights, author numbers not independently reproduced.

## Limitations and hidden observation dependencies

Foot-stance contact dynamics and environment-specific training; a learned ZVD event does not provide a deterministic physical guarantee.

## Relevance to proposed original research

Use as a direct learned-event/GLRT comparator before claiming a novel closure detector.

[Novelty audit](../proposals/02_prior_art_and_novelty.md) · [Proposed method](../proposals/01_closure_constrained_tilt.md)
