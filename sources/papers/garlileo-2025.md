# GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry

**Year:** 2025. **Authors:** Chiyun Noh; Sangwoo Jung; Hanjun Kim; Yafei Hu; Laura Herlant; Ayoung Kim. **Venue:** 2025 arXiv; later 2026 journal. **Primary source:** https://arxiv.org/abs/2511.13216. **DOI:** https://doi.org/10.1177/02783649261457941. **Access:** primary_full_arxiv_abstract_and_paper_pdf_text_truncated. **Independent execution:** none.

## Research objective

Foot slip and vibration degrade body gravity and vertical trajectory estimation in legged robots.

## Method, mathematical mechanism and assumptions

Continuous-time fusion of radar Doppler-derived ego velocity, leg kinematics and IMU with a soft S2 gravity factor.

S2 gravity factor coupled to velocity-aware continuous-time motion spline; radar/leg signals provide additional observations.

## What was actually verified

Primary arXiv landing and indexed paper manuscript inspected; no code executed and journal numeric results not independently reproduced.

## Limitations and hidden observation dependencies

External radar and leg kinematics at runtime; not raw six-axis standalone tilt. S2 gravity factors not novel by themselves.

## Relevance to proposed original research

Prior art for velocity-aided gravity, appropriate as an aided upper-bound, not same sensor-budget baseline.

[Novelty audit](../proposals/02_prior_art_and_novelty.md) · [Proposed method](../proposals/01_closure_constrained_tilt.md)
