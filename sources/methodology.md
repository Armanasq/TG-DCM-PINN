# Search and review protocol

**Date:** 2026-11-13. **Primary research time filter:** calendar years 2024–2026 (first arXiv submission or peer-reviewed publication). Pre-2024 techniques (Mahony, Madgwick, VQF, RIANN, preintegration, AirIMU) may be cited as historical baselines, not counted as new contributions.

## Current coverage

As of 2026-11-13 the corpus contains **26** curated 2024–2026 works and **10** codebases including the target, with exact inspected paths pinned in the repository index. Source discovery is iterative and **not exhaustive**; direct attitude studies were prioritized over adjacent odometry, and publisher-abstract-only studies remain explicitly qualified. The eight technical chapters are indexed in [review/README.md](review/README.md).

## Inclusion logic

1. **Direct:** raw six-axis accelerometer/gyro to tilt/roll/pitch, or six-axis filter error correction under dynamics, with or without learning.
2. **Near-direct:** attitude jointly estimated with velocity/position; state input, initial conditions and whether auxiliary sensors are needed must be exposed.
3. **Transfer:** learned inertial odometry with relevant architecture, invariant geometry, bias estimation, self-supervision, model uncertainty, online adaptation or physical constraints. The task and metrics must be described as different.

## Search sources and verification

Searches used scholarly indexing (CoChat/Crossref/arXiv-family discovery), arXiv landing and structured HTML, publisher pages, and direct GitHub file API. Cross-check DOI/author/year with scholarly index or publisher metadata where possible. For primary-source models, inspect the equations, training assumptions, evaluation metrics, operating window and linked paper sections before claiming implementation correspondence.

**Evidence tags**:
- primary_full_text_sections / primary_html_sections: primary paper accessible with relevant method and result sections inspected; **not proof of line-by-line complete read**.
- open_access_full_text_inspected_partly: accessible full-text data with selected technical sections inspected.
- primary_full_text_sections_plus_code: primary method sections and actual Python modules inspected.
- publisher_abstract_only / publisher_abstract_extended_preview / primary_abstract_only: do **not** infer exact equations, hyperparameters or confidence intervals.
- ...plus_author_errata: repository contains explicit corrections/concerns; not independently resolved.

## Exact reproducibility rules

Log every code repository URL and full revision SHA. A clone only counts as cloned after local checkout and SHA verification. Code inspection means reading non-README implementation files; it does not imply executing tests, running training or recovering paper metrics.

For each result require: prediction target (tilt, full attitude, velocity, displacement, pose); available sensor streams; train/validation/test disjointness; coordinate-frame conventions; reference-sensor synchronization and resampling; causal window and future lookahead; initialization and bias corrections; per-sequence results; aggregation estimator; uncertainty and paired confidence interval; checkpoint hash and command-line config. A quantitative result from one task cannot be transferred to another.

## Acquisition boundaries

The checkout environment had no usable GitHub/arXiv DNS path for local git clone or bulk paper download. Original analytical notes and verified remote code paths are committed. Cache directory is not populated and is Git-ignored; acquisition is reproducible via tools/acquire.py in a network-connected environment. Publisher-licensed papers are not mirrored to a public repository. For each paper, record licensing/terms from its landing page before redistributing or modifying source assets. An arXiv identifier is a metadata key, not a universal republication license.

## Deduplication and open gaps

Deduplicate by DOI and arXiv identifier across preprint, conference and journal versions; prefer journal citation when definitive; retain arXiv for inspectable equations. Scope is purposefully quality-filtered, not an exhaustive PRISMA systematic review. Follow-up research should search new publication months, handle paywalled methods separately, compare citations backward/forward and add unpublished counterexamples under explicit confidence tags.
