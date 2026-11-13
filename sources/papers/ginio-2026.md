# GINIO: A Geometric SO(3)-Equivariant Interface for Neural Inertial Odometry

**Reference:** Chankyo Kim; Minghan Zhu; Tzu-Yuan Lin; Avantika Rattan; Maani Ghaffari. CoRL 2026, 2026. Primary source: https://arxiv.org/abs/2609.25338. arXiv: 2609.25338. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** primary_full_text_sections_plus_code_plus_author_errata. **Code inspected:** chankyo123/ginio.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Rotating or remounting the IMU should transform motion means/covariances predictably; ordinary sensor-frame networks violate the geometric law.

## Architecture / algorithm / rationale

SO(3)-equivariant vector-neuron convolution/GRU and geometry-aware outputs; Last-Frame Alignment (LFA) for equivariant models; spectral covariance parameterization transported by congruence.

## Mathematical or identifiability mechanism

m(QZ)=Qm(Z) and Sigma(QZ)=Q Sigma(Z) Q^T. In vector-neuron GRU, update/reset gates are invariant scalars and hidden features are 3-vectors transformed equivariantly.

## Experimental evidence actually verified

Paper reports TLIO, NanoBench and Fetch remount robustness and ATE (meters); actual repository includes code and an author-side docs/KNOWN_ISSUES.md audit documenting several paper-to-checkpoint discrepancies.

## Limitations and critical interpretation

Important evidence failures in public erratum: some printed covariance NLL/NEES values lack measured support; an LFA-labeled checkpoint was trained without LFA; a NanoBench comparison mixed TLIO-pretrained and NanoBench-trained models; with-attitude input is not equivariant under certain physical remount transformations; some rotation robustness cells lack generators or reuse different weights. Do NOT present all paper tables as reproduced.

## Actionable connection to TG-DCM-PINN

Evaluate *actual* SO(3) remount transformations including attitude-channel transformation and covariance; demand checkpoint hashes, settings and regeneration scripts for every published cell.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/chankyo123/ginio, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
