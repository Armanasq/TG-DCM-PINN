# IONext: Unlocking the Next Era of Inertial Odometry

**Reference:** Shanshan Zhang and collaborators. 2025 arXiv preprint, 2025. Primary source: https://arxiv.org/abs/2507.17089. arXiv: 2507.17089. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** primary_abstract_only. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Pure Transformers can miss high-frequency local motion detail despite their long-range receptive field.

## Architecture / algorithm / rationale

CNN-style Dual-wing Adaptive Dynamic Mixer combines global and local motion features; Spatio-Temporal Gating Unit selects temporally relevant events.

## Mathematical or identifiability mechanism

Input-conditioned multi-scale mixing and temporal gating act as a selective motion representation, not a physical attitude observer.

## Experimental evidence actually verified

Authors report six inertial navigation datasets and average ATE/RTE improvements over iMoT on an RNIN split; no TG-DCM tilt evidence.

## Limitations and critical interpretation

Main task is inertial odometry. Code, matched latency and OOD sensor-mount effects were not verified.

## Actionable connection to TG-DCM-PINN

Test small multi-rate temporal convolutions vs the GRU under fixed parameter count/streaming latency.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
