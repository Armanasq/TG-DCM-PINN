# Effects of Gyroscope Bias Augmentation on Recurrent Neural Network Model Performance for Attitude Estimation

**Reference:** Ji Hoon Park; Chang June Lee; Jung Keun Lee. IEEE Access 2026, 2026. DOI: https://doi.org/10.1109/ACCESS.2026.3691717. Primary source: https://doi.org/10.1109/ACCESS.2026.3691717. 
**Applicability:** Six-axis tilt estimation. **Evidence accessed:** publisher_metadata_abstract_only. **Code inspected:** none verified.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Gyroscope bias sensitivity damages recurrent six-axis attitude estimators and synthetic bias augmentation may be insufficiently characterized.

## Architecture / algorithm / rationale

Investigate recurrent model performance when training gyro signals are augmented with bias perturbations. Detailed architecture/protocol unavailable in this review.

## Mathematical or identifiability mechanism

Synthetic omega_train = omega_measured + b_sample, with b_sample distribution and temporal correlation requiring confirmation against paper methods.

## Experimental evidence actually verified

Publisher DOI and abstract were located; quantitative findings, experiment setup and paper tables were not independently checked.

## Limitations and critical interpretation

Do not infer optimal bias distribution, improvement magnitude or model choice without full text.

## Actionable connection to TG-DCM-PINN

Run controlled constant/random-walk/temperature-proxy gyro bias ablations and report separate device OOD performance.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. No verified trainable implementation was inspected. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
