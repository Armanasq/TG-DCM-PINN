# Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics

**Reference:** Shibo Zhao and collaborators. CVPR 2025, 2025. Primary source: https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html. 
**Applicability:** Transferable inertial estimation. **Evidence accessed:** publisher_html_sections_plus_code. **Code inspected:** superxslam/TartanIMU.
**Review cutoff:** 2026-11-13. **Independent replication:** no.

## Problem and motivation

Models tied to one device, embodiment or dataset generalize poorly to other locomotion and sensor domains.

## Architecture / algorithm / rationale

Cross-platform IMU feature pretraining, lightweight ResNet-LSTM trunk and platform-specific output heads with efficient adaptation.

## Mathematical or identifiability mechanism

The shared sequence encoder maps sensor windows to latent motion representations; per-platform heads predict motion and are fine-tuned instead of retraining all parameters.

## Experimental evidence actually verified

CVPR paper reports multi-platform experiments and adaptation; released Python code contains registered LSTM Foundation_Model but its Transformer builder explicitly raises NotImplementedError because that core was not released.

## Limitations and critical interpretation

The public Transformer implementation is not available in the inspected revision; its inertial positioning targets and velocity/trajectory metrics differ from roll/pitch.

## Actionable connection to TG-DCM-PINN

Pretrain a compact causal encoder over heterogeneous IMU devices, evaluate adaptation by held-out device and mount; release exact inference core and checkpoints.

## Provenance and remaining verification

The linked primary source and the evidence level above establish the limits of the current review. No full PDF/LaTeX asset is stored in this Git-tracked note. Source-code evidence: https://github.com/superxslam/TartanIMU, with exact revision, inspected files and code-level notes in ../repositories/index.json and ../implementation/external-code-audit.md. Reproduce any numerical claim only after documenting common data splits, timing/frames, preprocessing and evaluation scripts.
