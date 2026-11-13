# DIODEM – A Diverse Inertial and Optical Dataset of kinEmatic chain Motion

**Authors:** Simon Bachhuber; Dustin Lehmann; Ive Weygers; Thomas Seel. **Year:** 2025. **Venue:** Scientific Data 12, 1285 (2025). **Primary:** https://www.nature.com/articles/s41597-025-05468-w. **DOI:** https://doi.org/10.1038/s41597-025-05468-w. **Access:** primary_full_text_html_and_scholarly_xml. **Reproduced:** no.

## Why this work exists

IMU motion-capture benchmarks may conflate attitude error, sensor-to-segment misalignment, rigid versus nonrigid attachment, repeated trials, optical-reference errors and evaluation dependence.

## Algorithm and observation model

Eleven continuous experiments of two five-segment kinematic chains with optical markers, five rigid and five foam-mounted IMUs. Published processing calibrates sensor S to body B and optical M to earth E. Source CSVs contain rigid/nonrigid IMU and optical per-segment quaternions; motion types are nested in continuous sessions.

Rigid calibration maps f_B=R_BS f_S, omega_B=R_BS omega_S; optical segment orientation R_EB=R_EM R_MB. Segment tilt error requires comparing quantities in the same body frame. Dependence hierarchy: segment, motion, experiment session.

## Evidence and implementation boundary

Published open-access primary description reviewed in full (nature.com and EuropePMC full text); 11 experiments, ten IMUs, foam artifacts broadband and known alignment processing. Released TG-DCM table has 422 segments nested in 86 motions and 11 experiments. No physical raw-recording or HDF5 converter replay.

## What the paper does not prove

If a derived HDF5 set does not preserve exact frame/calibration/mounting identity, apparent tilt failures may not be attributable to learned estimator. Only eleven independent continuous experiment sessions; category and mounting metadata not traceable in released test HDF5 pipeline.

## Consequences for TG-DCM research

Reconstruct DIODEM converter; verify rigid/foam, segment/frame, GT alignment, experiment clustering and per-motion p95 before claiming improved generalization.

[Second-pass verification](../review/09_second_pass_verification.md) · [full review](../review/08_literature_review.md)
