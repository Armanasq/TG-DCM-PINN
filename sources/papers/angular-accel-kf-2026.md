# Deep learning-aided Kalman filter for low-cost attitude estimation via angular accelerometer and gyroscope fusion

**Authors:** Chaoyang Zhai; Meiling Wang; Zhiheng Xiao; Zitian Xiong; Renjie Li. **Year:** 2026. **Venue:** Measurement 260, 119824 (February 2026).
**Primary source:** https://www.sciencedirect.com/science/article/pii/S0263224125031835. **DOI:** https://doi.org/10.1016/j.measurement.2025.119824.
**Classification:** transfer. **Access actually achieved:** publisher_abstract_extended_preview. **Code inspected:** none. **Independent reproduction:** no. **Reviewed:** 2026-11-13.

## Research objective and original problem

Low-cost gyro signal inaccuracies accumulate; physically informative angular acceleration and adaptive covariance can constrain integration error.

## Architecture and algorithmic design

Two Kalman fusion designs for gyroscope and separate angular accelerometer, augmented by learned process/measurement noise covariance rather than manually tuned fixed matrices.

## Mathematical or information-theoretic formulation

Classical innovation K=P^- H^T(HP^-H^T+R)^-1 with learned positive noise matrix R or Q; angular acceleration observation is an additional physical sensor, not derivable without uncertainty from the gyro itself.

## Evidence actually available

Publisher abstract reports >17% estimation error reduction in selected turntable experiments. Methods and raw files not independently checked.

## Limitations and alternative interpretation

Requires a distinct angular accelerometer, so not an equal-sensor six-axis attitude competitor.

## Relevance and falsifiable transfer to TG-DCM-PINN

Learned covariance mechanisms and turntable testing with explicitly declared hardware budget.

## Provenance

See [full 2024–2026 synthesis](../review/08_literature_review.md) and [pinned repository index](../repositories/index.json). Publisher/HTML/abstract content was reviewed with the stated access constraint; no large original paper asset or external model checkpoint was installed locally.
