# A Robust Transformer-Based Error Compensation Method for Gyroscope of IMUs

**Authors:** Xin Ye; Weijia Xing; Zhilou Yu; Jichen Chen; Qingyue Ma. **Year:** 2025. **Venue:** Journal of Field Robotics 2025.
**Primary source:** https://onlinelibrary.wiley.com/doi/full/10.1002/rob.70082. **DOI:** https://doi.org/10.1002/rob.70082.
**Classification:** near-direct. **Access actually achieved:** publisher_full_html_sections. **Code inspected:** none. **Independent reproduction:** no. **Reviewed:** 2026-11-13.

## Research objective and original problem

Nonlinear time-varying gyroscope bias/scale/noise causes cumulative integration drift despite correct quaternion kinematics.

## Architecture and algorithmic design

Transformer sliding-window temporal gyroscope error prediction with SO(3)-aware preintegration loss and robust rotation residuals.

## Mathematical or information-theoretic formulation

omega_corr=omega_m-delta_omega_theta(history); q_{k+1}=q_k tensor Exp_q(omega_corr dt); geometric residual penalizes integrated rotation disagreement on SO(3). Exact hyperparameters depend on the full implementation.

## Evidence actually available

Publisher claims mean orientation error 1.56° versus Gyro-Net 1.49° and describes it as better, although 1.56>1.49 for a lower-is-better metric; the reported ordering is inconsistent and no independent rerun resolves it.

## Limitations and alternative interpretation

Windowed historical dependence and integrated bias drift; published numerical claim requires correction or matched table context. No trainable repository verified.

## Relevance and falsifiable transfer to TG-DCM-PINN

Explicit gyro residual/bias head with physical propagation, compared against implicit GRU memory on bias drift and cross-device tests.

## Provenance

See [full 2024–2026 synthesis](../review/08_literature_review.md) and [pinned repository index](../repositories/index.json). Publisher/HTML/abstract content was reviewed with the stated access constraint; no large original paper asset or external model checkpoint was installed locally.
