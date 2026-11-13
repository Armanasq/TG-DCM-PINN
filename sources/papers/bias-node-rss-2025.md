# Debiasing 6-DOF IMU via Hierarchical Learning of Continuous Bias Dynamics

**Authors:** Ben Liu; Tzu-Yuan Lin; Wei Zhang; Maani Ghaffari. **Year:** 2025. **Venue:** Robotics: Science and Systems XXI, 2025. **Primary source:** https://arxiv.org/html/2504.09495. **DOI:** https://doi.org/10.15607/RSS.2025.XXI.009. **Evidence accessed:** primary_arxiv_html_and_RSS_pdf_plus_code. **External benchmark re-executed:** no.

## Why this research was necessary

Explicit time-varying accelerometer and gyro biases are difficult to train because direct bias labels are rarely available and are confounded with motion.

## Mechanism and model assumptions

Separate controlled Neural ODEs for gyro and accelerometer bias; causal backward-difference cubic-Hermite input interpolation; SO(3) Lie-group integration via local charts; hierarchical gyro-first then accelerometer training using pose ground truth.

dot(b_g)=f_theta(b_g,omega_m,dot(omega_m)), dot(b_a)=g_phi(b_a,f_m,dot(f_m)); dot(R)=R[omega_m-b_g]_x. Optimize orientation pose-log residual then translational losses instead of direct bias labels.

## What the evidence actually shows

RSS peer-reviewed primary paper and arXiv HTML sections, RSS PDF first page and repo source network_for_BiadDy.py, Interpolation.py, lie_algebra.py and learning.py examined. No training executed.

## Hidden constraints and failure conditions

No bias-GT labels does NOT mean no pose GT: each training interval uses pose-derived approximate initial bias; stationary startup calibration is assumed. EuRoC/TUMVI validation uses temporal 80/20 within training sequences (test sequences held out), and stochastic noise beyond deterministic bias drift is left as future work.

## Defensible TG-DCM experiment

Compare explicit slow bias against implicit GRU state under common six-axis tilt, independent device, bias jumps and cold-start conditions.

**Trace:** [comprehensive review](../review/08_literature_review.md), [math and limits](../review/03_mathematics.md), [repository sources](../repositories/index.json). Source text was analyzed without locally downloading the full paper to the public repository or reproducing training metrics.
