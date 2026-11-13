# Time and Frequency Domain Analysis of IMU-Based Orientation Estimation Algorithms with Comparison to Robotic Arm Orientation as Reference

**Authors:** Ruslan Sultan; Steffen Greiser. **Year:** 2025. **Venue:** Sensors 25(16), 5161 (2025).
**Primary source:** https://doi.org/10.3390/s25165161. **DOI:** https://doi.org/10.3390/s25165161.
**Classification:** direct-evaluation. **Access actually achieved:** open_access_full_text_sections. **Code inspected:** none. **Independent reproduction:** no. **Reviewed:** 2026-11-13.

## Research objective and original problem

Time-domain tilt RMSE alone misses phase delay and tracking bandwidth, especially under rapid motion.

## Architecture and algorithmic design

UR5e robot arm with Bosch 6-axis IMU; compare indirect KF, Madgwick and complementary filters using slow-amplitude poses and fast generalized binary noise; compute time RMSE/MaxAE, frequency response, coherence and averaged equivalent time delay.

## Mathematical or information-theoretic formulation

Dynamic transfer function and phase/coherence quantify frequency-specific attenuation and lag. AETD relates measured phase delay to frequency under the paper's composite-frequency calculation.

## Evidence actually available

Open-access article concludes Madgwick and complementary filters have best overall time/frequency accuracy on studied motions; IKF worst on those metrics but lowest averaged equivalent time delay.

## Limitations and alternative interpretation

One controlled robot-arm apparatus and tuned filters; orientation reference/extrinsic calibration and frequency window affect interpretation. Different filters may optimize different bandwidth targets.

## Relevance and falsifiable transfer to TG-DCM-PINN

Add frequency response, phase lag and transient delay to TG-DCM benchmarking, not only angular RMSE.

## Provenance

See [full 2024–2026 synthesis](../review/08_literature_review.md) and [pinned repository index](../repositories/index.json). Publisher/HTML/abstract content was reviewed with the stated access constraint; no large original paper asset or external model checkpoint was installed locally.
