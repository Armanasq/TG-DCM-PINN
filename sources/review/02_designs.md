# 02 — System design, experimental design, and methodology choices

**Purpose:** translate 2024–2026 findings into explicit, testable design contracts for [TG-DCM-PINN](../../README.md). [Architectures](01_architectures.md) identify algorithmic candidates; this chapter defines **how to compare them fairly and how to prevent silent failure**.

## 1. Input, output, and observability contract

A valid strict six-axis sample consists of body-frame measured specific force \(f_m\), measured angular velocity \(\omega_m\), timestamp \(t\) or positive \(\Delta t\), and declared units. The required observable output is \(\hat h=R_{WB}^{T}e_z\in S^2\). A quaternion may be returned as an implementation representation, but absolute yaw must not be presented as an externally identified state.

The baseline source [configs/model.yaml](../../configs/model.yaml) specifies seven channels, a GRU, wxyz quaternion, body-to-world rotation and train-time physics. Reference conventions must state **sensor-specific gravity sign**; an accelerometer measures specific force, not world gravitational acceleration directly.

| Contract | Must be recorded | Failure if omitted |
|---|---|---|
| Frame | Sensor axes; ENU/NED; body-to-world; quaternion convention; handedness | 180° errors and misleading cross-dataset means |
| Clock | Device timestamps; offset; drift; interpolation method; dt validity | False gyro inconsistency and invented angular increments |
| Sensors | Raw 6-axis only versus extra attitude, pressure, rotor speed, LiDAR teacher | Incomparable accuracy headline |
| Calibration | Accelerometer/gyro bias, scale, misalignment, lever arm | OOD model errors disguised as motion error |
| Inference context | Causal state, window length, future samples, reset and warmup | BiLSTM falsely compared as instant GRU |
| Label | GT orientation source, sync method, valid gaps, sign continuity | Supervision/evaluation leakage and untrusted targets |
| Evaluation | Tilt angle, aggregation unit, paired bootstrap cluster, p95 | Misleading small average gains |
| Provenance | Source revision, dataset version/hash, seed, checkpoint hash, commands | Results cannot be regenerated |

## 2. Data lifecycle: integrity before optimization

Treat raw IMU and raw ground-truth time bases as immutable records. Generate versioned validated data with exact transforms. The [data.py](../../src/tg_dcm/data.py) training split places Myon subjects and TUM-VI room groups into train/validation; external datasets and held-out Myon IDs are tested. It uses a weighted sampler across 50–500 Hz resampled training entries and unresampled validation/test entries. Any claim about rate generalization therefore needs a separate controlled frequency sweep and anti-aliasing check.

**Concrete source inspection:** the [so3.py](../../src/tg_dcm/so3.py) interpolation path filters invalid quaternion rows before searching adjacent timestamps; queries can bridge an originally invalid truth gap. Exact final timestamps are also excluded because the searchsorted right index equals the valid length. In [_common.py](../../src/tg_dcm/prep/_common.py), the public SLERP wrapper uses this code. Fix by interpolating only within uninterrupted, monotonic valid segments; support exact valid endpoints; preserve missing regions. Regenerate data and reevaluate rather than only patching a plotted metric.

The same [so3.py](../../src/tg_dcm/so3.py) has a GT/gyro time-offset estimator selecting the offset with smallest median angular-increment disagreement. This is physically sensible synchronization, but **selecting it using held-out test ground truth can alter the evaluation protocol**. An offset derived from test references must be explicitly documented and applied to all methods identically, or acquired from independent calibration motions. Do not silently optimize alignment for each method's own predictions.

## 3. Mechanistic versus statistical trust

The 2024–2026 papers exhibit three distinct notions of reliability. They should not be merged:

- **Reference validity:** a target pair agrees with gyro propagation within a tolerance; this informs which training supervision is credible, not whether a live accelerometer measures gravity.
- **Measurement validity:** accelerator residual is likely gravity-dominated; can be estimated heuristically, by modeled lever-arm compensation, or by a calibrated innovation covariance.
- **Model uncertainty:** predicted errors and learned variance are evaluated against observed residual statistics and must be calibrated on unseen domains.

[TG-DCM loss](../../src/tg_dcm/losses.py) uses a 2° target-gated dynamics threshold and a product of acceleration-norm, jerk and gyro-rate Gaussians with a \(10^{-4}\) support floor. Count-normalized weighted loss is not the same as confidence-mass normalization. A gate occupancy collapse can artificially lower an auxiliary branch's *unconditional* contribution; report accepted-pair fraction and effective weight mass, stratified by dataset and motion regime.

[RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097) adapts a Kalman measurement covariance or compensates dynamics; [KISS-IMU 2026](https://arxiv.org/html/2603.06205) predicts uncertainty for preintegration and PGO; [MosaicIMU 2026](https://arxiv.org/html/2606.09355) predicts velocity covariance for a filter. A next TG-DCM experiment should predict **SPD tangent covariance** and test NLL, empirical coverage, NEES and error calibration, not only use variance to weight loss.

## 4. Motion compensation versus censoring

[Brändle et al. 2025](https://arxiv.org/pdf/2502.03681) show that sensor lever arms coupled with angular acceleration alter the closed-loop dynamics of attitude filters; [Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784) model and remove \(\alpha\times r+\omega\times(\omega\times r)\) when enough geometry is known. **Decision:** first distinguish systematic rigid-body acceleration from arbitrary independent translation. Only the former can be compensated with a known sensor-to-pivot vector and angular motion. A network that sees \(\|f\|\approx g\) cannot infer “gravity-only” in general, even with zero jerk and gyro rate; see [adversarial construction](03_mathematics.md#8-a-constructive-non-identifiability-counterexample).

Design paired cases: pure rotation around the IMU; identical angular rate with a nonzero lever arm; translational acceleration at constant velocity change; changes in angular acceleration; and unknown/misestimated lever arm. No selected strategy is universally trustworthy; a fallback policy must detect low observability rather than assert accuracy.

## 5. Temporal context and streaming

Measured inference latency equals signal acquisition/window buffering + future lookahead + neural compute + queuing and state update. A 2-layer causal GRU can emit every sample after a warmup; a BiLSTM sees both directions of the **presented window**. [Ding 2026](https://doi.org/10.1016/j.measurement.2026.122485) explicitly exposes a latency-configurable sliding window. [MosaicIMU public source](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) contains bidirectional LSTM modules in its mean and variance heads; the training config has a 500-sample window at 200 Hz, or 2.5 s of signal. The actual online delay is **unknown without the output-to-window alignment policy**; do not equate full window duration to delay or report zero lookahead.

Use fixed-delay evaluation: 0 ms pure past, then explicitly 10/25/50/100 ms budget at each dataset's rate, with identical delay allowed to the baseline and competitor. Record first usable estimate after initialization, steady-state compute latency on CPU/GPU, memory, batch-1 throughput and behavior after missing data.

## 6. Domain transfer, augmentation and equivariance

[EqNIO](https://arxiv.org/html/2408.06321) uses group-theoretic canonicalization after gravity alignment; [GINIO](https://arxiv.org/abs/2609.25338) uses full SO(3)-equivariant layers; [TartanIMU 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html) and [MosaicIMU](https://arxiv.org/html/2606.09355) pursue cross-platform learned adaptation. These are orthogonal choices: invariance to **coordinate change** is not adaptation to a **different sensor noise process**, a changed lever arm, temperature drift or new motion semantics.

Define two tests explicitly: (i) virtual *coordinate remount* with known orthogonal Q applied consistently to body vector data, attitude, covariance and calibrations; (ii) real *physical remount* that changes bias, lever arm and sensor contact. Only the first has an exact algebraic group law without additional physical modeling.

## 7. Experimental protocol and aggregation

TG-DCM's [released evaluation](../../docs/evaluation.md) has 137 standard sequences and 422 DIODEM segments from 86 raw motions. Its [evaluation.py](../../src/tg_dcm/evaluation.py) uses a grouped bootstrap for the latter. Specify the primary estimand before reporting a confidence interval:

1. Equal-weight sequence/segment mean: every logged segment contributes once.
2. Equal-weight raw-motion mean: first aggregate segments from the same original motion.
3. Dataset-balanced mean: each dataset contributes the same top-level weight.
4. Time-weighted frame mean: longer recordings contribute more frames.

These four estimands are **not equivalent**. For DIODEM, resample *raw motion groups*, not segments. For architecture claims use at least five independent training seeds, matched tuning budget, immutable test sets, and report paired differences, 95% intervals, p50/p95/max, per-domain ranking and whether the effect survives excluding any one dataset.

## 8. Reproducibility, source-management and guardrails

The canonical research pipeline should be:

    original paper / code URL + DOI or arXiv id
       -> version/commit, license and format metadata
       -> accessible primary text and exact inspected functions
       -> claim vs implementation vs experiment comparison
       -> reproducible hypothesis/ablation specification
       -> immutable experiment config, checkpoint and data
       -> per-sequence results and numerical/property assertions.

The [sources directory](../README.md) stores original analytical summaries and a network-enabled acquisition tool; blocked or unacquired PDFs and unexecuted cloned repos must remain marked **not acquired**. Copyright terms of an arXiv HTML/PDF and code licenses are distinct. Keep the main Git repository navigable: research review markdown tracked; large paper/repository caches ignored.

## 9. Design decision record: what not to do

Do not declare end-to-end full attitude observable from six axes. Do not compare GT-aided, LiDAR-supervised or barometer-assisted baselines as if identical. Do not treat data augmentation as an equivariance proof. Do not infer uncertainty calibration from an NLL optimization alone. Do not use architecture size as a proxy for scientific contribution. Do not select the best result among dozens of seeds and report it as an unbiased effect. Do not modify training/evaluation implementation during the literature-only phase without isolated tests and a separate benchmark record.
