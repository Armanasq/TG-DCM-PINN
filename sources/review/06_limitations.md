# 06 · Limitations — exact claim boundary and evidential debt

**Purpose:** separate the limits imposed by physics, those imposed by code, and those imposed by available evidence. A mathematical impossibility is not a software bug; a source-code hazard is not evidence of a changed reported benchmark until replayed.

## 1. Structural limits no architecture can remove

- **Global yaw gauge.** From gravity direction and body-rate alone, \(R\) and \(R_z(\phi)R\) are observationally equivalent for constant world yaw. A quaternion network may select a training-conditioned heading representative; that is not a globally measured yaw.
- **Translation/gravity ambiguity.** An arbitrary specific-force measurement includes unknown reference-point acceleration. Acc norm approximately g and low jerk are not evidence that it is gravity-only. A 30° false-gravity vector can have the same norm g and zero jerk/rate; see [construction](03_mathematics.md).
- **Calibration/initialization dependence.** A bias head cannot identify every IMU nuisance variable under non-exciting motion. A known lever arm can remove rotational terms, not arbitrary translation; gyro differentiation adds noise.
- **Domain dependence of learned motion priors.** Learned regularity of gait, drone thrust or hand motion is contingent on carrier, mounting and rate. Covariance may be overconfident after shift.

**Consequence:** a physically defensible paper should state the motion/inference assumptions under which it expects accuracy, not claim universal six-axis absolute orientation or indefinitely drift-free pose.

## 2. Directly observed target implementation risks

| Evidence source | Source-level finding | Affected claim or metric | Still unknown |
|---|---|---|---|
| [so3.py](../../src/tg_dcm/so3.py) | SLERP chooses neighboring **valid compressed** rows and may cross original GT gaps; exact final endpoint excluded | Training labels, gyro-reference gate and any prepared-data metric | Number of real affected frames and error change |
| [losses.py](../../src/tg_dcm/losses.py) | dt clamped before pair validity; finite nonpositive dt can be used as tiny positive interval | Gyro physics consistency and jerk | Incidence in actual training |
| [losses.py](../../src/tg_dcm/losses.py) | 0×NaN propagation through soft-mask reduction; finite mask calculated after some nonlinear residuals | Robustness of forward loss and gradients to invalid IMU | Whether preprocessing eliminated all such frames |
| [losses.py](../../src/tg_dcm/losses.py) | Finite-but-zero quaternion passes target validity | Nonphysical training target and inconsistent error conventions | Presence of degenerate GT rows |
| [evaluate.py](../../scripts/evaluate.py) | VQF initial output replaced with later orientation | Startup baseline causal semantics | How many sequences/ranks change on removal |
| [evaluate.py](../../scripts/evaluate.py) | “Heading alignment” computes full GT-relative SO(3) correction | Secondary roll/pitch metrics use truth-derived tilt correction | Magnitude of bias in published secondary columns |
| [evaluate.py](../../scripts/evaluate.py) | Fixed dt for RIANN/VQF, stored per-frame dt for TG | Fair comparison under jitter | Actual dt distributions |
| [evaluate.py](../../scripts/evaluate.py) | No-valid-GT case uses an all-true mask after identity substitution; input lengths truncated to minimum | Spurious sequence score / hidden misalignment | Whether any released files trigger this branch |
| [evaluate.py](../../scripts/evaluate.py) | “Static” is abs(acc norm−g) threshold | Static-gated metrics | Whether classified motions were truly static |
| [data.py](../../src/tg_dcm/data.py) | Same raw training motion appears at many resampled rates | Effective data independence | Exact duplicate exposure, seed effects |
| [train_impl.py](../../src/tg_dcm/train_impl.py) | A selected 512-epoch run, without controlled released factorial loss ablations | Contribution attribution and seed uncertainty | Variability across independent runs |

No training code was executed in this review. The numerical fact 0×NaN=NaN was verified independently with PyTorch arithmetic; the full loss-path incidence still requires targeted package tests.

## 3. Published TG-DCM evidence: what follows and what does not

**Released standard result:** 137 sequences, mean tilt RMSE 1.521°, paired mean difference from RIANN -0.091°, 95% sequence bootstrap interval [-0.147,-0.036]. This supports a **fixed checkpoint's aggregate mean** on the named standard split, conditional on current data/evaluation protocol. It does not prove that the gate, the optimizer, the interpolation changes, or the model family caused the effect. It does not include run-to-run confidence.

**DIODEM:** 422 segments derived from 86 raw motions; mean 10.41°, median 2.70°, p95 44.20°, max 62.38°. Clustered paired mean difference CI versus RIANN includes 0. Thus a resolved DIODEM mean advantage is **not** shown at 95%; tail failure is prominent. Segment-level p95 cannot be reported as a frame-level p95. A raw-motion-balanced analysis might alter the weighting even when bootstrap clustering is correct.

**Inference speed:** the current evaluation times different backends on different processing devices and may run them concurrently; it is not a same-hardware sample-to-output latency comparison. An early-output VQF backfill also means its reported startup accuracy is not causal.

## 4. Key paper-specific limitations from method and artifact inspection

**[Wei et al., RAESKF 2025](https://doi.org/10.1016/j.measurement.2024.116097):** adaptive covariance/acceleration compensation is a direct physical baseline, but detailed filter tuning, threshold sensitivity and full experimental tables were unavailable in the accessible publisher material. The reported pitch/roll percentage gains cannot be transferred to this dataset.

**[Ding et al., 2026](https://doi.org/10.1016/j.measurement.2026.122485):** direct six-axis CNN–BiLSTM, frequency encoding and QSGD loss. Publisher content establishes a future-frame latency tradeoff, not a zero-lookahead result; median full-angle errors are not sequence tilt RMSE. Full trainable code and exact splits were not verified. A [2023 CNN–BiLSTM six-axis paper](https://doi.org/10.1016/j.measurement.2023.113105) is essential prior art.

**[Liu et al., RSS 2025](https://arxiv.org/html/2504.09495):** continuous bias NODE avoids direct bias labels but still relies on pose GT to train, approximate bias initialization from pose differences and stationary calibration for startup. The initial training/validation split uses adjacent portions of the same recordings, while held-out test recordings are separate. Noise is not fully modeled. [Actual code](https://github.com/UMich-CURLY/Debias_IMU/tree/7eb2afcaba71d27c9c340850c19158f33c8b2f69/BiasDy) confirms a backward cubic Hermite spline, separate bias nets and Lie ODE integration. Causal inference and **online bias identifiability under arbitrary motion** are different claims.

**[Tsai/Chung preprint, 2026](https://doi.org/10.31224/7144):** 2,811 experiments, roughly 97% dataset contribution to variance and under 1% architecture in the authors' specific two-dataset bias task, and no gain from one SO(3) regularization tested. This is **not peer-reviewed** and the label sources differ substantially between EuRoC and TUM-VI. It is a valuable methodological warning, not a general theorem that neural architecture never matters.

**[TransIMU, 2025](https://doi.org/10.1002/rob.70082):** the publisher abstract compares 1.56° proposed orientation error against 1.49° Gyro-Net and calls the proposed model superior. Under a lower-is-better shared metric that comparison is inconsistent. The underlying table/definition may clarify it; the review must not repair it by guessing.

**[EqNIO 2024](https://arxiv.org/html/2408.06321):** O(2) symmetry is constructed **after gravity alignment** and used for displacement/covariance, not raw-six-axis tilt. A gravity estimate from the current filter is an input dependency; the group law is not automatically full physical mounting robustness.

**[GINIO 2026](https://arxiv.org/abs/2609.25338):** the vector-neuron mechanism is meaningful, but the [maintained author-side known-issues document](https://github.com/chankyo123/ginio/blob/6b97f178be1a253ef44e9a05ed17432b41c00287/docs/KNOWN_ISSUES.md) reports unsupported spectral NLL/NEES values, a model labeled with LFA trained without it, mixed trained/zero-shot comparisons and physical remount non-equivariance of an added attitude channel. These are author-maintained artifact assertions, not independently replayed archives. Do not repeat the paper's affected rows as verified benchmark wins.

**[Tartan IMU 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.html):** [public transformer.py](https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/tartan_imu/model/backbones/transformer.py) explicitly raises NotImplementedError. Its full Transformer result cannot be reproduced from the inspected release alone.

**[KISS-IMU 2026](https://arxiv.org/html/2603.06205):** LiDAR ICP/pose graph supplies the training teacher; “self-supervised” does not mean training without a secondary modality. Pseudo-label corruption can be amplified by rare-motion weights.

**[MosaicIMU 2026](https://arxiv.org/html/2606.09355):** prototype experts are conditioned on carrier statistics. The inspected [MoE model](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py) uses bidirectional temporal heads; paper limitations include unfamiliar carrier, mount/bias conditions and adaptation supervision. Mean ATE gains coexist with carrier-specific losses.

**[Arzberger–Nüchter 2026](https://arxiv.org/html/2607.25784):** lever-arm compensation requires estimated geometry, angular acceleration and trustworthy calibration; unknown translational acceleration remains.

## 5. Review-specific limitations and reproducibility

This is a source-backed targeted research synthesis, **not an exhaustive systematic or reproduced meta-analysis**. Some papers were read in full via open HTML/PDF text, others only via publisher preview. GitHub files were inspected at pinned revisions; local cloning was blocked by unavailable GitHub DNS in the execution environment and third-party training/checkpoint execution was not performed. A source-level bug report is separate from a confirmed change in benchmark statistics. Paper registries record access status; no paywalled content is represented as fully read.

**Publication boundary:** at present a defensible claim is “target-gated tilt-oriented recurrent supervision with a reported modest standard-domain improvement.” Universal acceleration rejection, calibrated runtime confidence, exact full-system equivariance, strong OOD tails, independently reproduced external baselines and novelty of every component **are not demonstrated**.
