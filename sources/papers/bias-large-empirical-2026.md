# A Large-Scale Empirical Study of Deep Learning for MEMS Gyroscope Bias Estimation

**Authors:** Hsiu-Chi Tsai; Chia-Tung Chung (EngrXiv metadata; the extracted PDF title page reverses order). **Year:** 2026. **Venue:** engrXiv 2026 preprint; not peer reviewed. **Primary source:** https://doi.org/10.31224/7144. **DOI:** https://doi.org/10.31224/7144. **Evidence accessed:** full_text_via_scholarly_connector_primary_pdf_index. **External benchmark re-executed:** no.

## Why this research was necessary

IMU bias architecture comparisons can be dominated by ground-truth construction, preprocessing, recording domains and cross-sequence leakage instead of learning architecture.

## Mechanism and model assumptions

Empirical study of MLP, LSTM, TCN and Transformer bias regressors with LOSO on EuRoC and TUM-VI, 17 random seeds per core setting, ablations of label denoising and SO(3) auxiliary regularization.

Variance decomposition of experiment outcomes into dataset and architecture contributions; study reports approximately 97% variation attributable to dataset choice and <1% to model family, conditional on the two datasets and tested settings.

## What the evidence actually shows

Original full-text extraction was available through scholarly index, and engrXiv DOI/landing metadata verified. Authors report 2,811 runs, no benefit of tested physics regularization in 390 confirmatory runs, and label corruption effects larger than architecture effects. No independent numeric reproduction.

## Hidden constraints and failure conditions

Non-peer-reviewed preprint; two datasets, different bias label construction: EuRoC batch-optimization labels vs TUMVI angular-rate GT differentiated from smoothed pose. Variance attribution is NOT a global result across all IMU sensing or tilt tasks; author order conflicts between PDF and publisher landing metadata.

## Defensible TG-DCM experiment

Treat label-process variation and hardware/domain shift as first-class experimental factors before increasing network size or claiming a physics auxiliary-loss novelty.

**Trace:** [comprehensive review](../review/08_literature_review.md), [math and limits](../review/03_mathematics.md), [repository sources](../repositories/index.json). Source text was analyzed without locally downloading the full paper to the public repository or reproducing training metrics.
