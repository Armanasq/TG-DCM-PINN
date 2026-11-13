# MosaicIMU: Composing Carrier Experts for Generalizable Neural Inertial Odometry

**Authors:** Junye Zou; Huiyi Yan; Xinning Xu; Xiaolei Li; Pengkun Zhou; Jinhui Zhang; Ziyang Meng. **Year:** 2026. **Venue:** arXiv 2026 preprint.
**Primary source:** https://arxiv.org/html/2606.09355. **DOI:** https://doi.org/10.48550/arXiv.2606.09355.
**Classification:** transfer. **Access actually achieved:** primary_html_sections_plus_code. **Code inspected:** zoujy02/MosaicIMU. **Independent reproduction:** no. **Reviewed:** 2026-11-13.

## Research objective and original problem

Different human/robot carriers and sensor mounts exhibit distinct motion statistics and biases; a single learned motion prior generalizes unevenly.

## Architecture and algorithmic design

Depthwise/residual convolution encoder, cosine-prototype router with Gumbel softmax, four carrier experts, shared upsampling sequence decoder with bidirectional-LSTM velocity and variance heads; a cloned-state MSCKF combines velocity constraints. Unseen carrier adapts a new residual expert and router rather than all pretrained weights.

## Mathematical or information-theoretic formulation

Router weights alpha_i=softmax(sim(normalize(z),normalize(p_i))*scale/tau); fused temporal features sum_i alpha_i E_i(X). Predicted mean mu and softplus-positive variances feed a history-clone Kalman innovation. Prototype orthogonality and carrier-prior losses regularize the router.

## Evidence actually available

Primary arXiv paper reports average ATE reduction 40% and RTE-10s reduction 34% over cited baselines. Actual source modules and config inspected; no numeric result independently recomputed. A quadruped ATE row can be worse than TLIO despite favorable averages.

## Limitations and alternative interpretation

Multi-carrier generalization is conditional. Authors cite unknown carrier, unusual mount, severe bias and initially supervised adaptation. BiLSTM heads consume bidirectional window context, and latency is not necessarily zero-lookahead; 500-sample 200Hz window covers 2.5 s of signal, not automatically 2.5 s output delay.

## Relevance and falsifiable transfer to TG-DCM-PINN

Small domain-specific residual expert or training-only motion balancing, tested on held-out device. Do not conflate odometry ATE with tilt RMSE.

## Provenance

See [full 2024–2026 synthesis](../review/08_literature_review.md) and [pinned repository index](../repositories/index.json). Publisher/HTML/abstract content was reviewed with the stated access constraint; no large original paper asset or external model checkpoint was installed locally.
