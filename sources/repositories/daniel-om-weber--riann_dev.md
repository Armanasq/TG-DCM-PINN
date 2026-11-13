# RIANN training reimplementation (2026 code artifact)

**Code:** https://github.com/daniel-om-weber/riann_dev/tree/435274141c08ab318c01be904de9754943b4feed

**Revision:** 435274141c08ab318c01be904de9754943b4feed. **License:** Apache-2.0. **Related historical 2021 paper:** [RIANN, AI 2(3), 444–463](https://doi.org/10.3390/ai2030028). **Status:** inspected source remotely; not cloned or run.

| File | Inspected source behavior |
|---|---|
| [train.py](https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/train.py) | 2-layer GRU, inclination objective, Ranger, fixed train normalization, bias/noise augmentation, rate-resampled training |
| [riann/data.py](https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/riann/data.py) | Myon/TUM-VI splits and weighted multi-rate training windows |
| [riann/prep/euroc.py](https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/riann/prep/euroc.py) | Reference-state estimated gyro bias subtracted before HDF5; Vicon/room data |
| [scripts/validate_vqf.py](https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/scripts/validate_vqf.py) | Uses offline VQF and GT offset for *validation*, optionally mag; do not mistake that for a deployed raw-six-axis estimator |
| [scripts/compare_hdf5.py](https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/scripts/compare_hdf5.py) | Programmatic comparison of newly prepared versus reference HDF5; relevant to Caruso artifact provenance |

**Important attribution:** the repository's README states that original RIANN training code was lost and this is a reimplementation. It is distinct from the pretrained RIANN ONNX benchmark model. TG-DCM borrows established architecture/data practices, so its plausible novel contribution is the *specific target-gated objective and demonstrated effects*, not GRU/rate training by itself.

**Unresolved:** external code not executed; LFS HDF5 bytes and checksum parity not inspected; exact lineage of TG-DCM implementation cannot be proved solely from similar structure.
