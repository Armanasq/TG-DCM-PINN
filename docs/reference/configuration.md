# TG-DCM Constants

This file lists the constants and hardcoded values used by the implemented TG-DCM code.

## Quaternion and Numeric Constants

Source: `src/tg_dcm/losses.py`, `src/tg_dcm/so3.py`

| Constant | Value | Meaning |
|---|---:|---|
| `EPS` in losses | `1e-7` | Small value for stable quaternion normalization, vector norms, square roots, and angle clamps. |
| `EPS` in SO(3) utilities | `1e-8` | Small value for NumPy quaternion normalization and interpolation. |
| `G` | `9.81` | Gravity magnitude in `m/s^2`; used by the gravity reliability gate. |
| Quaternion order | `w, x, y, z` | Scalar-first Hamilton quaternion convention used across training and evaluation. |
| Model input width | `7` | Acceleration `3`, gyroscope `3`, and timestep `dt` `1`. |
| Model output width | `4` | Predicted quaternion components. |
| Minimum `dt` in loss | `1e-6` | Prevents division by zero during gyro propagation and jerk calculation. |
| Charbonnier default epsilon | `1e-3` | Default smoothness value for `sqrt(x^2 + eps^2) - eps`. |

## DCM Vertical Direction

Source: `src/tg_dcm/losses.py`

The code does not build the full `3 x 3` DCM. It directly computes the body-frame world-vertical direction:

```text
h(q) = [
  2(xz - wy),
  2(yz + wx),
  1 - 2(x^2 + y^2)
]
```

| Value | Meaning |
|---|---|
| `[0, 0, 1]` | World vertical direction represented through the equivalent DCM row/column. |
| `2.0` | Quaternion-to-DCM coefficient. |
| `1.0` | Unit vertical basis term. |

Purpose: compare tilt through vertical direction instead of full quaternion orientation.

## Supervised Loss Constants

Source: `src/tg_dcm/losses.py`, `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| `inc_weight` | `1.0` | Weight of supervised tilt error. |
| `norm_weight` | `0.01` | Weight of raw quaternion unit-norm regularization. |
| Norm target | `1.0` | Desired quaternion norm. |

The supervised loss is:

```text
L_sup = 1.0 * tilt_charbonnier + 0.01 * quaternion_norm_error
```

## Target-Gated DCM Dynamics Constants

Source: `src/tg_dcm/losses.py`, `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| `dyn_weight` | `0.10` | Weight of the target-gated gyro dynamics loss. |
| `target_gate_deg` | `2.0 deg` | Accepts a target transition only if gyro-propagated target tilt matches the next target within this threshold. |
| `dyn_deadzone_deg` | `0.15 deg` | Ignores very small dynamics disagreement. |
| `dyn_delta_deg` | `0.5 deg` | Charbonnier scale for the dynamics residual. |
| Minimum sequence length | `2` samples | Dynamics loss needs at least one consecutive pair. |

Purpose: train the predicted vertical direction to follow measured gyro evolution only on trusted labelled transitions.

## Gravity-Gated Anchor Constants

Source: `src/tg_dcm/losses.py`, `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| `gravity_weight` | `0.05` | Weight of the gravity anchor loss in the released training configuration. |
| `gravity_deadzone_deg` | `0.20 deg` | Ignores very small predicted-vs-accelerometer vertical disagreement. |
| `gravity_delta_deg` | `0.5 deg` | Charbonnier scale for the gravity residual. |
| `gravity_sigma` | `0.25 m/s^2` | Width of the acceleration-magnitude gate around `9.81 m/s^2`. |
| `jerk_sigma` | `45.0 m/s^3` | Width of the acceleration-change gate. |
| `rate_sigma` | `1.4 rad/s` | Width of the angular-rate gate. |
| `gate_floor` | `1e-4` | Samples below this gate weight are ignored. |

The implemented gate is:

```text
m_grav =
exp(-(|a| - 9.81)^2 / (2 * gravity_sigma^2))
* exp(-jerk^2 / (2 * jerk_sigma^2))
* exp(-|gyro|^2 / (2 * rate_sigma^2))
```

Purpose: use acceleration as a gravity direction only when the IMU sample looks gravity-dominant.

## Total Training Objective

Source: `src/tg_dcm/train_impl.py`

```text
L_total = L_sup + 0.10 * L_dyn + 0.05 * L_grav
```

`L_sup` already contains:

```text
1.0 * supervised_tilt + 0.01 * quaternion_norm
```

## Model Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| RNN type | `GRU` | Recurrent model type. |
| Number of recurrent layers | `2` | Two stacked GRU layers. |
| Hidden size | `200` | Hidden channels per GRU layer. |
| `n_skip` | `0` | No initial output samples are skipped in training loss. |
| `sub_seq_len` | `1800` | Truncated subsequence length used by the learner. |
| Gradient clip | `1.0` | Maximum gradient norm used by the learner. |
| Input normalization mean | `[-0.6845, -0.4654, 0.2514, -0.0715, 0.0343, 0.1019, 0.0035]` | Fixed normalization mean for `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt]`. |
| Input normalization std | `[7.1055, 7.0015, 7.1748, 2.0147, 1.9856, 2.0225, 0.0018847]` | Fixed normalization standard deviation for `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt]`. |

## Training Schedule Constants

Source: `src/tg_dcm/train_impl.py`, `scripts/train.py`

| Constant | Value | Meaning |
|---|---:|---|
| `epochs` | `512` | Default training length. |
| `lr` | `3e-3` | Maximum learning rate used by learner setup. |
| `pct_start` | `0.15` | Fraction of schedule used for learning-rate warmup. |
| `weight_decay` | `0.01` | Optimizer weight decay. |
| `batch_size` | `64` | Training batch size. |
| `seed` | `42` | Default random seed. |
| `save_every` | `1` | Save checkpoint every epoch. |
| `metric_every` | `10` | Update training metrics every 10 batches. |
| `metric_samples` | `8` | Number of samples used when computing batch metrics. |

The scheduler uses:

```text
total_steps = epochs * train_batches
warmup_fraction = 0.15
```

## Optimizer Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| Optimizer | `Ranger` | Optimizer used in the released training path. |
| `betas` | `(0.95, 0.99)` | Momentum parameters. |
| `eps` | `1e-6` | Optimizer numerical stability value. |
| `use_gc` | `False` | Gradient centralization disabled. |

## EMA Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| `max_decay` | `0.9999` | Maximum EMA decay. |
| Warmup numerator offset | `1.0` | Used in `(1 + step) / (10 + step)`. |
| Warmup denominator offset | `10.0` | Slows EMA early in training. |

Implemented decay:

```text
decay(step) = min(0.9999, (1 + step) / (10 + step))
```

Purpose: average model weights while reducing early random-initialization influence.

## Data Constants

Source: `src/tg_dcm/data.py`, `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| Input columns | `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt` | Input features loaded from HDF5. |
| Target columns | `opt_a, opt_b, opt_c, opt_d` | Target quaternion loaded from HDF5. |
| Training window size | `9000` samples | Window length for train/valid datasets. |
| Training step size | `60` samples | Stride between training windows. |
| Training batches per epoch | `300` | Weighted sampler draws `300 * batch_size` samples per epoch. |
| Target frequency range | `50` to `500 Hz` | Resampling range for training entries. |
| Target frequency count | `427` | Number of equally spaced target frequencies in released training. |
| Default DataLoader workers | `4` | Worker count in the training wrapper. |
| `prefetch_factor` | `2` | Batches prefetched per DataLoader worker. |
| `persistent_workers` | `False` | Worker persistence disabled. |
| `cache_data` | `False` | HDF5 data caching disabled by default. |
| `drop_last` train | `True` when sampled | Drops incomplete train batches when finite sampler is used. |
| `drop_last` valid | `False` | Keeps all validation samples. |

## Data Split Constants

Source: `src/tg_dcm/data.py`

| Constant | Value | Meaning |
|---|---|---|
| `MYON_VALID_IDS` | `{14, 39, 21}` | Myon subjects assigned to validation. |
| `MYON_TEST_IDS` | `{29, 22, 35}` | Myon subjects assigned to test. |
| `TUMVI_TRAIN_ROOMS` | `{room1, room2, room3}` | TUM-VI rooms assigned to training. |
| `TEST_DATASETS` | `OxIOD, EuRoC-MAV, Caruso-Sassari, RepoIMU, Caruso-Sassari_orig` | External datasets assigned to test. |

Split logic:

```text
train = Myon except valid/test IDs + TUM-VI room1-room3
valid = Myon valid IDs + remaining TUM-VI rooms
test = Myon test IDs + external test datasets
```

## Augmentation Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---:|---|
| Quaternion augmentation groups | `[3, 3]` | Quaternion sign/group augmentation setting. |
| Gyro bias probability | `0.1067` | Probability of applying gyroscope bias augmentation. |
| Gyro bias std | `[0.006556, 0.006461, 0.006581]` | Bias std for `gyr_x, gyr_y, gyr_z`. |
| Acc bias std | `[0, 0, 0]` | No acceleration bias in this augmentation. |
| `dt` bias std | `0` | No timestep bias. |
| Noise probability | `0.6123` | Probability of grouped noise augmentation. |
| Acc noise std scale | `0.6410` | Grouped noise scale for acceleration channels. |
| Gyro noise std scale | `0.00712` | Grouped noise scale for gyroscope channels. |
| `dt` noise std scale | `0.0` | No grouped noise on timestep. |
| Noise groups | `[0, 0, 0, 1, 1, 1, 2]` | Acceleration, gyroscope, and timestep channel grouping. |

## Backend Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Value | Meaning |
|---|---|---|
| Device default | `auto` | Use CUDA when available, otherwise CPU. |
| CUDA graph | `True` | Enabled when CUDA and subsequence length are active. |
| cuDNN | `True` | Enabled by default. |
| BF16 autocast | CUDA only | Training and validation use BF16 autocast on CUDA. |
| `torch.compile` minimum | PyTorch `2.3` | Loss compilation is skipped below 2.3. |
| Compile mode | `default` | Mode used for supervised and auxiliary losses. |
| Compile auxiliary losses | `True` | Auxiliary losses are compiled when compile is allowed. |
| Finite gradient guard | enabled | NaN and Inf gradients are replaced with zero. |
| Orthogonal recurrent init | enabled | Applied to GRU recurrent matrices. |

## Smoke-Mode Constants

Source: `src/tg_dcm/train_impl.py`

| Constant | Smoke value | Meaning |
|---|---:|---|
| `epochs` | `1` | One epoch only. |
| `batch_size` | `min(batch_size, 2)` | Caps batch size at 2. |
| `win_sz` | `min(win_sz, 128)` | Caps window length at 128. |
| `stp_sz` | `min(stp_sz, 64)` | Caps stride at 64. |
| `n_batches_train` | `1` | One training batch. |
| `targ_fs_count` | `1` | One resampling frequency. |
| `sub_seq_len` | `0` | Disables subsequence truncation. |
| `cuda_graph` | `False` | Disables CUDA graph. |

## Evaluation Constants

Source: `scripts/evaluate.py`

| Constant | Value | Meaning |
|---|---:|---|
| Default checkpoint | `checkpoints/tg_dcm.pt` | Model checkpoint evaluated by default. |
| `GRAVITY_MAG` | `9.81` | Gravity magnitude for static gate. |
| `STATIC_THR` | `0.5` | Static gate accepts `abs(|acc| - 9.81) < 0.5`. |
| `N_ALIGN` | `100` frames | Frames used for heading alignment before roll/pitch RMSE. |
| `CHUNK_SIZE` | `50000` samples | Inference chunk size for bounded memory. |
| `BACKEND_WORKERS` | `3` | Runs model, RIANN ONNX, and VQF concurrently where possible. |
| VQF warmup cap | `500` frames | Initial VQF output is replaced for up to 500 frames. |
| VQF warmup fraction | `len(q) // 4` | Warmup is also capped by one quarter of the sequence. |
| RIANN hidden shape | `(2, 1, 1, 200)` | ONNX initial hidden state for 2 GRU layers and hidden size 200. |
| RIANN dt | `1 / fs` | Constant timestep used for RIANN ONNX evaluation input. |
| TG-DCM default split | `test` | Evaluation default split. |
| Sequence plots default | `none` | Per-sequence plots disabled by default. |
| Max sequence workers default | `min(8, max(1, cpu_count // 4))` | Default sequence-level parallelism. |

Primary tilt metric:

```text
tilt = 2 * arccos(sqrt(w^2 + z^2))
```

Roll/pitch RMSE uses heading alignment over the first `100` valid frames.

## SO(3) Preprocessing Constants

Source: `src/tg_dcm/so3.py`

| Constant | Value | Meaning |
|---|---:|---|
| `max_offset_s` | `0.05 s` | Maximum ground-truth time offset searched during gyro alignment. |
| `step_s` | `0.0025 s` | Offset search step. |
| Minimum samples for offset search | `3` | Returns zero offset when fewer samples are available. |
| SLERP linear threshold | `0.9995` | Uses linear interpolation when quaternions are nearly identical. |
| Anti-alias cutoff factor | `0.45` | Low-pass cutoff fraction of source/target sampling rate. |
| Fallback moving-average width minimum | `1` | Minimum smoothing width when SciPy filter is unavailable. |
