"""SO(3) and quaternion utilities used by TG-DCM.

The repository stores scalar-first Hamilton quaternions in w, x, y, z order.
These helpers keep interpolation, sign continuity, and rotation increments
explicit instead of relying on component-wise quaternion arithmetic.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, EPS, None)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_log(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    v = q[..., 1:]
    vnorm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vnorm, np.clip(q[..., :1], EPS, None))
    scale = np.where(vnorm < EPS, 2.0 + vnorm**2 / 3.0, angle / np.clip(vnorm, EPS, None))
    return v * scale


def estimate_time_offset_from_gyro(
    imu_time: np.ndarray,
    gyro: np.ndarray,
    gt_time: np.ndarray,
    quat: np.ndarray,
    max_offset_s: float = 0.05,
    step_s: float = 0.0025,
) -> float:
    """Choose a constant GT time offset by matching GT increments to gyro."""
    imu_time = np.asarray(imu_time, dtype=np.float64)
    gt_time = np.asarray(gt_time, dtype=np.float64)
    if len(imu_time) < 3 or len(gt_time) < 3:
        return 0.0
    dt = np.diff(imu_time)
    finite = np.isfinite(gyro[:-1]).all(axis=1) & np.isfinite(dt) & (dt > EPS)
    if finite.sum() < 3:
        return 0.0
    offsets = np.arange(-max_offset_s, max_offset_s + 0.5 * step_s, step_s)
    best_offset = 0.0
    best_score = np.inf
    for offset in offsets:
        q0 = interpolate_quat(gt_time, quat, imu_time[:-1] + offset)
        q1 = interpolate_quat(gt_time, quat, imu_time[1:] + offset)
        valid = finite & np.isfinite(q0).all(axis=1) & np.isfinite(q1).all(axis=1)
        if valid.sum() < 3:
            continue
        rot_gt = quat_log(quat_mul(quat_conj(q0[valid]), q1[valid]))
        rot_gyro = gyro[:-1][valid] * dt[valid, None]
        score = float(np.nanmedian(np.linalg.norm(rot_gt - rot_gyro, axis=1)))
        if score < best_score:
            best_score = score
            best_offset = float(offset)
    return best_offset


def fix_quaternion_continuity(quat: np.ndarray) -> np.ndarray:
    """Apply antipodal dot-product continuity without crossing NaN gaps."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    valid = np.isfinite(quat).all(axis=1)
    if not valid.any():
        return quat
    quat[valid] = normalize_quat(quat[valid])
    valid_idx = np.flatnonzero(valid)
    breaks = np.flatnonzero(np.diff(valid_idx) > 1) + 1
    for segment in np.split(valid_idx, breaks):
        if len(segment) < 2:
            continue
        dots = np.sum(quat[segment[1:]] * quat[segment[:-1]], axis=1)
        flips = np.where(dots < 0.0, -1.0, 1.0)
        signs = np.empty(len(segment), dtype=np.float64)
        signs[0] = 1.0
        signs[1:] = np.cumprod(flips)
        quat[segment] *= signs[:, None]
    return quat


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    alpha = np.asarray(alpha, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)

    linear = dot > 0.9995
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - alpha) * theta) / np.clip(sin_theta, EPS, None)
    w1 = np.sin(alpha * theta) / np.clip(sin_theta, EPS, None)
    out = w0 * q0 + w1 * q1
    lerp = (1.0 - alpha) * q0 + alpha * q1
    return normalize_quat(np.where(linear, lerp, out))


def interpolate_quat(times: np.ndarray, quat: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Gap-aware SLERP interpolation at query times."""
    times = np.asarray(times, dtype=np.float64)
    quat = fix_quaternion_continuity(quat)
    query = np.asarray(query, dtype=np.float64)
    out = np.full((len(query), 4), np.nan, dtype=np.float64)
    valid = np.isfinite(quat).all(axis=1) & np.isfinite(times)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return out

    valid_times = times[valid_idx]
    valid_quat = quat[valid_idx]
    right = np.searchsorted(valid_times, query, side="right")
    ok = (right > 0) & (right < len(valid_times))
    if not ok.any():
        return out

    li = right[ok] - 1
    ri = right[ok]
    t0 = valid_times[li]
    t1 = valid_times[ri]
    span = t1 - t0
    ok_span = span > EPS
    q = np.full((ok.sum(), 4), np.nan, dtype=np.float64)
    q[ok_span] = slerp(
        valid_quat[li[ok_span]],
        valid_quat[ri[ok_span]],
        (query[ok][ok_span] - t0[ok_span]) / span[ok_span],
    )
    out[np.flatnonzero(ok)] = q
    return out


def resample_linear(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty((len(query), values.shape[1]), dtype=np.float64)
    for i in range(values.shape[1]):
        valid = np.isfinite(values[:, i]) & np.isfinite(times)
        if valid.sum() < 2:
            out[:, i] = np.nan
        else:
            out[:, i] = np.interp(query, times[valid], values[valid, i], left=np.nan, right=np.nan)
    return out


def anti_alias_then_resample(
    times: np.ndarray, values: np.ndarray, query: np.ndarray, target_dt: float
) -> np.ndarray:
    """Low-pass vectors before downsampling, then interpolate at query times."""
    values = np.asarray(values, dtype=np.float64)
    if len(times) < 4 or target_dt <= 0.0:
        return resample_linear(times, values, query)

    src_dt = float(np.nanmedian(np.diff(times)))
    filtered = values.copy()
    if np.isfinite(src_dt) and src_dt > 0.0 and target_dt > src_dt * 1.05:
        try:
            from scipy.signal import butter, sosfiltfilt

            src_fs = 1.0 / src_dt
            target_fs = 1.0 / target_dt
            cutoff = min(0.45 * target_fs, 0.45 * src_fs)
            if 0.0 < cutoff < 0.5 * src_fs and np.isfinite(filtered).all(axis=0).all():
                sos = butter(4, cutoff, btype="lowpass", fs=src_fs, output="sos")
                filtered = sosfiltfilt(sos, filtered, axis=0)
        except Exception:
            width = max(1, int(round(target_dt / src_dt)))
            if width > 1:
                kernel = np.ones(width, dtype=np.float64) / width
                filtered = np.vstack(
                    [
                        np.convolve(filtered[:, i], kernel, mode="same")
                        for i in range(filtered.shape[1])
                    ]
                ).T
    return resample_linear(times, filtered, query)


def resample_gyro_increment(
    times: np.ndarray, gyro: np.ndarray, query: np.ndarray, target_dt: float
) -> np.ndarray:
    """Resample gyro by preserving integrated angular increments per interval."""
    gyro = np.asarray(gyro, dtype=np.float64)
    if len(times) < 2 or len(query) == 0 or target_dt <= 0.0:
        return resample_linear(times, gyro, query)

    end_times = query + target_dt
    all_times = np.unique(np.concatenate([times, query, end_times]))
    interp = resample_linear(times, gyro, all_times)
    cumulative = np.zeros_like(interp)
    dt = np.diff(all_times)
    area = 0.5 * (interp[:-1] + interp[1:]) * dt[:, None]
    cumulative[1:] = np.cumsum(area, axis=0)
    out = np.empty((len(query), gyro.shape[1]), dtype=np.float64)
    for axis in range(gyro.shape[1]):
        c0 = np.interp(query, all_times, cumulative[:, axis], left=np.nan, right=np.nan)
        c1 = np.interp(end_times, all_times, cumulative[:, axis], left=np.nan, right=np.nan)
        out[:, axis] = (c1 - c0) / target_dt
    return out
