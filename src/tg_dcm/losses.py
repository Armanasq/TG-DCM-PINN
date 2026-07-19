"""TG-DCM training losses and IMU augmentations.

The model still maps acc/gyr/dt to a four-channel quaternion with the original
GRU; these modules only change training objectives and augmentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-7
G = 9.81


def qnorm(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1, eps=EPS)


def vec_norm(v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return torch.sqrt(v.square().sum(dim=-1, keepdim=keepdim).clamp_min(EPS * EPS))


def qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def qexp(rotvec: torch.Tensor) -> torch.Tensor:
    angle = vec_norm(rotvec, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle < EPS,
        0.5 - angle.square() / 48.0,
        torch.sin(half) / angle.clamp_min(EPS),
    )
    return torch.cat([torch.cos(half), rotvec * scale], dim=-1)


def finite_target_quat(
    target: torch.Tensor, fallback: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_raw = target[..., :4]
    valid = torch.isfinite(q_raw).all(dim=-1)
    q_safe = torch.where(valid.unsqueeze(-1), q_raw, fallback.detach())
    return qnorm(q_safe), valid


def inclination_error_rad(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    q_err = qnorm(qmul(q_pred, qconj(q_target)))
    w, x, y, z = q_err.unbind(dim=-1)
    sin_half_tilt = torch.sqrt((x.square() + y.square()).clamp_min(EPS * EPS))
    cos_half_tilt = torch.sqrt((w.square() + z.square()).clamp_min(EPS))
    return 2.0 * torch.atan2(sin_half_tilt, cos_half_tilt)


def geodesic_error_rad(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(qnorm(q_pred) * qnorm(q_target), dim=-1).abs().clamp(0.0, 1.0 - EPS)
    return 2.0 * torch.acos(dot)


def robust_charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x.square() + eps * eps) - eps


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Soft-mask multiply: no GPU→CPU sync, no variable-size boolean indexing,
    # fuseable by torch.compile. Clamp avoids div-by-zero on all-invalid batches.
    v = mask.to(values.dtype)
    return (values * v).sum() / v.sum().clamp_min(1.0)


def gravity_dir_body(q: torch.Tensor) -> torch.Tensor:
    """Body-frame world-vertical direction for scalar-first Hamilton quaternions.

    This is equivalent to ``qrot(qconj(q), [0, 0, 1])`` but avoids constructing
    the constant vector and two full quaternion multiplications on every loss
    call. Target-DCM calls this on large ``batch x time`` tensors, so the
    closed-form DCM row is materially faster while preserving the same math.
    """
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ],
        dim=-1,
    )


def angle_between_unit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)
    # For unit vectors, ||a x b|| = sqrt(1 - dot(a,b)^2).  This avoids a
    # three-channel cross product and norm in the target-DCM hot path.
    cross = torch.sqrt((1.0 - dot.square()).clamp_min(EPS * EPS))
    return torch.atan2(cross, dot)


def gyro_step(q: torch.Tensor, gyr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Integrate one quaternion step using gyroscope reading and timestep."""
    return qnorm(qmul(q, qexp(gyr * dt)))


class TargetDCM3Loss(nn.Module):
    """Target-gated DCM training loss for tilt accuracy.

    This loss keeps the yaw-free target-gated dynamics term, then adds
    an input-only gravity anchor for high-confidence quasi-static samples. This
    targets the failure mode seen on OxIOD: device/use-pattern shifts where
    gravity should be highly informative but the learned model drifts away from
    the accelerometer cue. The gravity gate never uses test labels; it depends
    only on acceleration norm, acceleration jerk, angular rate, and finite input
    checks.
    """

    def __init__(
        self,
        dyn_weight: float = 0.10,
        gravity_weight: float = 0.04,
        target_gate_deg: float = 2.0,
        dyn_deadzone_deg: float = 0.15,
        dyn_delta_deg: float = 0.5,
        gravity_deadzone_deg: float = 0.20,
        gravity_delta_deg: float = 0.5,
        gravity_sigma: float = 0.45,
        jerk_sigma: float = 45.0,
        rate_sigma: float = 1.4,
        gate_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dyn_weight = float(dyn_weight)
        self.gravity_weight = float(gravity_weight)
        self.target_gate_rad = float(torch.deg2rad(torch.tensor(target_gate_deg)))
        self.dyn_deadzone_rad = float(torch.deg2rad(torch.tensor(dyn_deadzone_deg)))
        self.dyn_delta_rad = float(torch.deg2rad(torch.tensor(dyn_delta_deg)))
        self.gravity_deadzone_rad = float(torch.deg2rad(torch.tensor(gravity_deadzone_deg)))
        self.gravity_delta_rad = float(torch.deg2rad(torch.tensor(gravity_delta_deg)))
        self.gravity_sigma = float(gravity_sigma)
        self.jerk_sigma = float(jerk_sigma)
        self.rate_sigma = float(rate_sigma)
        self.gate_floor = float(gate_floor)

    def forward(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        if self.dyn_weight == 0.0 and self.gravity_weight == 0.0:
            return pred.sum() * 0.0
        pred = qnorm(pred)
        q_target, valid = finite_target_quat(yb, pred)
        raw = xb[..., :7].to(pred.dtype)
        acc, gyr, dt = raw[..., :3], raw[..., 3:6], raw[..., 6:7].clamp_min(1e-6)
        dyn = self._target_gated_vertical_dynamics(pred, q_target, valid, gyr, dt)
        grav = self._reliable_gravity_anchor(pred, acc, gyr, dt)
        return (self.dyn_weight * dyn + self.gravity_weight * grav).to(pred.dtype)

    def _target_gated_vertical_dynamics(
        self,
        pred: torch.Tensor,
        q_target: torch.Tensor,
        valid: torch.Tensor,
        gyr: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        if pred.shape[-2] < 2:
            return pred.sum() * 0.0
        pair_mask = valid[..., :-1] & valid[..., 1:]
        pair_mask = pair_mask & torch.isfinite(gyr[..., :-1, :]).all(dim=-1)
        pair_mask = pair_mask & torch.isfinite(dt[..., :-1, 0])

        q_target_prop = gyro_step(q_target[..., :-1, :], gyr[..., :-1, :], dt[..., :-1, :])
        target_tilt_err = inclination_error_rad(q_target_prop, q_target[..., 1:, :])
        pair_mask = pair_mask & (target_tilt_err <= self.target_gate_rad)

        q_pred_prop = gyro_step(pred[..., :-1, :], gyr[..., :-1, :], dt[..., :-1, :])
        h_prop = gravity_dir_body(q_pred_prop)
        h_next = gravity_dir_body(pred[..., 1:, :])
        err = angle_between_unit(h_prop, h_next)
        residual = (err - self.dyn_deadzone_rad).clamp_min(0.0)
        penalty = robust_charbonnier(residual, eps=self.dyn_delta_rad)
        return masked_mean(penalty, pair_mask).to(pred.dtype)

    def _reliable_gravity_anchor(
        self,
        pred: torch.Tensor,
        acc: torch.Tensor,
        gyr: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        acc_norm = vec_norm(acc)
        gyro_rate = vec_norm(gyr)
        jerk_score = torch.zeros_like(acc_norm)
        if acc.shape[-2] > 1:
            jerk = vec_norm(acc[..., 1:, :] - acc[..., :-1, :]) / dt[..., :-1, 0].clamp_min(1e-6)
            jerk_score[..., 1:] = torch.maximum(jerk_score[..., 1:], jerk)
            jerk_score[..., :-1] = torch.maximum(jerk_score[..., :-1], jerk)

        finite = (
            torch.isfinite(acc).all(dim=-1)
            & torch.isfinite(gyr).all(dim=-1)
            & torch.isfinite(acc_norm)
            & torch.isfinite(gyro_rate)
            & torch.isfinite(dt[..., 0])
        )
        acc_gate = torch.exp(-((acc_norm - G).square()) / (2.0 * self.gravity_sigma**2))
        jerk_gate = torch.exp(-(jerk_score.square()) / (2.0 * self.jerk_sigma**2))
        rate_gate = torch.exp(-(gyro_rate.square()) / (2.0 * self.rate_sigma**2))
        gate_weight = torch.where(
            finite, acc_gate * jerk_gate * rate_gate, torch.zeros_like(acc_gate)
        )
        gate = gate_weight > self.gate_floor

        acc_unit = acc / acc_norm.clamp_min(EPS).unsqueeze(-1)
        h_pred = gravity_dir_body(pred)
        err = angle_between_unit(h_pred, acc_unit)
        residual = (err - self.gravity_deadzone_rad).clamp_min(0.0)
        penalty = robust_charbonnier(residual, eps=self.gravity_delta_rad)
        return masked_mean(gate_weight * penalty, gate).to(pred.dtype)


class RiannV2Loss(nn.Module):
    """Supervised tilt-only loss for IMU orientation without magnetometer.

    Only inclination (roll+pitch) is penalized — yaw is unobservable from
    acc+gyro alone, so penalizing it injects pure gradient noise and halves
    convergence speed. The dynamics consistency loss supplies yaw-rate signal
    separately via gyroscope integration (which measures all 3 axes).
    """

    def __init__(self, inc_weight: float = 1.0, norm_weight: float = 0.01) -> None:
        super().__init__()
        self.inc_weight = float(inc_weight)
        self.norm_weight = float(norm_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        q_pred = qnorm(pred)
        q_target, valid = finite_target_quat(target, q_pred)
        inc = robust_charbonnier(inclination_error_rad(q_pred, q_target))
        norm = (vec_norm(pred) - 1.0).square()
        return (self.inc_weight * masked_mean(inc, valid) + self.norm_weight * norm.mean()).to(
            pred.dtype
        )


def masked_tilt_mean_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_pred = qnorm(pred)
    q_target, mask = finite_target_quat(target, q_pred)
    err = inclination_error_rad(q_pred, q_target)
    if mask.any():
        return torch.rad2deg(err[mask].mean()).to(pred.dtype)
    return (pred.sum() * 0.0).to(pred.dtype)


def masked_tilt_rmse_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_pred = qnorm(pred)
    q_target, mask = finite_target_quat(target, q_pred)
    err = inclination_error_rad(q_pred, q_target)
    if mask.any():
        return torch.rad2deg(err[mask].square().mean().sqrt()).to(pred.dtype)
    return (pred.sum() * 0.0).to(pred.dtype)


def roll_pitch_deg(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = qnorm(q)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0 + EPS, 1.0 - EPS))
    return torch.rad2deg(roll), torch.rad2deg(pitch)


def circular_rmse_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    err = torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0
    return err.square().mean().sqrt()


def masked_roll_rmse_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_pred = qnorm(pred)
    q_target, mask = finite_target_quat(target, q_pred)
    pred_roll, _ = roll_pitch_deg(q_pred)
    target_roll, _ = roll_pitch_deg(q_target)
    if mask.any():
        return circular_rmse_deg(pred_roll[mask], target_roll[mask]).to(pred.dtype)
    return (pred.sum() * 0.0).to(pred.dtype)


def masked_pitch_rmse_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_pred = qnorm(pred)
    q_target, mask = finite_target_quat(target, q_pred)
    _, pred_pitch = roll_pitch_deg(q_pred)
    _, target_pitch = roll_pitch_deg(q_target)
    if mask.any():
        return circular_rmse_deg(pred_pitch[mask], target_pitch[mask]).to(pred.dtype)
    return (pred.sum() * 0.0).to(pred.dtype)


def masked_geodesic_mean_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_pred = qnorm(pred)
    q_target, mask = finite_target_quat(target, q_pred)
    err = geodesic_error_rad(q_pred, q_target)
    if mask.any():
        return torch.rad2deg(err[mask].mean()).to(pred.dtype)
    return (pred.sum() * 0.0).to(pred.dtype)
