import json
import numpy as np


def rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))


def run():
    g = 9.80665
    T = 2.0
    samples = 4096
    dt = T / samples
    t = (np.arange(samples) + 0.5) * dt
    e3 = np.array([0.0, 0.0, 1.0])
    rel_rot = np.array([rotation_z(0.8 * np.sin(2 * np.pi * x / T)) @ rotation_y(0.5 * np.sin(np.pi * x / T)) for x in t])
    accel_world = np.column_stack([7 * np.sin(2 * np.pi * t / T), 3 * np.sin(4 * np.pi * t / T), 2 * np.sin(2 * np.pi * t / T)])
    force_body = np.einsum("nij,nj->ni", rel_rot.transpose(0, 2, 1), accel_world + g * e3)
    integrated = np.einsum("nij,nj->i", rel_rot, force_body) * dt
    tilted = integrated / np.linalg.norm(integrated)
    instantaneous = [angle_deg(rel_rot[k] @ force_body[k], e3) for k in range(samples)]
    velocity_delta = accel_world.sum(axis=0) * dt
    vmax = 0.5
    r = g * T
    rho = np.linalg.norm(integrated)
    cosine = (rho * rho + r * r - vmax * vmax) / (2 * rho * r)
    radius_deg = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    phi = np.radians(30.0)
    misleading_force = g * np.array([np.sin(phi), 0.0, np.cos(phi)])
    misleading_accel = misleading_force - g * e3
    misleading_integrated = misleading_force * T
    false_closure_norm_residual = abs(np.linalg.norm(misleading_integrated) - r)
    q = rotation_z(0.37) @ rotation_y(-0.43)
    rel_prime = np.einsum("ab,nbc,cd->nad", q, rel_rot, q.T)
    force_prime = np.einsum("ab,nb->na", q, force_body)
    integrated_prime = np.einsum("nij,nj->i", rel_prime, force_prime) * dt
    proper_axis_equivariance_residual = float(np.linalg.norm(integrated_prime - q @ integrated))
    output = {
        "assumptions": "noise-free relative rotations, calibrated specific force, sensor-point velocity closure",
        "duration_s": T,
        "midpoint_samples": samples,
        "dynamic_close_interval": {
            "delta_velocity_norm_m_s": float(np.linalg.norm(velocity_delta)),
            "tilt_error_from_integral_deg": angle_deg(tilted, e3),
            "max_instantaneous_apparent_tilt_deg": max(instantaneous),
            "radius_given_0_5m_s_closure_bound_deg": radius_deg,
        },
        "false_closure_case": {
            "instantaneous_force_norm_m_s2": float(np.linalg.norm(misleading_force)),
            "closure_norm_residual_m_s": false_closure_norm_residual,
            "real_delta_velocity_norm_m_s": float(np.linalg.norm(misleading_accel) * T),
            "false_assumed_tilt_deg": angle_deg(misleading_force, e3),
        },
        "coordinate_rotation_equivariance_residual": proper_axis_equivariance_residual,
    }
    assert output["dynamic_close_interval"]["tilt_error_from_integral_deg"] < 1e-10
    assert output["dynamic_close_interval"]["max_instantaneous_apparent_tilt_deg"] > 20.0
    assert radius_deg > 0.0
    assert false_closure_norm_residual < 1e-10
    assert output["false_closure_case"]["real_delta_velocity_norm_m_s"] > 10.0
    assert proper_axis_equivariance_residual < 1e-10
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
