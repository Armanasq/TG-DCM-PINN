from __future__ import annotations

import math

import torch

from tg_dcm.losses import TargetDCM3Loss, gravity_dir_body, qexp


def _roll_quaternion(degrees: float) -> torch.Tensor:
    half = math.radians(degrees) / 2.0
    return torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0])


def test_identity_quaternion_preserves_vertical_direction() -> None:
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(gravity_dir_body(quaternion), expected)


def test_zero_rotation_vector_maps_to_identity_quaternion() -> None:
    rotation = torch.zeros(2, 3)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    torch.testing.assert_close(qexp(rotation), expected)


def test_dynamics_branch_rejects_target_gyro_mismatch() -> None:
    loss = TargetDCM3Loss(dyn_weight=1.0, gravity_weight=0.0)
    prediction = torch.stack((torch.tensor([1.0, 0.0, 0.0, 0.0]), _roll_quaternion(10.0)))[None]
    target = torch.stack((torch.tensor([1.0, 0.0, 0.0, 0.0]), _roll_quaternion(20.0)))[None]
    inputs = torch.zeros(1, 2, 7)
    inputs[..., 6] = 0.01

    value = loss(prediction, target, inputs)

    assert value.item() == 0.0


def test_dynamics_branch_penalizes_accepted_inconsistent_prediction() -> None:
    loss = TargetDCM3Loss(dyn_weight=1.0, gravity_weight=0.0)
    prediction = torch.stack((torch.tensor([1.0, 0.0, 0.0, 0.0]), _roll_quaternion(10.0)))[None]
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    inputs = torch.zeros(1, 2, 7)
    inputs[..., 6] = 0.01

    value = loss(prediction, target, inputs)

    assert value.item() > 0.0


def test_gravity_branch_uses_calm_acceleration_and_rejects_implausible_magnitude() -> None:
    loss = TargetDCM3Loss(dyn_weight=0.0, gravity_weight=1.0, gravity_sigma=0.25)
    prediction = _roll_quaternion(10.0).reshape(1, 1, 4)
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

    calm = torch.zeros(1, 1, 7)
    calm[..., 2] = 9.81
    calm[..., 6] = 0.01
    assert loss(prediction, target, calm).item() > 0.0

    implausible = calm.clone()
    implausible[..., 2] = 20.0
    assert loss(prediction, target, implausible).item() == 0.0
