from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_release_checkpoint_is_weights_only_compatible() -> None:
    checkpoint = torch.load(
        ROOT / "checkpoints" / "tg_dcm.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["schema_version"] == 1
    assert checkpoint["epoch"] == 491
    assert checkpoint["source_weights"] == "ema"
    assert checkpoint["config"]["quaternion_order"] == "wxyz"
    assert checkpoint["config"]["quaternion_direction"] == "body_to_world"
    assert set(checkpoint["model"]) == set(checkpoint["ema"])
    assert all(torch.isfinite(value).all() for value in checkpoint["ema"].values())
