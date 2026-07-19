from __future__ import annotations

import zipfile

import pytest

from tg_dcm.prep._common import extract_archive


def test_zip_extraction_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "not allowed")

    with pytest.raises(ValueError, match="escapes destination"):
        extract_archive(archive, tmp_path / "output")

    assert not (tmp_path / "outside.txt").exists()


def test_zip_extraction_accepts_regular_members(tmp_path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/sample.txt", "imu")

    output = extract_archive(archive, tmp_path / "output")

    assert (output / "dataset" / "sample.txt").read_text() == "imu"
