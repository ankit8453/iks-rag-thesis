"""Tests for src.utils.data_validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.data_validators import (
    ImageValidationReport,
    validate_image_directory,
    write_corrupt_list,
)


def _write_valid_rgb(path: Path, size: tuple[int, int] = (96, 96)) -> None:
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", size, color=(128, 64, 200))
    img.save(path, format="JPEG")


def _write_valid_grayscale(path: Path, size: tuple[int, int] = (96, 96)) -> None:
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("L", size, color=128)
    img.save(path, format="PNG")


def _write_too_small(path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (32, 32), color=(0, 0, 0))
    img.save(path, format="JPEG")


def _write_zero_byte(path: Path) -> None:
    path.write_bytes(b"")


def _write_truncated_jpeg(path: Path) -> None:
    """Write only the leading bytes of a JPEG — header without payload."""
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")


def test_validator_handles_mixed_directory(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()

    _write_valid_rgb(tmp_path / "good" / "rgb.jpg")
    _write_valid_grayscale(tmp_path / "good" / "gray.png")
    _write_too_small(tmp_path / "bad" / "tiny.jpg")
    _write_zero_byte(tmp_path / "bad" / "empty.jpg")
    _write_truncated_jpeg(tmp_path / "bad" / "trunc.jpg")
    # A non-image file should be ignored entirely.
    (tmp_path / "good" / "notes.txt").write_text("not an image")

    report = validate_image_directory(tmp_path, "fixture")
    assert isinstance(report, ImageValidationReport)
    assert report.total_files == 5  # only the 5 image-extension files
    assert report.valid_files == 2
    assert set(report.corrupt_files) == {
        "bad/tiny.jpg",
        "bad/empty.jpg",
        "bad/trunc.jpg",
    }
    for path in report.corrupt_files:
        assert path in report.failure_reasons


def test_validator_raises_on_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_image_directory(tmp_path / "does_not_exist", "x")


def test_validator_empty_directory(tmp_path: Path) -> None:
    report = validate_image_directory(tmp_path, "empty")
    assert report.total_files == 0
    assert report.valid_files == 0
    assert report.corrupt_files == []


def test_write_corrupt_list_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _write_zero_byte(tmp_path / "bad.jpg")
    _write_valid_rgb(tmp_path / "good.jpg")

    report = validate_image_directory(tmp_path, "demo")
    out = write_corrupt_list(report, tmp_path / "results")
    assert out is not None and out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "bad.jpg" in body
    assert "good.jpg" not in body


def test_write_corrupt_list_returns_none_when_clean(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _write_valid_rgb(tmp_path / "good.jpg")

    report = validate_image_directory(tmp_path, "demo")
    assert write_corrupt_list(report, tmp_path / "results") is None
