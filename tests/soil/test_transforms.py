"""Tests for src.soil.transforms."""

from __future__ import annotations

import pytest


def _make_dummy_image(size: int = 288):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_train_pipeline_shape() -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from src.soil.transforms import build_soil_train_aug

    img = _make_dummy_image()
    aug = build_soil_train_aug(224, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out = aug(image=img)["image"]
    assert tuple(out.shape) == (3, 224, 224)


def test_eval_pipeline_is_deterministic() -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    from src.soil.transforms import build_soil_eval_aug

    img = _make_dummy_image()
    aug = build_soil_eval_aug(224, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out1 = aug(image=img)["image"]
    out2 = aug(image=img)["image"]
    assert torch.equal(out1, out2)
    assert tuple(out1.shape) == (3, 224, 224)
