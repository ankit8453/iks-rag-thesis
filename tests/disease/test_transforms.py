"""Tests for src.disease.transforms — output shape, dtype, value range."""

from __future__ import annotations

import pytest


def _make_dummy_image(size: int = 512):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_train_pipeline_produces_3xHxW_tensor() -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from src.disease.transforms import build_disease_train_aug

    img = _make_dummy_image()
    aug = build_disease_train_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out = aug(image=img)["image"]
    assert tuple(out.shape) == (3, 380, 380)


def test_eval_pipeline_is_deterministic_and_correct_shape() -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from src.disease.transforms import build_disease_eval_aug

    img = _make_dummy_image()
    aug = build_disease_eval_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out1 = aug(image=img)["image"]
    out2 = aug(image=img)["image"]
    assert tuple(out1.shape) == (3, 380, 380)
    import torch

    assert torch.equal(out1, out2), "Eval pipeline must be deterministic."


def test_normalised_values_roughly_within_three_sigma() -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    from src.disease.transforms import build_disease_eval_aug

    img = _make_dummy_image()
    aug = build_disease_eval_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out = aug(image=img)["image"]
    # After dividing by 255 and (x - 0.5) / 0.25, values are in roughly
    # [-2, 2]. Allow some slack.
    assert out.min().item() > -3.5
    assert out.max().item() < 3.5
