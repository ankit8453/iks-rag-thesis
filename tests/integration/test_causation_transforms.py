"""Tests for src.integration.causation_transforms.

Importantly: the train pipeline must NOT include colour augmentations
(ColorJitter, RandomBrightnessContrast, ToGray, HueSaturationValue),
because nutrient-deficiency labels in OLID I are hue-cued.
"""

from __future__ import annotations

import pytest


def _make_dummy_image(size: int = 512):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_train_pipeline_shape() -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from src.integration.causation_transforms import build_olid_train_aug

    img = _make_dummy_image()
    aug = build_olid_train_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out = aug(image=img)["image"]
    assert tuple(out.shape) == (3, 380, 380)


def test_train_pipeline_contains_no_colour_augmentation() -> None:
    pytest.importorskip("albumentations")
    from src.integration.causation_transforms import build_olid_train_aug

    aug = build_olid_train_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    transform_names = {t.__class__.__name__ for t in aug.transforms}
    forbidden = {
        "ColorJitter",
        "RandomBrightnessContrast",
        "HueSaturationValue",
        "ToGray",
        "RGBShift",
        "ChannelShuffle",
    }
    assert transform_names.isdisjoint(forbidden), (
        f"OLID train aug contains a colour transform that would erase the "
        f"hue-cued nutrient-deficiency signal: "
        f"{transform_names & forbidden}"
    )


def test_eval_pipeline_is_deterministic() -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    from src.integration.causation_transforms import build_olid_eval_aug

    img = _make_dummy_image()
    aug = build_olid_eval_aug(380, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    out1 = aug(image=img)["image"]
    out2 = aug(image=img)["image"]
    assert torch.equal(out1, out2)
