"""Tests for :mod:`src.soil.transforms_v2` — strong train aug + TTA views."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")
pytest.importorskip("cv2")

from src.soil.transforms_v2 import build_soil_train_aug_v2, build_tta_views  # noqa: E402


_MEAN = (0.534, 0.459, 0.400)
_STD = (0.216, 0.200, 0.210)


def _fake_image(size: int = 224) -> np.ndarray:
    """A reproducible HWC uint8 numpy image for albumentations input."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_build_soil_train_aug_v2_returns_callable_and_preserves_shape() -> None:
    aug = build_soil_train_aug_v2(224, _MEAN, _STD)
    assert callable(aug)
    out = aug(image=_fake_image(224))["image"]
    # Albumentations + ToTensorV2 -> torch.Tensor of shape (3, 224, 224).
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_build_tta_views_returns_five_composes() -> None:
    views = build_tta_views(224, _MEAN, _STD)
    assert len(views) == 5


def test_tta_views_all_produce_same_shape_and_distinct_tensors() -> None:
    views = build_tta_views(224, _MEAN, _STD)
    img = _fake_image(224)
    outputs = [v(image=img)["image"] for v in views]
    for t in outputs:
        assert t.shape == (3, 224, 224)
        assert t.dtype == torch.float32
    # At least one pair must differ: the 5 views aren't identical.
    distinct = False
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if not torch.allclose(outputs[i], outputs[j]):
                distinct = True
                break
        if distinct:
            break
    assert distinct, "All 5 TTA views produced identical tensors — pipeline is broken."


def test_train_aug_v2_is_stochastic() -> None:
    aug = build_soil_train_aug_v2(224, _MEAN, _STD)
    img = _fake_image(224)
    a = aug(image=img)["image"]
    b = aug(image=img)["image"]
    # Strong augmentation must randomise — two passes shouldn't be identical.
    assert not torch.allclose(a, b)
