"""Tests for src.disease.gradcam (Phase 5 §F)."""

from __future__ import annotations

import pytest


def test_compute_gradcam_returns_expected_shape() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("pytorch_grad_cam")
    import numpy as np

    from src.disease.gradcam import compute_gradcam
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=10, pretrained=False).eval()
    image_tensor = torch.randn(1, 3, 380, 380)
    cam = compute_gradcam(model, image_tensor, target_class=3)
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    # H, W match the input.
    assert cam.shape == (380, 380)
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6


def test_compute_gradcam_default_target_is_argmax() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("pytorch_grad_cam")
    from src.disease.gradcam import compute_gradcam
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=5, pretrained=False).eval()
    image_tensor = torch.randn(3, 224, 224)  # 3D input also works
    cam = compute_gradcam(model, image_tensor)  # no target_class
    assert cam.shape == (224, 224)


def test_overlay_renders_uint8_rgb() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from src.disease.gradcam import overlay_gradcam_on_image

    # 96x96 fake image + matching CAM
    img = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    cam = np.linspace(0, 1, 96 * 96, dtype=np.float32).reshape(96, 96)
    overlay = overlay_gradcam_on_image(img, cam, alpha=0.5)
    assert overlay.dtype == np.uint8
    assert overlay.shape == (96, 96, 3)
    assert overlay.min() >= 0 and overlay.max() <= 255
