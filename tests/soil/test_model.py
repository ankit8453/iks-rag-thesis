"""Phase 6 smoke tests for :class:`src.soil.model.SoilMultiTaskClassifier`.

Verifies (a) construction with the locked B0 backbone succeeds,
(b) forward returns a dict with the three expected per-head keys and
shapes, and (c) ``freeze_backbone()`` / ``unfreeze_backbone()`` toggle
``requires_grad`` on the timm backbone as advertised.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.soil.model import SoilMultiTaskClassifier  # noqa: E402


def _build_no_pretrained() -> SoilMultiTaskClassifier:
    """Construct without pretrained weights so the test stays offline."""
    return SoilMultiTaskClassifier(
        backbone_name="efficientnet_b0", pretrained=False, dropout=0.3,
    )


def test_construct_b0_backbone() -> None:
    model = _build_no_pretrained()
    assert model.backbone_name == "efficientnet_b0"
    # Backbone (~4.0M) + heads (~17K). Total in [3.5M, 4.5M] is sane for B0
    # with num_classes=0 and three small heads.
    total = sum(p.numel() for p in model.parameters())
    assert 3_500_000 <= total <= 4_500_000, (
        f"Unexpected total parameter count {total:,}; expected ~4.02M for B0."
    )


def test_forward_returns_three_head_dict_with_correct_shapes() -> None:
    model = _build_no_pretrained()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"soil_type", "moisture", "texture"}
    assert out["soil_type"].shape == (2, 7)
    assert out["moisture"].shape == (2, 3)
    assert out["texture"].shape == (2, 3)


def test_freeze_backbone_turns_off_grad_on_backbone_only() -> None:
    model = _build_no_pretrained()
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # Heads must still be trainable so the warmup epochs do something.
    for head in (model.soil_type_head, model.moisture_head, model.texture_head):
        assert all(p.requires_grad for p in head.parameters())


def test_unfreeze_backbone_turns_grad_back_on_everywhere() -> None:
    model = _build_no_pretrained()
    model.freeze_backbone()
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.parameters())
