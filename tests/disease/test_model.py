"""Tests for src.disease.model.DiseaseClassifier (Phase 5 §D).

The DiseaseClassifier is the real EfficientNet-B4 wrapper from this
phase on. These tests instantiate the model with ``pretrained=False``
so they don't hit the timm pretrained-weights cache on a clean CI
machine.
"""

from __future__ import annotations

import pytest


def test_disease_classifier_constructs_without_error() -> None:
    pytest.importorskip("timm")
    pytest.importorskip("torch")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=38, pretrained=False)
    assert model.num_classes == 38
    assert model.pretrained is False
    assert model.dropout_rate == 0.3


def test_forward_returns_correct_logits_shape() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=38, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 380, 380))
    assert logits.shape == (2, 38)


def test_freeze_backbone_leaves_only_head_trainable() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=10, pretrained=False)
    total = model.total_param_count()
    head_only = model.freeze_backbone()

    head_param_count = sum(p.numel() for p in model.head.parameters())
    assert head_only == head_param_count, (
        f"freeze_backbone left {head_only} trainable; expected only the "
        f"head ({head_param_count})."
    )
    # Backbone should be a tiny fraction trainable (zero in fact).
    backbone_trainable = sum(
        p.numel()
        for p in model.get_feature_extractor().parameters()
        if p.requires_grad
    )
    assert backbone_trainable == 0
    # Sanity: head is much smaller than the full model.
    assert head_only < total / 10


def test_unfreeze_backbone_restores_all_grads() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=10, pretrained=False)
    model.freeze_backbone()
    full_trainable = model.unfreeze_backbone()
    assert full_trainable == model.total_param_count()


def test_seed_reproducibility() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier
    from src.utils.seeding import set_global_seed

    set_global_seed(42)
    m1 = DiseaseClassifier(num_classes=10, pretrained=False)
    set_global_seed(42)
    m2 = DiseaseClassifier(num_classes=10, pretrained=False)

    # Compare every parameter tensor — same seed must produce identical
    # random initialisation.
    pairs = list(zip(m1.parameters(), m2.parameters(), strict=True))
    assert len(pairs) > 0
    for p1, p2 in pairs:
        assert torch.equal(p1, p2), "Same seed produced different weights"


def test_get_feature_extractor_returns_backbone() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    model = DiseaseClassifier(num_classes=10, pretrained=False)
    feat = model.get_feature_extractor()
    # Backbone is the timm efficientnet_b4 — should have a num_features attr.
    assert hasattr(feat, "num_features")
    assert feat.num_features > 0


def test_state_dict_round_trip() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.disease.model import DiseaseClassifier

    m1 = DiseaseClassifier(num_classes=10, pretrained=False)
    sd = m1.state_dict()
    m2 = DiseaseClassifier(num_classes=10, pretrained=False)
    m2.load_state_dict(sd)
    # The two models should now produce identical outputs.
    m1.eval(); m2.eval()
    x = torch.randn(1, 3, 380, 380)
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)
    assert torch.allclose(out1, out2, atol=1e-6)
