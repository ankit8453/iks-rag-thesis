"""Smoke tests for the disease module.

These were authored in Phase 4 against the Week-2 stub
(``DiseaseClassifier(config)``). Phase 5 §D rewrote the class as a
real EfficientNet-B4 wrapper with constructor
``DiseaseClassifier(num_classes, pretrained, dropout_rate)``. The
smoke tests have been updated to that signature; deeper coverage
lives in :mod:`tests.disease.test_model`.
"""

from __future__ import annotations

import pytest


def test_disease_config_defaults() -> None:
    from src.disease import DiseaseConfig

    cfg = DiseaseConfig()
    assert cfg.backbone == "efficientnet_b4"
    assert cfg.image_size == 380
    # num_classes is now optional (derived per-dataset by the loader).
    assert cfg.num_classes is None
    # Stage budgets locked in Phase 5 §D.
    assert cfg.pretrain_epochs == 25
    assert cfg.finetune_paddy_epochs == 20
    assert cfg.finetune_plantdoc_epochs == 30
    assert cfg.freeze_backbone_epochs == 3
    assert cfg.augmentation.normalize_mean == (0.485, 0.456, 0.406)


def test_disease_classifier_instantiation() -> None:
    pytest.importorskip("timm")
    pytest.importorskip("torch")
    from src.disease import DiseaseClassifier

    clf = DiseaseClassifier(num_classes=10, pretrained=False)
    assert clf.num_classes == 10


def test_disease_classifier_has_docstring() -> None:
    from src.disease import DiseaseClassifier

    assert DiseaseClassifier.__doc__ is not None
    assert "EfficientNet-B4" in DiseaseClassifier.__doc__
