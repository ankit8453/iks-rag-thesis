"""Smoke tests for the disease module."""

from __future__ import annotations

from src.disease import DiseaseClassifier, DiseaseConfig


def test_disease_config_defaults() -> None:
    cfg = DiseaseConfig()
    assert cfg.backbone == "efficientnet_b4"
    assert cfg.num_classes == 38
    assert cfg.image_size == 380
    assert cfg.augmentation.normalize_mean == (0.485, 0.456, 0.406)


def test_disease_classifier_instantiation() -> None:
    cfg = DiseaseConfig()
    clf = DiseaseClassifier(cfg)
    assert clf.config is cfg


def test_disease_classifier_has_docstring() -> None:
    assert DiseaseClassifier.__doc__ is not None
    assert "EfficientNet-B4" in DiseaseClassifier.__doc__
