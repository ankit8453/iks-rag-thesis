"""Smoke tests for the soil module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.soil import SoilClassifier, SoilConfig


def test_soil_config_defaults() -> None:
    cfg = SoilConfig()
    assert cfg.backbone == "efficientnet_b0"
    assert "soil_type" in cfg.multi_task_heads
    assert "npk" in cfg.disallowed_outputs
    assert cfg.cross_region_validation is True


def test_soil_classifier_instantiation() -> None:
    cfg = SoilConfig()
    clf = SoilClassifier(cfg)
    assert clf.config is cfg


def test_soil_classifier_has_docstring_with_visual_only_warning() -> None:
    """Guardrail #2: the docstring must remind future maintainers."""
    doc = SoilClassifier.__doc__ or ""
    assert "VISUAL ONLY" in doc or "visual" in doc.lower()


def test_soil_config_rejects_unknown_disallowed_output() -> None:
    """Adding a nonsense chemical key to disallowed_outputs fails validation."""
    with pytest.raises(ValidationError):
        SoilConfig(disallowed_outputs=["npk", "made_up_attribute"])


def test_soil_config_rejects_empty_heads() -> None:
    with pytest.raises(ValidationError):
        SoilConfig(multi_task_heads=[])
