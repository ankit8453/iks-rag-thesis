"""Scope-gate tests — the system's honest "which plants do I support?" boundary.

The classifier has a fixed class set and cannot answer "I don't know" — given an
untrained plant it still returns its closest trained class. So support is decided
from the farmer's declared plant against the list derived from the model's OWN
class names, never inferred from the model's output.
"""

from __future__ import annotations

import json
from pathlib import Path

from app import config as app_config

_CLASS_MAP = Path(__file__).resolve().parent.parent / "data/splits/plantdoc/class_map.json"


def _trained_class_names() -> list[str]:
    return list(json.loads(_CLASS_MAP.read_text(encoding="utf-8")).keys())


def test_supported_crops_are_derived_from_the_models_own_classes() -> None:
    """The list must come from the checkpoint's classes so it can't drift."""
    crops = app_config.supported_crops(_trained_class_names())
    # every supported crop must be reachable from at least one trained class
    from_classes = {app_config.crop_from_disease(n) for n in _trained_class_names()}
    assert set(crops) == from_classes
    assert crops == sorted(crops), "list should be stable/sorted for the dropdown"


def test_supported_crops_covers_the_expected_plants() -> None:
    crops = app_config.supported_crops(_trained_class_names())
    for expected in ("apple", "corn", "potato", "tomato", "grape", "bell pepper"):
        assert expected in crops
    # things we never trained on must NOT appear
    for absent in ("brinjal", "rice", "wheat", "mango"):
        assert absent not in crops


def test_supported_crops_handles_empty_and_blank_input() -> None:
    assert app_config.supported_crops([]) == []
    assert app_config.supported_crops(["", None]) == []  # type: ignore[list-item]


def test_other_sentinel_is_not_a_real_crop() -> None:
    """The 'Other' entry must never collide with a supported plant name."""
    crops = app_config.supported_crops(_trained_class_names())
    assert app_config.OTHER_CROP not in crops


def test_disease_type_strips_the_crop_word() -> None:
    """On an untrained plant we must show the DISEASE, not the wrong plant."""
    assert app_config.disease_type_from_class("Tomato Septoria leaf spot") == "Septoria leaf spot"
    assert app_config.disease_type_from_class("Corn rust leaf") == "rust leaf"
    assert app_config.disease_type_from_class("Bell_pepper leaf spot") == "leaf spot"
    # single-token label falls back to itself, never empty
    assert app_config.disease_type_from_class("Healthy") == "Healthy"


def test_untrained_and_low_confidence_messages_format() -> None:
    caution = app_config.UNTRAINED_PLANT_CAUTION.format(
        plant="brinjal", disease="leaf spot", conf=0.63)
    assert "brinjal" in caution and "leaf spot" in caution and "63%" in caution

    low = app_config.LOW_CONFIDENCE_MESSAGE.format(
        conf=0.31, floor=app_config.CONFIDENCE_ADVISE_MIN,
        n=app_config.RETRAIN_IMAGES_REQUESTED)
    assert "31%" in low and str(app_config.RETRAIN_IMAGES_REQUESTED) in low


def test_confidence_threshold_is_sane() -> None:
    assert 0.0 < app_config.CONFIDENCE_ADVISE_MIN < 1.0
    assert app_config.RETRAIN_IMAGES_REQUESTED >= 1


def test_scope_and_mismatch_messages_format() -> None:
    out = app_config.OUT_OF_SCOPE_MESSAGE.format(crop="Brinjal")
    assert "Brinjal" in out
    assert "will not guess" in out          # we must promise not to guess

    mm = app_config.CROP_MISMATCH_MESSAGE.format(selected="tomato", detected="potato")
    assert "tomato" in mm and "potato" in mm
