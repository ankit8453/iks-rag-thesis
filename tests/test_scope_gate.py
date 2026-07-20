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


def test_scope_and_mismatch_messages_format() -> None:
    out = app_config.OUT_OF_SCOPE_MESSAGE.format(crop="Brinjal")
    assert "Brinjal" in out
    assert "will not guess" in out          # we must promise not to guess

    mm = app_config.CROP_MISMATCH_MESSAGE.format(selected="tomato", detected="potato")
    assert "tomato" in mm and "potato" in mm
