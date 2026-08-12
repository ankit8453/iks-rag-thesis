"""Tests for the crop-agnostic disease-type mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.disease.disease_type_map import (
    CANONICAL_TYPES,
    is_disease,
    to_disease_type,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def _classes(name: str) -> list[str]:
    return list(json.loads((ROOT / f"data/splits/{name}/class_map.json").read_text()).keys())


# ------------------------------------------------------------------ #
# ordering-sensitive cases (specific beats generic)
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("cls,expected", [
    ("Tomato___Bacterial_spot", "bacterial"),         # bacterial before 'spot'
    ("Potato___Early_blight", "early_blight"),         # early before generic blight
    ("Potato___Late_blight", "late_blight"),
    ("Corn_(maize)___Northern_Leaf_Blight", "blight"),
    ("bacterial_leaf_blight", "bacterial"),            # bacterial before 'blight'
    ("Squash___Powdery_mildew", "powdery_mildew"),
    ("Paddy downy_mildew", "downy_mildew"),
    ("Tomato___Leaf_Mold", "leaf_mold"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "mosaic_virus"),
    ("Apple___Cedar_apple_rust", "rust"),
    ("Apple___Apple_scab", "scab"),
    ("Grape___Black_rot", "rot"),
    ("Tomato___Septoria_leaf_spot", "leaf_spot"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "pest_damage"),
])
def test_specific_beats_generic(cls: str, expected: str) -> None:
    assert to_disease_type(cls) == expected


# ------------------------------------------------------------------ #
# healthy detection across naming styles
# ------------------------------------------------------------------ #


def test_healthy_names_all_map_to_healthy() -> None:
    for cls in ["Apple___healthy", "normal", "Apple leaf", "grape leaf",
                "Soja (Soybean) - Saudavel (Healthy) - 1",
                "Citros (Citrus) - Sadia (Healthy) - 1"]:
        assert to_disease_type(cls) == "healthy", cls


def test_plaindoc_healthy_leaves_are_not_other() -> None:
    """Regression: 'Apple leaf' etc. must be healthy, not 'other'."""
    for cls in _classes("plantdoc"):
        if cls.strip().lower().endswith(" leaf") and " " not in cls.strip()[:-5].strip():
            assert to_disease_type(cls) == "healthy", cls


# ------------------------------------------------------------------ #
# portuguese labels (Dr. Pandey's dataset)
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("cls,expected", [
    ("Café (Coffee) - Ferrugem (Rust) - 1", "rust"),
    ("Soja (Soybean) - Oidio (Powdery mildew) - 1", "powdery_mildew"),
    ("Videira (Grapevine) - Mildio (Downy Mildew) - 1", "downy_mildew"),
    ("Feijao (Dry bean) - Antracnose (Anthracnose) - 1", "anthracnose"),
    ("Maracuja (Passion Fruit) - Mancha Bacteriana (Bacterial spot) - 1", "bacterial"),
    ("Trigo (Wheat) - Brusone (Wheat blast) - 1", "blight"),      # blast -> blight
])
def test_portuguese_labels_map(cls: str, expected: str) -> None:
    assert to_disease_type(cls) == expected


def test_spot_blight_boundary_is_documented_not_a_bug() -> None:
    """A label carrying BOTH 'Mancha' (spot) and 'Blight' is inherently
    ambiguous; the mapper resolves it to leaf_spot (spot wins). Pinned so the
    behaviour is explicit and reviewable, not accidental."""
    assert to_disease_type("Milho (Corn) - Mancha_Turcicum (Northern Leaf Blight)") == "leaf_spot"
    # an unambiguous blight (no 'spot'/'mancha') still maps to blight
    assert to_disease_type("Corn leaf blight") == "blight"


# ------------------------------------------------------------------ #
# invariants
# ------------------------------------------------------------------ #


def test_every_output_is_a_canonical_type() -> None:
    everything = _classes("plantvillage") + _classes("paddy_doctor") + _classes("plantdoc")
    for cls in everything:
        assert to_disease_type(cls) in CANONICAL_TYPES


def test_coverage_of_real_classes_is_high() -> None:
    """Most real disease classes should map to a disease, not fall through."""
    everything = _classes("plantvillage") + _classes("paddy_doctor") + _classes("plantdoc")
    mapped = [to_disease_type(c) for c in everything]
    other = sum(1 for t in mapped if t == "other")
    assert other <= 2, f"too many unmapped: {other}"   # only Orange HLB is expected


def test_is_disease_excludes_pest_and_other() -> None:
    assert is_disease("rust") and is_disease("healthy")
    assert not is_disease("other")
    assert not is_disease("pest_damage")
