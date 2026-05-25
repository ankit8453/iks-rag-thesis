"""Smoke tests for the VIT Vellore latha-soil texture integration.

These tests verify the on-disk artefact produced by
``scripts/integrate_latha_soil.py``. They are intentionally non-strict:
if the script hasn't been run yet (clean checkout, no local data),
every test skips with a clear message rather than failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.utils.paths import CONFIGS_DIR, DATA_SOIL_DIR

VIT_ROOT = DATA_SOIL_DIR / "vit_texture"
VIT_RAW = VIT_ROOT / "raw"
AUDIT_PATH = VIT_ROOT / "INTEGRATION_AUDIT.json"
LABEL_MAP = CONFIGS_DIR / "data" / "soil_texture_label_mapping.yaml"

_REQUIRED_AUDIT_KEYS = (
    "source_url",
    "source_commit_sha",
    "integration_date",
    "class_counts",
    "canonical_classes",
)


def _class_subdirs() -> list[Path]:
    if not VIT_RAW.is_dir():
        return []
    return sorted(p for p in VIT_RAW.iterdir() if p.is_dir())


def _skip_unless_integrated() -> None:
    if not VIT_RAW.is_dir() or not _class_subdirs():
        pytest.skip(
            "VIT texture integration not run on this machine yet — "
            "execute scripts/integrate_latha_soil.py to materialise data."
        )


def test_raw_has_class_subfolders() -> None:
    _skip_unless_integrated()
    subdirs = _class_subdirs()
    assert len(subdirs) >= 1, f"Expected at least one class subdir under {VIT_RAW}"


def test_audit_json_is_valid_and_has_required_keys() -> None:
    if not AUDIT_PATH.is_file():
        pytest.skip(f"{AUDIT_PATH} missing — integration not run.")
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    for key in _REQUIRED_AUDIT_KEYS:
        assert key in payload, f"Audit JSON missing required key: {key!r}"
    assert isinstance(payload["class_counts"], dict)
    assert isinstance(payload["canonical_classes"], list)


def test_each_class_subdir_appears_in_label_mapping() -> None:
    _skip_unless_integrated()
    assert LABEL_MAP.is_file(), f"Label mapping config missing at {LABEL_MAP}"
    cfg = yaml.safe_load(LABEL_MAP.read_text(encoding="utf-8"))
    vit_section = cfg.get("vit_texture")
    assert isinstance(vit_section, dict), (
        "soil_texture_label_mapping.yaml must contain a 'vit_texture:' "
        "mapping block — was it removed?"
    )
    missing = [d.name for d in _class_subdirs() if d.name not in vit_section]
    assert not missing, (
        f"vit_texture section is missing entries for: {missing}. "
        f"Present keys: {sorted(vit_section)}"
    )
    coarse_fine_mixed = {"coarse", "fine", "mixed"}
    bad = {k: v for k, v in vit_section.items() if v not in coarse_fine_mixed}
    assert not bad, f"vit_texture values must be coarse/fine/mixed; got: {bad}"


def test_one_image_per_class_opens_with_pil() -> None:
    _skip_unless_integrated()
    pil = pytest.importorskip("PIL.Image")
    for class_dir in _class_subdirs():
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not images:
            pytest.fail(f"Class subdir {class_dir} is empty.")
        with pil.open(images[0]) as im:
            im.verify()
