"""Tests for the disease-type dataset builder's pure logic."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import importlib.util

# load the script as a module (it lives under scripts/, not a package)
_spec = importlib.util.spec_from_file_location(
    "build_disease_type_dataset",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "build_disease_type_dataset.py",
)
bdt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdt)


def test_deepest_disease_folder_wins_over_misleading_parent() -> None:
    root = Path("/data")
    # a Rust image nested under a mislabelled "...Leaf Scald...Cropped" parent
    p = (root / "Arroz (Rice) - Escaldadura (Leaf Scald) - Cropped"
              / "Cafe (Coffee) - Ferrugem (Rust) - 1" / "1" / "x.jpg")
    assert bdt.disease_type_from_path(p, root) == "rust"


def test_standard_single_level_and_healthy_leaf() -> None:
    root = Path("/data")
    assert bdt.disease_type_from_path(root / "Tomato___Late_blight" / "a.jpg", root) == "late_blight"
    assert bdt.disease_type_from_path(root / "Apple leaf" / "a.jpg", root) == "healthy"
    assert bdt.disease_type_from_path(root / "random_junk_folder" / "a.jpg", root) == "other"


def test_stratified_split_ratios_and_determinism() -> None:
    keys = [str(i) for i in range(100)]
    a = bdt.stratified_split(keys, (0.8, 0.1, 0.1), seed=42)
    c = Counter(a.values())
    assert c["train"] == 80 and c["val"] == 10 and c["test"] == 10
    # same seed => same assignment
    assert bdt.stratified_split(keys, (0.8, 0.1, 0.1), seed=42) == a
    # different seed => generally different
    assert bdt.stratified_split(keys, (0.8, 0.1, 0.1), seed=7) != a


def test_split_covers_every_key_exactly_once() -> None:
    keys = [f"k{i}" for i in range(37)]
    a = bdt.stratified_split(keys, (0.7, 0.15, 0.15), seed=1)
    assert set(a) == set(keys)
    assert all(v in {"train", "val", "test"} for v in a.values())
