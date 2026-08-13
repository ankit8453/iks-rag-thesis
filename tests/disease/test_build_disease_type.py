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


def test_crop_leaf_flag_invokes_the_cropper(tmp_path, monkeypatch) -> None:
    """With --crop-leaf, every image must pass through the YOLO LeafCropper."""
    from PIL import Image
    import src.disease.leaf_detect as ld

    # a source tree with two 'rust' images
    src = tmp_path / "src" / "Apple___Cedar_apple_rust"
    src.mkdir(parents=True)
    # distinct colours so content-hash dedup keeps BOTH images
    for n, col in (("a.jpg", (0, 128, 0)), ("b.jpg", (10, 90, 40))):
        Image.new("RGB", (40, 40), col).save(src / n)

    calls = {"n": 0}

    class _FakeCropper:                       # no YOLO / no network
        def __init__(self, *a, **k): ...
        def crop(self, img):
            calls["n"] += 1
            return img.crop((5, 5, 35, 35)), True   # pretend a leaf was found
    monkeypatch.setattr(ld, "LeafCropper", _FakeCropper)

    out = tmp_path / "out"
    summary = bdt.build({"t": tmp_path / "src"}, out, size=32,
                        min_per_class=1, crop_leaf=True)
    assert calls["n"] == 2                     # both images cropped
    assert summary["leaf_cropped"] == 2
    assert summary["total_images"] == 2

    # and NOT called when the flag is off
    calls["n"] = 0
    bdt.build({"t": tmp_path / "src"}, tmp_path / "out2", size=32,
              min_per_class=1, crop_leaf=False)
    assert calls["n"] == 0
