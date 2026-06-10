"""Phase 5-R Part 2 — randomized-dataset wrapper regressions.

Stub images / stub mask / stub background pool — no GPU, no network,
no model load. Locks the four invariants the trainer relies on:

1. ``mode="raw"`` returns the raw image untouched (Paddy Doctor path).
2. ``mode="no_leaf"`` carries the reject-class label downstream.
3. ``mode="randomize"`` actually composites — output differs from the
   raw input — and the same epoch+index pair re-produces the same
   composite (reproducible per-epoch seeding).
4. A flagged rel-path falls back to raw even when a mask file exists on
   disk; the trainer must not crash, and the composite step must NOT
   touch the background pool for that row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.disease.backgrounds import BackgroundEntry
from src.disease.randomized_dataset import (
    RandomizedDiseaseDataset,
    SampleSpec,
    build_no_leaf_samples,
    load_class_map_with_no_leaf,
)


def _save_solid(path: Path, colour: tuple[int, int, int], size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path)


def _save_mask(path: Path, fg_fraction: float = 0.5, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size), dtype=np.uint8)
    half = int(size * fg_fraction)
    arr[:half, :] = 255  # top stripe = foreground
    Image.fromarray(arr, mode="L").save(path)


def test_raw_mode_returns_image_unchanged(tmp_path: Path) -> None:
    img = tmp_path / "raw.jpg"
    _save_solid(img, (10, 20, 30))
    sample = SampleSpec(
        rel_path="raw.jpg", abs_path=img, label_idx=3, mode="raw",
    )
    ds = RandomizedDiseaseDataset(
        samples=[sample], dataset_id="paddy_doctor", bg_pool=[],
        transform=None, seed=42, flagged_rel_paths=set(),
    )
    arr, label = ds[0]
    assert label == 3
    assert arr.shape == (32, 32, 3)
    # Top-left pixel is (R=10, G=20, B=30) — RGB order from PIL convert.
    assert tuple(arr[0, 0]) == (10, 20, 30)


def test_no_leaf_mode_carries_reject_label(tmp_path: Path) -> None:
    img = tmp_path / "bg.jpg"
    _save_solid(img, (40, 40, 40))
    sample = SampleSpec(
        rel_path="bg.jpg", abs_path=img, label_idx=27, mode="no_leaf",
    )
    ds = RandomizedDiseaseDataset(
        samples=[sample], dataset_id="plantdoc", bg_pool=[],
        transform=None, seed=42, flagged_rel_paths=set(),
    )
    arr, label = ds[0]
    assert label == 27, "no_leaf row must keep its reject label."
    assert arr.shape == (32, 32, 3)


def test_randomize_mode_composites_when_mask_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached mask exists → composite path runs and the output differs
    from the raw input (since the background is a distinct colour)."""
    img = tmp_path / "raws" / "Cls" / "leaf.jpg"
    _save_solid(img, (10, 200, 10))      # green leaf
    bg = tmp_path / "bg" / "soil.jpg"
    _save_solid(bg, (200, 30, 30))       # red soil

    # Redirect MASK_CACHE_ROOT so make_randomized_dataset reads from tmp.
    mask_root = tmp_path / "_masks"
    mask_root.mkdir()
    monkeypatch.setattr(
        "src.disease.segment_cache.MASK_CACHE_ROOT", mask_root, raising=True,
    )
    # And monkeypatch in randomized_dataset's import too — mask_path_for
    # is re-exported by the helper module.
    monkeypatch.setattr(
        "src.disease.randomized_dataset.mask_path_for",
        lambda dataset, rel: mask_root / dataset / (Path(rel).with_suffix(".png")),
        raising=True,
    )
    _save_mask(mask_root / "plantvillage" / "Cls" / "leaf.png", fg_fraction=0.5)

    sample = SampleSpec(
        rel_path="Cls/leaf.jpg", abs_path=img,
        label_idx=5, mode="randomize",
    )
    bg_entry = BackgroundEntry(path=bg, source="test", rel_path="soil.jpg")
    ds = RandomizedDiseaseDataset(
        samples=[sample], dataset_id="plantvillage", bg_pool=[bg_entry],
        transform=None, seed=42, flagged_rel_paths=set(),
    )

    arr, label = ds[0]
    assert label == 5
    assert arr.shape == (32, 32, 3)
    # The composite must contain SOME red (background) AND SOME green (leaf).
    has_red = np.any(arr[:, :, 0] > 100)
    has_green = np.any(arr[:, :, 1] > 100)
    assert has_red and has_green, (
        "randomize composite did not blend bg and leaf — got "
        f"R={arr[:,:,0].max()}, G={arr[:,:,1].max()}, B={arr[:,:,2].max()}"
    )


def test_randomize_mode_falls_back_to_raw_when_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row in ``flagged_rel_paths`` must NOT composite, even if a mask
    file exists. The trainer relies on this so a bad mask (full image)
    doesn't paste leaf-coloured noise everywhere."""
    img = tmp_path / "raws" / "Cls" / "bad.jpg"
    _save_solid(img, (5, 5, 5))
    bg = tmp_path / "bg" / "loud.jpg"
    _save_solid(bg, (255, 255, 255))

    mask_root = tmp_path / "_masks"
    mask_root.mkdir()
    monkeypatch.setattr(
        "src.disease.randomized_dataset.mask_path_for",
        lambda dataset, rel: mask_root / dataset / (Path(rel).with_suffix(".png")),
        raising=True,
    )
    _save_mask(mask_root / "plantvillage" / "Cls" / "bad.png", fg_fraction=0.99)

    sample = SampleSpec(
        rel_path="Cls/bad.jpg", abs_path=img, label_idx=2, mode="randomize",
    )
    bg_entry = BackgroundEntry(path=bg, source="test", rel_path="loud.jpg")
    ds = RandomizedDiseaseDataset(
        samples=[sample], dataset_id="plantvillage", bg_pool=[bg_entry],
        transform=None, seed=42,
        flagged_rel_paths={"Cls/bad.jpg"},   # flagged → raw fallback
    )
    arr, _ = ds[0]
    # The bg is pure white. If randomize had run, the canvas would be
    # mostly white. We want it mostly the original dark grey (5,5,5).
    assert arr[:, :, 0].mean() < 50, (
        "flagged row composited a bright background instead of falling back to raw"
    )


def test_randomize_is_reproducible_for_same_epoch_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same epoch + same index → same composite (per-epoch seeding)."""
    img = tmp_path / "raws" / "Cls" / "rep.jpg"
    _save_solid(img, (0, 180, 0))
    bg_a = tmp_path / "bg" / "a.jpg"
    _save_solid(bg_a, (200, 0, 0))
    bg_b = tmp_path / "bg" / "b.jpg"
    _save_solid(bg_b, (0, 0, 200))

    mask_root = tmp_path / "_masks"
    mask_root.mkdir()
    monkeypatch.setattr(
        "src.disease.randomized_dataset.mask_path_for",
        lambda dataset, rel: mask_root / dataset / (Path(rel).with_suffix(".png")),
        raising=True,
    )
    _save_mask(mask_root / "plantvillage" / "Cls" / "rep.png", fg_fraction=0.5)

    sample = SampleSpec(
        rel_path="Cls/rep.jpg", abs_path=img, label_idx=0, mode="randomize",
    )
    pool = [
        BackgroundEntry(path=bg_a, source="t", rel_path="a.jpg"),
        BackgroundEntry(path=bg_b, source="t", rel_path="b.jpg"),
    ]
    ds = RandomizedDiseaseDataset(
        samples=[sample], dataset_id="plantvillage", bg_pool=pool,
        transform=None, seed=42, flagged_rel_paths=set(),
    )
    ds.set_epoch(0)
    arr1, _ = ds[0]
    ds.set_epoch(0)
    arr2, _ = ds[0]
    assert np.array_equal(arr1, arr2), (
        "Same epoch+index must produce identical composites — needed for "
        "reproducible Grad-CAM comparisons."
    )

    # Different epoch → almost certainly a different composite (RNG diverges)
    ds.set_epoch(1)
    arr3, _ = ds[0]
    assert not np.array_equal(arr1, arr3), (
        "Different epochs returned identical composites — per-epoch "
        "reseed is broken."
    )


def test_build_no_leaf_samples_returns_reject_label(tmp_path: Path) -> None:
    src = tmp_path / "no_leaf_src"
    src.mkdir()
    for i in range(3):
        _save_solid(src / f"img_{i}.jpg", (i * 20, i * 20, i * 20))
    samples = build_no_leaf_samples([src], label_idx=27)
    assert len(samples) == 3
    for s in samples:
        assert s.label_idx == 27
        assert s.mode == "no_leaf"


def test_load_class_map_with_no_leaf_appends_index(tmp_path: Path) -> None:
    """A 27-class map gains a 28th ``no_leaf`` slot at idx 27 — existing
    ids must NOT shift."""
    cm_path = tmp_path / "class_map.json"
    base = {f"class_{i}": i for i in range(27)}
    import json

    cm_path.write_text(json.dumps(base), encoding="utf-8")
    extended, idx = load_class_map_with_no_leaf(cm_path)
    assert idx == 27
    assert extended["no_leaf"] == 27
    for i in range(27):
        assert extended[f"class_{i}"] == i   # unchanged
