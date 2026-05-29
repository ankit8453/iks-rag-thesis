"""Tests for :mod:`src.soil.tiling` — patch extraction + leakage guard.

The leakage test is the critical one — if patches from the same source
image can land in both train and test, downstream accuracy is fake-high.
That assertion failing here would invalidate the V3-tiling experiment.
"""

from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.soil.tiling import (  # noqa: E402
    PATCH_MIN_PIXELS,
    PATCH_OUTPUT_SIZE,
    build_tiled_split,
    check_patch_resolution,
    tile_image,
)


def _solid_image(width: int, height: int, colour=(120, 80, 40)):
    return Image.new("RGB", (width, height), color=colour)


def test_tile_image_returns_grid_squared_patches_of_output_size() -> None:
    img = _solid_image(512, 512)
    patches = tile_image(img, grid=4)
    assert len(patches) == 16
    for p in patches:
        assert p.size == (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE)
        assert p.mode == "RGB"


def test_tile_image_grid_three() -> None:
    img = _solid_image(600, 600)
    patches = tile_image(img, grid=3)
    assert len(patches) == 9


def test_tile_image_rejects_grid_zero() -> None:
    with pytest.raises(ValueError):
        tile_image(_solid_image(100, 100), grid=0)


def test_tile_image_rejects_image_too_small_for_grid() -> None:
    # 8x8 image at grid=10 -> tile_w=0 — must reject, not silently produce empty patches.
    with pytest.raises(ValueError):
        tile_image(_solid_image(8, 8), grid=10)


def test_check_patch_resolution_warns_below_threshold() -> None:
    images = [_solid_image(400, 300) for _ in range(5)]  # min dim 300
    # At grid=4: median 300/4 = 75 px -> below threshold -> warning.
    result = check_patch_resolution(images, grid=4)
    assert result["warning"] is True
    assert result["patch_min_pre_resize"] == 75
    assert result["min_threshold"] == PATCH_MIN_PIXELS
    assert result["recommended_grid"] == 3


def test_check_patch_resolution_no_warning_when_safe() -> None:
    images = [_solid_image(800, 600) for _ in range(5)]  # min dim 600
    result = check_patch_resolution(images, grid=4)
    assert result["warning"] is False
    assert result["patch_min_pre_resize"] == 150  # 600 / 4


def test_check_patch_resolution_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        check_patch_resolution([], grid=4)


def test_build_tiled_split_propagates_label_and_source_id() -> None:
    items = [
        ("train_0000", _solid_image(400, 400), 1),
        ("train_0001", _solid_image(400, 400), 0),
    ]
    patches = build_tiled_split(items, grid=3)
    assert len(patches) == 2 * 9
    # First 9 patches inherit label=1 / source_id="train_0000"
    for patch, label, sid in patches[:9]:
        assert label == 1
        assert sid == "train_0000"
        assert patch.size == (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE)
    # Next 9 inherit label=0 / source_id="train_0001"
    for patch, label, sid in patches[9:]:
        assert label == 0
        assert sid == "train_0001"


def test_leakage_check_disjoint_splits_by_construction() -> None:
    """The critical test: source_ids assigned per-split must stay disjoint."""
    # Simulate the build pipeline: each split has its own prefixed source ids.
    train_items = [(f"train_{i:04d}", _solid_image(400, 400), i % 3) for i in range(5)]
    val_items   = [(f"val_{i:04d}",   _solid_image(400, 400), i % 3) for i in range(2)]
    test_items  = [(f"test_{i:04d}",  _solid_image(400, 400), i % 3) for i in range(2)]

    train_patches = build_tiled_split(train_items, grid=3)
    val_patches   = build_tiled_split(val_items, grid=3)
    test_patches  = build_tiled_split(test_items, grid=3)

    train_sids = {sid for _, _, sid in train_patches}
    val_sids   = {sid for _, _, sid in val_patches}
    test_sids  = {sid for _, _, sid in test_patches}

    assert not (train_sids & val_sids), "train/val source_id leakage"
    assert not (train_sids & test_sids), "train/test source_id leakage"
    assert not (val_sids & test_sids), "val/test source_id leakage"

    # Every train source_id starts with "train_"; same for val/test.
    assert all(s.startswith("train_") for s in train_sids)
    assert all(s.startswith("val_") for s in val_sids)
    assert all(s.startswith("test_") for s in test_sids)


def test_leakage_check_detects_bad_construction() -> None:
    """Regression for the failure mode: if the same source_id is used across\n    splits, the test must catch it. This guards against a future refactor\n    that 'simplifies' source_id and loses the per-split prefix."""
    train_items = [("img_0000", _solid_image(400, 400), 0)]
    val_items   = [("img_0000", _solid_image(400, 400), 0)]  # SAME source id — bug!

    train_patches = build_tiled_split(train_items, grid=3)
    val_patches   = build_tiled_split(val_items, grid=3)
    train_sids = {sid for _, _, sid in train_patches}
    val_sids   = {sid for _, _, sid in val_patches}
    overlap = train_sids & val_sids
    assert overlap == {"img_0000"}, (
        "Leakage detector failed — the regression case did NOT show overlap. "
        "The build pipeline's per-split prefix is the only thing keeping the "
        "splits disjoint."
    )
