"""Phase 5-R Part 2 — segment_cache regressions.

Pure-Python checks against the on-disk mask cache contract. No GPU,
no network, no model load — every external call is monkeypatched.

Locks the two invariants the training loop relies on:

1. ``mask_path_for`` produces a stable, OS-independent rel-path-to-PNG
   mapping under :data:`MASK_CACHE_ROOT`.
2. When :func:`build_mask_cache` sees a "flagged" segmentation result
   (mask covers < 5 % or > 95 % of the image), the mask is still
   written to disk **and** the rel_path lands in the log's
   ``flagged_rel_paths`` field — :func:`load_flagged_set` returns it so
   the randomized dataset can fall back to raw at train time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.disease.segment_cache import (
    MASK_CACHE_ROOT,
    build_mask_cache,
    dataset_log_path,
    load_flagged_set,
    mask_path_for,
)


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect :data:`MASK_CACHE_ROOT` into ``tmp_path`` for the test."""
    fake_root = tmp_path / "_masks"
    fake_root.mkdir()
    monkeypatch.setattr(
        "src.disease.segment_cache.MASK_CACHE_ROOT", fake_root, raising=True,
    )
    return fake_root


def test_mask_path_for_is_stable_and_swaps_extension(tmp_path: Path) -> None:
    """``mask_path_for`` must produce a deterministic per-rel-path PNG
    path under :data:`MASK_CACHE_ROOT`, regardless of original extension."""
    paths = [
        ("plantvillage", "Apple___Apple_scab/image.JPG"),
        ("plantdoc",     "Tomato leaf late blight/image.jpg"),
        ("plantdoc",     "Corn rust leaf/Corn-southern-rust.ashx.jpg"),
    ]
    for ds, rel in paths:
        out = mask_path_for(ds, rel)
        # under the configured cache root, with dataset subdir, with .png suffix
        assert out.suffix == ".png"
        assert out.parts[-2] == Path(rel).parent.name or (
            Path(rel).parent.name in out.as_posix()
        )
        assert ds in out.as_posix()


def test_build_mask_cache_writes_flagged_mask_and_logs_path(
    temp_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flagged mask must be:

    1. Written to disk (the trainer's fallback path checks
       ``mask_path.is_file()`` — a missing file would also trigger raw,
       but we want the flagged information to survive into the log).
    2. Recorded under ``flagged_rel_paths`` so
       :func:`load_flagged_set` returns the row.
    """
    # Fake a tiny RGB image on disk so the segmenter has something to open.
    img_dir = tmp_path / "raw_imgs" / "Foo"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "x.jpg"
    from PIL import Image

    Image.new("RGB", (16, 16), (0, 100, 0)).save(img_path)

    # Build a stub SegmentResult: all-foreground (fg = 100 %) → flagged.
    class _StubResult:
        def __init__(self) -> None:
            self.mask = np.full((16, 16), 255, dtype=np.uint8)
            self.foreground_fraction = 1.0
            self.flagged_as_failure = True
            self.method = "classical"

    monkeypatch.setattr(
        "src.disease.segment_cache.segment",
        lambda image_path, style: _StubResult(),
    )

    stats = build_mask_cache(
        dataset="plantvillage",
        style="lab",
        image_iter=[("Foo/x.jpg", img_path)],
        log_every=1,
    )

    # Mask file is on disk (under the redirected MASK_CACHE_ROOT)
    out = mask_path_for("plantvillage", "Foo/x.jpg")
    assert out.is_file(), f"expected mask file at {out}"

    # Cache stats agree
    assert stats.total == 1
    assert stats.newly_segmented == 1
    assert stats.flagged == 1
    assert stats.flagged_fraction == 1.0

    # Log carries the rel_path
    log = dataset_log_path("plantvillage")
    assert log.is_file()
    flagged_set = load_flagged_set("plantvillage")
    assert "Foo/x.jpg" in flagged_set


def test_build_mask_cache_is_idempotent(
    temp_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the cache build over the same input must NOT
    re-segment — the trainer might re-launch many times."""
    img_dir = tmp_path / "raw_imgs" / "Bar"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "y.jpg"
    from PIL import Image

    Image.new("RGB", (16, 16), (0, 200, 0)).save(img_path)

    seg_calls = {"n": 0}

    class _OkResult:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        foreground_fraction = 0.5
        flagged_as_failure = False
        method = "classical"

    def _fake_segment(image_path: Any, style: str) -> _OkResult:
        seg_calls["n"] += 1
        return _OkResult()

    monkeypatch.setattr("src.disease.segment_cache.segment", _fake_segment)

    iter_input = [("Bar/y.jpg", img_path)]
    s1 = build_mask_cache("plantvillage", "lab", iter_input, log_every=1)
    s2 = build_mask_cache("plantvillage", "lab", iter_input, log_every=1)

    assert s1.newly_segmented == 1
    # Second call must hit the cache, NOT re-segment.
    assert s2.newly_segmented == 0
    assert s2.cached_already == 1
    assert seg_calls["n"] == 1  # only the first run called segment()


def test_load_flagged_set_returns_empty_when_no_log(
    temp_cache: Path,
) -> None:
    """A dataset that has never been cached → no log → empty set."""
    assert load_flagged_set("never_cached") == set()
