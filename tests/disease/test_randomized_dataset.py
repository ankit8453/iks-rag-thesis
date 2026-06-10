"""Phase 5-R Part 2 — HFRandomizedDiseaseDataset regressions.

Stub HF split + stub mask cache + stub bg pool — no GPU, no network,
no model load. Locks the five invariants the trainer relies on:

1. ``mode="raw"`` returns the HF image unchanged (Paddy path).
2. ``mode="no_leaf"`` rows append AFTER the HF rows; their label is
   the reject index, regardless of the HF row's own ``label_idx``.
3. ``mode="randomize"`` composites when a cached mask exists at the
   expected ``mask_path_for(dataset_id, split, idx)`` location.
4. A row whose key is in the flagged set falls back to raw even when a
   mask file exists — flagged ⇒ no surprise composite.
5. Per-epoch seeded RNG: ``(epoch, idx)`` reproduces the same pixel
   array; different epoch ⇒ different array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.disease.backgrounds import BackgroundEntry
from src.disease.randomized_dataset import (
    HFRandomizedDiseaseDataset,
    NoLeafRow,
    build_no_leaf_rows,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


class _FakeHFSplit:
    """Stand-in for an HF Dataset split."""

    def __init__(self, images: list, label_idxs: list[int]) -> None:
        self.images = images
        self.label_idxs = label_idxs

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"image": self.images[idx], "label_idx": self.label_idxs[idx]}


def _solid(colour: tuple[int, int, int], size: int = 32):
    return Image.new("RGB", (size, size), colour)


def _save_mask(path: Path, fg_fraction: float = 0.5, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size), dtype=np.uint8)
    half = int(size * fg_fraction)
    arr[:half, :] = 255   # top stripe = foreground
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture()
def patched_mask_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect mask_path_for + load_flagged_set / is_flagged to tmp_path."""
    mask_root = tmp_path / "_masks"
    mask_root.mkdir()

    def fake_mask_path(dataset, split, row_idx):
        return mask_root / dataset / split / f"{int(row_idx):06d}.png"

    # randomized_dataset imports these symbols by name, so patch them
    # in that module's namespace.
    monkeypatch.setattr(
        "src.disease.randomized_dataset.mask_path_for",
        fake_mask_path,
        raising=True,
    )

    # Default: no flagged set; per-test can override.
    monkeypatch.setattr(
        "src.disease.segment_cache.load_flagged_set",
        lambda dataset: set(),
        raising=True,
    )
    return mask_root


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_raw_mode_returns_hf_image_unchanged(
    patched_mask_cache: Path,
) -> None:
    img = _solid((10, 20, 30))
    split = _FakeHFSplit(images=[img], label_idxs=[5])
    ds = HFRandomizedDiseaseDataset(
        hf_split=split, dataset_id="paddy_doctor", split="train",
        mode="raw", bg_pool=[], transform=None, no_leaf_rows=[],
    )
    arr, label = ds[0]
    assert label == 5
    assert arr.shape == (32, 32, 3)
    assert tuple(arr[0, 0]) == (10, 20, 30)


def test_no_leaf_rows_append_and_carry_reject_label(
    patched_mask_cache: Path, tmp_path: Path,
) -> None:
    """no_leaf rows come AFTER the HF rows and use the reject label."""
    img = _solid((50, 50, 50))
    split = _FakeHFSplit(images=[img], label_idxs=[3])
    nl_path = tmp_path / "nl.jpg"
    _solid((100, 100, 100)).save(nl_path)
    no_leaf_rows = [NoLeafRow(abs_path=nl_path, label_idx=27)]

    ds = HFRandomizedDiseaseDataset(
        hf_split=split, dataset_id="plantdoc", split="train",
        mode="raw", bg_pool=[], transform=None,
        no_leaf_rows=no_leaf_rows,
    )
    assert len(ds) == 2
    _arr_hf, label_hf = ds[0]
    assert label_hf == 3
    _arr_nl, label_nl = ds[1]
    assert label_nl == 27, (
        "no_leaf row must carry the reject label even though it lives "
        "in the same dataset object."
    )


def test_randomize_mode_composites_when_mask_present(
    patched_mask_cache: Path, tmp_path: Path,
) -> None:
    """Cached mask exists → composite path runs and the output blends
    leaf colour with background colour."""
    leaf = _solid((10, 200, 10))         # green leaf
    split = _FakeHFSplit(images=[leaf], label_idxs=[2])
    _save_mask(
        patched_mask_cache / "plantvillage" / "train" / "000000.png",
        fg_fraction=0.5,
    )
    bg_path = tmp_path / "soil.jpg"
    _solid((200, 30, 30)).save(bg_path)   # red soil

    ds = HFRandomizedDiseaseDataset(
        hf_split=split, dataset_id="plantvillage", split="train",
        mode="randomize",
        bg_pool=[BackgroundEntry(path=bg_path, source="t", rel_path="soil.jpg")],
        transform=None, no_leaf_rows=[],
    )
    arr, label = ds[0]
    assert label == 2
    has_red = np.any(arr[:, :, 0] > 100)
    has_green = np.any(arr[:, :, 1] > 100)
    assert has_red and has_green, (
        f"randomize composite did not blend bg + leaf: "
        f"R_max={arr[:,:,0].max()} G_max={arr[:,:,1].max()}"
    )


def test_randomize_falls_back_to_raw_when_flagged(
    patched_mask_cache: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose key is in the flagged set must NOT composite even if
    a mask file exists on disk."""
    leaf = _solid((5, 5, 5))            # dark grey
    split = _FakeHFSplit(images=[leaf], label_idxs=[0])
    _save_mask(
        patched_mask_cache / "plantvillage" / "train" / "000000.png",
        fg_fraction=0.99,
    )
    bg_path = tmp_path / "loud.jpg"
    _solid((255, 255, 255)).save(bg_path)

    # Inject the flagged set BEFORE constructing the dataset (the wrapper
    # caches it at __init__ time).
    monkeypatch.setattr(
        "src.disease.segment_cache.load_flagged_set",
        lambda dataset: {"train/000000"},
        raising=True,
    )

    ds = HFRandomizedDiseaseDataset(
        hf_split=split, dataset_id="plantvillage", split="train",
        mode="randomize",
        bg_pool=[BackgroundEntry(path=bg_path, source="t", rel_path="loud.jpg")],
        transform=None, no_leaf_rows=[],
    )
    arr, _ = ds[0]
    # If the composite had run, the bright-white bg would push the
    # mean above 50. The raw leaf is (5,5,5) → mean well below 50.
    assert arr[:, :, 0].mean() < 50, (
        "flagged row composited a bright bg instead of falling back to raw."
    )


def test_per_epoch_seeded_rng_is_reproducible(
    patched_mask_cache: Path, tmp_path: Path,
) -> None:
    leaf = _solid((0, 180, 0))
    split = _FakeHFSplit(images=[leaf], label_idxs=[0])
    _save_mask(
        patched_mask_cache / "plantvillage" / "train" / "000000.png",
        fg_fraction=0.5,
    )
    bg_a = tmp_path / "a.jpg"; _solid((200, 0, 0)).save(bg_a)
    bg_b = tmp_path / "b.jpg"; _solid((0, 0, 200)).save(bg_b)

    pool = [
        BackgroundEntry(path=bg_a, source="t", rel_path="a.jpg"),
        BackgroundEntry(path=bg_b, source="t", rel_path="b.jpg"),
    ]
    ds = HFRandomizedDiseaseDataset(
        hf_split=split, dataset_id="plantvillage", split="train",
        mode="randomize", bg_pool=pool, transform=None, no_leaf_rows=[],
    )
    ds.set_epoch(0); arr1, _ = ds[0]
    ds.set_epoch(0); arr2, _ = ds[0]
    assert np.array_equal(arr1, arr2), (
        "Same (epoch, idx) must produce identical composites."
    )
    ds.set_epoch(1); arr3, _ = ds[0]
    assert not np.array_equal(arr1, arr3), (
        "Different epochs returned identical composites — per-epoch "
        "reseed is broken."
    )


def test_build_no_leaf_rows_filters_by_source(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"; _solid((1, 2, 3)).save(a)
    b = tmp_path / "b.jpg"; _solid((4, 5, 6)).save(b)
    pool = [
        BackgroundEntry(path=a, source="phantomfs", rel_path="a.jpg"),
        BackgroundEntry(path=b, source="pandey_background", rel_path="b.jpg"),
    ]
    rows = build_no_leaf_rows(pool, label_idx=27, sources=("pandey_background",))
    assert len(rows) == 1
    assert rows[0].label_idx == 27
    assert rows[0].abs_path == b
