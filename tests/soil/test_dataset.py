"""Tests for src.soil.dataset (Phase 4 §G)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.soil.dataset import _PrefixedCrossRegionDataset


def test_cross_region_resolves_prefixed_paths(tmp_path: Path) -> None:
    """Paths like ``phantomfs/foo.jpg`` resolve to ``<root>/phantomfs/raw/foo.jpg``."""
    pil = pytest.importorskip("PIL.Image")
    soil_root = tmp_path / "soil"
    (soil_root / "phantomfs" / "raw").mkdir(parents=True)
    pil.new("RGB", (64, 64), color=(10, 20, 30)).save(
        soil_root / "phantomfs" / "raw" / "img.jpg", format="JPEG"
    )

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(
        json.dumps(
            [{"path": "phantomfs/img.jpg", "label": "Black", "label_idx": 0}]
        ),
        encoding="utf-8",
    )

    ds = _PrefixedCrossRegionDataset(split_dir / "train.json", soil_root, transform=None)
    arr, label = ds[0]
    import numpy as np

    assert isinstance(arr, np.ndarray)
    assert label == 0
