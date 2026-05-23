"""Tests for src.disease.dataset (Phase 4 §G)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.disease.dataset import JSONIndexedImageDataset


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny on-disk dataset and a matching split JSON."""
    pil = pytest.importorskip("PIL.Image")
    raw_root = tmp_path / "raw"
    (raw_root / "a").mkdir(parents=True)
    (raw_root / "b").mkdir(parents=True)
    for label in ("a", "b"):
        for i in range(3):
            img = pil.new("RGB", (80, 80), color=(i * 30, 40, 60))
            img.save(raw_root / label / f"{i}.jpg", format="JPEG")

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train = [
        {"path": "a/0.jpg", "label": "a", "label_idx": 0},
        {"path": "a/1.jpg", "label": "a", "label_idx": 0},
        {"path": "b/0.jpg", "label": "b", "label_idx": 1},
    ]
    (splits_dir / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (splits_dir / "class_map.json").write_text(
        json.dumps({"a": 0, "b": 1}), encoding="utf-8"
    )
    return splits_dir / "train.json", raw_root


def test_dataset_loads_items_and_labels(tmp_path: Path) -> None:
    split, raw_root = _make_fixture(tmp_path)
    ds = JSONIndexedImageDataset(split, raw_root, transform=None)
    assert len(ds) == 3
    arr, label = ds[0]
    # No transform => numpy HWC uint8
    import numpy as np

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (80, 80, 3)
    assert label == 0


def test_dataset_applies_transform(tmp_path: Path) -> None:
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    from src.disease.transforms import build_disease_eval_aug

    split, raw_root = _make_fixture(tmp_path)
    aug = build_disease_eval_aug(64, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    ds = JSONIndexedImageDataset(split, raw_root, transform=aug)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert tuple(img.shape) == (3, 64, 64)
    assert label == 0


def test_class_map_round_trips(tmp_path: Path) -> None:
    split, raw_root = _make_fixture(tmp_path)
    ds = JSONIndexedImageDataset(split, raw_root)
    assert ds.class_map == {"a": 0, "b": 1}


def test_plantdoc_class_map_has_canonical_27_classes() -> None:
    """After the §A spider-mites merge, PlantDoc must have 27 classes
    (Singh et al. 2020 canonical count), not the upstream repo's 28.

    This is a real-data integration check — it relies on
    ``data/splits/plantdoc/class_map.json`` existing locally. If
    Phase 4 hasn't been run on this machine, the test skips.
    """
    plantdoc_class_map = (
        Path(__file__).resolve().parents[2]
        / "data" / "splits" / "plantdoc" / "class_map.json"
    )
    if not plantdoc_class_map.is_file():
        pytest.skip(f"No PlantDoc class map at {plantdoc_class_map}; skip integration check.")

    cm = json.loads(plantdoc_class_map.read_text(encoding="utf-8"))
    assert len(cm) == 27, f"Expected 27 PlantDoc classes after the merge, got {len(cm)}"
    # The vestigial folder must NOT appear as its own class.
    assert "Tomato two spotted spider mites leaf" not in cm
