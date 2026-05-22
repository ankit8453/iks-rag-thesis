"""Tests for src.integration.causation_dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integration.causation_dataset import MultiLabelImageDataset


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    pil = pytest.importorskip("PIL.Image")
    raw_root = tmp_path / "raw"
    classes = ["bottle_gourd__DM", "bottle_gourd__JAS", "bottle_gourd__healthy"]
    for cls in classes:
        (raw_root / cls).mkdir(parents=True)
        pil.new("RGB", (80, 80), color=(10, 10, 10)).save(
            raw_root / cls / "img.JPG", format="JPEG"
        )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    rows = [
        {"path": f"{cls}/img.JPG", "label": cls, "label_idx": i}
        for i, cls in enumerate(classes)
    ]
    (splits_dir / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    (splits_dir / "class_map.json").write_text(
        json.dumps({cls: i for i, cls in enumerate(classes)}), encoding="utf-8"
    )
    return splits_dir / "train.json", raw_root, classes


def test_returns_multi_hot_vector(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    split, raw_root, classes = _make_fixture(tmp_path)
    ds = MultiLabelImageDataset(split, raw_root, label_vocab=classes, transform=None)
    assert len(ds) == 3
    _, vec = ds[0]
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (len(classes),)
    # First entry's label is classes[0]
    assert vec[0].item() == 1.0
    assert vec[1].item() == 0.0
    assert vec[2].item() == 0.0


def test_class_map_round_trips_through_loader(tmp_path: Path) -> None:
    split, raw_root, classes = _make_fixture(tmp_path)
    ds = MultiLabelImageDataset(split, raw_root, label_vocab=classes, transform=None)
    assert ds.label_to_idx == {classes[0]: 0, classes[1]: 1, classes[2]: 2}
