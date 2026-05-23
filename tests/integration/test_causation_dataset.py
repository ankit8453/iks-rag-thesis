"""Tests for src.integration.causation_dataset.

OLID I folder names are ``<crop>__<symptom>``. After the Phase-4-fix
:func:`_labels_for` update, each folder name expands to two multi-label
tags (the crop tag + the symptom tag). These tests verify that the
multi-hot vector reflects both halves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integration.causation_dataset import MultiLabelImageDataset, _labels_for


def _make_olid_like_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Three OLID-style class folders, one image each.

    Folder names mimic real OLID-I class folders (verified against the
    extracted Kaggle archive):
    - ``bottle_gourd__DM``  (crop=bottle_gourd, symptom=DM)
    - ``tomato__healthy``   (crop=tomato, symptom=healthy)
    - ``ash_gourd__PM``     (crop=ash_gourd, symptom=PM = powdery mildew)
    """
    pil = pytest.importorskip("PIL.Image")
    folder_names = [
        "bottle_gourd__DM",
        "tomato__healthy",
        "ash_gourd__PM",
    ]
    raw_root = tmp_path / "raw"
    for folder in folder_names:
        (raw_root / folder).mkdir(parents=True)
        pil.new("RGB", (80, 80), color=(10, 20, 30)).save(
            raw_root / folder / "img.JPG", format="JPEG"
        )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    rows = [
        {"path": f"{folder}/img.JPG", "label": folder, "label_idx": 0}
        for folder in folder_names
    ]
    (splits_dir / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    return splits_dir / "train.json", raw_root, folder_names


def test_labels_for_simple_crop_plus_symptom() -> None:
    assert _labels_for("bottle_gourd__DM") == ["bottle_gourd", "DM"]
    assert _labels_for("tomato__healthy") == ["tomato", "healthy"]
    # Non-OLID-style names fall back to single-label.
    assert _labels_for("Alluvial_Soil") == ["Alluvial_Soil"]


def test_labels_for_compound_symptoms_split_on_underscore() -> None:
    """Compound symptoms like JAS_MIT or N_K must produce individual tags."""
    # Pest co-occurrence
    assert _labels_for("bottle_gourd__JAS_MIT") == ["bottle_gourd", "JAS", "MIT"]
    # Nutrient deficiency co-occurrence
    assert _labels_for("ash_gourd__N_K") == ["ash_gourd", "N", "K"]
    assert _labels_for("bitter_gourd__K_Mg") == ["bitter_gourd", "K", "Mg"]
    assert _labels_for("ash_gourd__N_Mg") == ["ash_gourd", "N", "Mg"]


def test_multi_hot_expands_crop_plus_symptom(tmp_path: Path) -> None:
    """Each OLID-style entry must light up TWO bits in the multi-hot vector."""
    torch = pytest.importorskip("torch")
    split, raw_root, folder_names = _make_olid_like_fixture(tmp_path)

    # Build the multi-label vocab the way the real OLID artifact builder
    # does: union of crop tags + symptom tags.
    vocab = sorted(
        {tag for folder in folder_names for tag in _labels_for(folder)}
    )
    ds = MultiLabelImageDataset(split, raw_root, label_vocab=vocab, transform=None)

    for idx, folder in enumerate(folder_names):
        _, vec = ds[idx]
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (len(vocab),)
        crop, symptom = folder.split("__")
        assert vec[vocab.index(crop)].item() == 1.0, f"crop bit missing for {folder}"
        assert vec[vocab.index(symptom)].item() == 1.0, f"symptom bit missing for {folder}"
        # Every other bit is 0.
        assert vec.sum().item() == 2.0, f"expected exactly 2 hot bits in {folder}"


def test_unknown_label_is_dropped_silently(tmp_path: Path) -> None:
    """Tags outside ``label_vocab`` produce a zeroed slot, not an error."""
    pil = pytest.importorskip("PIL.Image")
    raw_root = tmp_path / "raw"
    (raw_root / "tomato__unknown_symptom").mkdir(parents=True)
    pil.new("RGB", (80, 80), color=(10, 20, 30)).save(
        raw_root / "tomato__unknown_symptom" / "img.JPG", format="JPEG"
    )
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    rows = [
        {
            "path": "tomato__unknown_symptom/img.JPG",
            "label": "tomato__unknown_symptom",
            "label_idx": 0,
        }
    ]
    (splits_dir / "train.json").write_text(json.dumps(rows), encoding="utf-8")

    # Vocab knows ``tomato`` but not ``unknown_symptom``.
    vocab = ["tomato", "healthy"]
    ds = MultiLabelImageDataset(splits_dir / "train.json", raw_root, label_vocab=vocab)
    _, vec = ds[0]
    assert vec[0].item() == 1.0  # tomato
    assert vec[1].item() == 0.0  # healthy NOT present
    assert vec.sum().item() == 1.0
