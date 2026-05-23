"""Tests for the §14 soil cross-region split."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.data_splits import make_soil_cross_region_split


def _phantomfs() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for label in ("alluvial", "black", "clay", "red", "laterite"):
        for i in range(40):
            items.append((Path(f"phantomfs/{label}/{i}.jpg"), label))
    return items


def _irsid() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for label in ("sand", "clay", "sandy_loam", "loam", "loamy_sand"):
        for i in range(20):
            items.append((Path(f"irsid/{label}/{i}.jpg"), label))
    return items


def test_test_set_contains_only_irsid_paths() -> None:
    split = make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.1, seed=42)
    for path, _ in split["test"]:
        assert path.parts[0] == "irsid"
        assert path.parts[0] != "phantomfs"


def test_train_and_val_contain_only_phantomfs() -> None:
    split = make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.1, seed=42)
    for path, _ in split["train"] + split["val"]:
        assert path.parts[0] == "phantomfs"


def test_no_phantomfs_path_leaks_into_test() -> None:
    split = make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.1, seed=42)
    train_paths = {str(p) for p, _ in split["train"]}
    val_paths = {str(p) for p, _ in split["val"]}
    test_paths = {str(p) for p, _ in split["test"]}
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert train_paths.isdisjoint(val_paths)


def test_phantomfs_classes_present_in_both_train_and_val() -> None:
    """Stratified split must keep every Phantom-fs class represented."""
    split = make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.1, seed=42)
    train_labels = {lab for _, lab in split["train"]}
    val_labels = {lab for _, lab in split["val"]}
    assert train_labels == val_labels


def test_val_fraction_approximately_correct() -> None:
    split = make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.1, seed=42)
    n_phantomfs = len(split["train"]) + len(split["val"])
    val_ratio = len(split["val"]) / n_phantomfs
    assert abs(val_ratio - 0.1) < 0.02


def test_test_size_equals_all_irsid() -> None:
    irsid = _irsid()
    split = make_soil_cross_region_split(_phantomfs(), irsid, val_frac=0.1, seed=42)
    assert len(split["test"]) == len(irsid)


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError):
        make_soil_cross_region_split([], _irsid(), val_frac=0.1, seed=42)
    with pytest.raises(ValueError):
        make_soil_cross_region_split(_phantomfs(), [], val_frac=0.1, seed=42)
    with pytest.raises(ValueError):
        make_soil_cross_region_split(_phantomfs(), _irsid(), val_frac=0.0, seed=42)
