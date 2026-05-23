"""Tests for src.utils.data_splits — stratified splits and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.data_splits import (
    SplitEntry,
    build_class_map,
    discover_class_folder_items,
    load_class_map,
    load_split,
    save_split,
    stratified_split,
)


def _make_items(per_class: dict[str, int]) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for label, n in per_class.items():
        for i in range(n):
            items.append((Path(f"{label}/{i}.jpg"), label))
    return items


def test_ratios_are_correct_to_within_one_percent() -> None:
    items = _make_items({"a": 200, "b": 200, "c": 200})  # total 600
    split = stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=42)
    total = sum(len(v) for v in split.values())
    assert total == 600
    assert abs(len(split["train"]) / total - 0.8) < 0.01
    assert abs(len(split["val"]) / total - 0.1) < 0.01
    assert abs(len(split["test"]) / total - 0.1) < 0.01


def test_stratification_preserves_class_distribution() -> None:
    items = _make_items({"a": 100, "b": 200, "c": 300})  # 1:2:3 ratio
    split = stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=42)
    for split_items in split.values():
        labels = [lab for _, lab in split_items]
        n = len(labels)
        ratio_a = labels.count("a") / n
        ratio_b = labels.count("b") / n
        ratio_c = labels.count("c") / n
        # Expected 1/6, 2/6, 3/6. Allow generous slack on tiny splits.
        assert abs(ratio_a - 1 / 6) < 0.05
        assert abs(ratio_b - 2 / 6) < 0.05
        assert abs(ratio_c - 3 / 6) < 0.05


def test_same_seed_yields_identical_splits() -> None:
    items = _make_items({"a": 50, "b": 50, "c": 50})
    split1 = stratified_split(items, seed=42)
    split2 = stratified_split(items, seed=42)
    for key in ("train", "val", "test"):
        assert split1[key] == split2[key]


def test_different_seed_yields_different_splits() -> None:
    items = _make_items({"a": 50, "b": 50, "c": 50})
    split1 = stratified_split(items, seed=42)
    split2 = stratified_split(items, seed=43)
    assert split1["train"] != split2["train"]


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        stratified_split([], ratios=(0.8, 0.1, 0.1), seed=42)


def test_ratios_must_sum_to_one() -> None:
    items = _make_items({"a": 30, "b": 30})
    with pytest.raises(ValueError):
        stratified_split(items, ratios=(0.5, 0.5, 0.5), seed=42)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    items = _make_items({"a": 40, "b": 40})
    # Use the tmp_path as a fake raw_root by faking file existence.
    raw_root = tmp_path / "raw"
    for label in ("a", "b"):
        (raw_root / label).mkdir(parents=True)
    for path, label in items:
        (raw_root / path).write_bytes(b"")  # placeholder file

    items_abs = [(raw_root / rel, label) for rel, label in items]
    split = stratified_split(items_abs, ratios=(0.8, 0.1, 0.1), seed=42)
    cm = build_class_map(["a", "b"])

    out_dir = tmp_path / "splits"
    save_split(split, out_dir, cm, raw_root=raw_root)

    assert (out_dir / "train.json").is_file()
    assert (out_dir / "val.json").is_file()
    assert (out_dir / "test.json").is_file()
    assert (out_dir / "class_map.json").is_file()

    loaded = load_split(out_dir / "train.json")
    assert all(isinstance(e, SplitEntry) for e in loaded)
    assert load_class_map(out_dir / "class_map.json") == cm
    # Paths in JSON should be relative POSIX paths.
    for entry in loaded:
        assert "\\" not in entry.path
        assert not Path(entry.path).is_absolute()


def test_corrupt_files_excluded_via_discover(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "a").mkdir(parents=True)
    (raw_root / "b").mkdir(parents=True)
    (raw_root / "a" / "good.jpg").write_bytes(b"x")
    (raw_root / "a" / "bad.jpg").write_bytes(b"x")
    (raw_root / "b" / "good.jpg").write_bytes(b"x")

    items = discover_class_folder_items(raw_root, exclude_paths={"a/bad.jpg"})
    paths = {p.relative_to(raw_root).as_posix() for p, _ in items}
    assert "a/good.jpg" in paths
    assert "b/good.jpg" in paths
    assert "a/bad.jpg" not in paths


def test_build_class_map_is_sorted_alphabetically() -> None:
    cm = build_class_map(["zebra", "ant", "mango", "ant"])
    assert cm == {"ant": 0, "mango": 1, "zebra": 2}


def test_save_split_json_is_valid_json(tmp_path: Path) -> None:
    items = _make_items({"a": 30, "b": 30, "c": 30})
    raw_root = tmp_path / "raw"
    for label in ("a", "b", "c"):
        (raw_root / label).mkdir(parents=True)
    for path, _ in items:
        (raw_root / path).write_bytes(b"")
    items_abs = [(raw_root / rel, lab) for rel, lab in items]
    split = stratified_split(items_abs, seed=42)
    cm = build_class_map(["a", "b", "c"])
    out_dir = tmp_path / "splits"
    save_split(split, out_dir, cm, raw_root=raw_root)
    for name in ("train", "val", "test"):
        with (out_dir / f"{name}.json").open() as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        for row in data:
            assert {"path", "label", "label_idx"} <= row.keys()
