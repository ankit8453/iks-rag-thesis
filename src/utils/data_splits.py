"""Stratified train/val/test split utilities (Phase 4 §C, §D).

Per supervisor guardrail #5: images never move. Splits are JSON files
listing ``(relative_path, label, label_idx)`` triples plus a
``class_map.json`` that holds the label→index mapping.

Per supervisor guardrail #1: every random operation is seeded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

_LOGGER = get_logger(__name__)

SplitName = Literal["train", "val", "test"]


class SplitEntry(BaseModel):
    """One row in a split JSON file.

    Attributes
    ----------
    path : str
        Relative POSIX path under the dataset's ``raw/`` directory.
    label : str
        Human-readable class label.
    label_idx : int
        Index of ``label`` in ``class_map.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    label: str
    label_idx: int


def _stratified_two_stage(
    items: list[tuple[Path, str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[SplitName, list[tuple[Path, str]]]:
    """Split ``items`` into train/val/test, stratified by class.

    Implemented via two calls to ``sklearn.model_selection.train_test_split``
    (first peeling off the test set, then splitting the remainder). The
    second-stage val fraction is computed so the overall ratios match
    ``ratios`` to within rounding.
    """
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    train_r, val_r, test_r = ratios
    if not abs((train_r + val_r + test_r) - 1.0) < 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    paths = [p for p, _ in items]
    labels = [lab for _, lab in items]

    # Stage 1: peel off the test set.
    rest_paths, test_paths, rest_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_r,
        random_state=seed,
        stratify=labels,
    )

    # Stage 2: split rest into train + val. After removing test_r, val
    # takes up val_r / (1 - test_r) of what's left.
    val_fraction_of_rest = val_r / (train_r + val_r)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        rest_paths,
        rest_labels,
        test_size=val_fraction_of_rest,
        random_state=seed,
        stratify=rest_labels,
    )

    return {
        "train": list(zip(train_paths, train_labels, strict=True)),
        "val": list(zip(val_paths, val_labels, strict=True)),
        "test": list(zip(test_paths, test_labels, strict=True)),
    }


def stratified_split(
    items: list[tuple[Path, str]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[SplitName, list[tuple[Path, str]]]:
    """Public entrypoint — two-stage stratified split.

    Parameters
    ----------
    items : list of (path, label)
        Input items to split. ``path`` may be absolute or relative; the
        caller decides what semantics to use when persisting via
        :func:`save_split`.
    ratios : (train, val, test)
        Must sum to 1.0. Default ``(0.8, 0.1, 0.1)`` per the locked
        decisions table.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    dict
        Keys ``"train"``, ``"val"``, ``"test"``. Values are lists of
        ``(path, label)`` tuples in the same shape as the input.
    """
    if not items:
        raise ValueError("stratified_split received zero items.")
    set_global_seed(seed)
    return _stratified_two_stage(items, ratios, seed)


def build_class_map(labels: Iterable[str]) -> dict[str, int]:
    """Return a deterministic ``label -> index`` map (sorted alphabetically)."""
    return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def save_split(
    split_dict: dict[SplitName, list[tuple[Path, str]]],
    output_dir: Path,
    class_map: dict[str, int],
    *,
    raw_root: Path,
) -> None:
    """Persist a split to ``output_dir`` as JSON files.

    Writes ``train.json``, ``val.json``, ``test.json``, and
    ``class_map.json``. Paths in the JSON are made relative to
    ``raw_root`` so the JSON survives a repository move.

    Parameters
    ----------
    split_dict : dict
        Output of :func:`stratified_split`.
    output_dir : Path
        Where to write the four JSON files.
    class_map : dict
        Label→index mapping. Use :func:`build_class_map` to construct.
    raw_root : Path
        Root the JSON ``path`` field is relative to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(raw_root).resolve()

    for split_name, items in split_dict.items():
        entries: list[dict[str, object]] = []
        for path, label in items:
            try:
                rel = Path(path).resolve().relative_to(raw_root)
            except ValueError:
                # ``path`` already relative.
                rel = Path(path)
            entry = SplitEntry(
                path=rel.as_posix(),
                label=label,
                label_idx=class_map[label],
            )
            entries.append(entry.model_dump())
        out_path = output_dir / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        _LOGGER.info(
            "Saved %s -> %d entries (%s)", out_path.name, len(entries), out_path
        )

    cm_path = output_dir / "class_map.json"
    with cm_path.open("w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)
    _LOGGER.info("Saved class_map.json -> %d classes (%s)", len(class_map), cm_path)


def load_split(path: Path) -> list[SplitEntry]:
    """Load a split JSON file produced by :func:`save_split`.

    Returns a list of :class:`SplitEntry`. Raises on bad rows.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [SplitEntry.model_validate(row) for row in rows]


def load_class_map(path: Path) -> dict[str, int]:
    """Load a ``class_map.json`` file as a dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_class_folder_items(
    raw_root: Path,
    exclude_paths: set[str] | None = None,
    *,
    image_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ),
) -> list[tuple[Path, str]]:
    """Walk a class-folder directory layout and return ``(path, label)`` pairs.

    Assumes ``raw_root`` contains one subdirectory per class, each
    populated with images. Skips paths whose POSIX form (relative to
    ``raw_root``) appears in ``exclude_paths`` — the typical use is to
    pass the corrupt-file list from :mod:`src.utils.data_validators`.
    """
    raw_root = Path(raw_root)
    exclude = exclude_paths or set()
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in image_extensions:
                continue
            rel = image_path.relative_to(raw_root).as_posix()
            if rel in exclude:
                continue
            items.append((image_path, label))
    return items


# ---------------------------------------------------------------------------
# Soil cross-region split (Phase 4 §D, PDF §14)
# ---------------------------------------------------------------------------


def make_soil_cross_region_split(
    phantomfs_items: list[tuple[Path, str]],
    irsid_items: list[tuple[Path, str]],
    *,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[SplitName, list[tuple[Path, str]]]:
    """Build the §14-mandated cross-region soil split.

    - **Train**: ``1 - val_frac`` of Phantom-fs, stratified by class.
    - **Val**: ``val_frac`` of Phantom-fs, stratified by class.
    - **Test**: 100% of IRSID.

    Phantom-fs labels are deposit names (alluvial / black / clay / red /
    laterite / peat / yellow); IRSID labels are texture names (sand /
    clay / sandy_loam / loam / loamy_sand). These are **different label
    spaces** — they are NOT merged. The cross-region evaluation in
    Phase 6 will report two separate metrics; see
    ``data/splits/soil_cross_region/README.md``.
    """
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    if not phantomfs_items:
        raise ValueError("phantomfs_items is empty.")
    if not irsid_items:
        raise ValueError("irsid_items is empty.")

    set_global_seed(seed)
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    paths = [p for p, _ in phantomfs_items]
    labels = [lab for _, lab in phantomfs_items]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_frac,
        random_state=seed,
        stratify=labels,
    )

    return {
        "train": list(zip(train_paths, train_labels, strict=True)),
        "val": list(zip(val_paths, val_labels, strict=True)),
        "test": list(irsid_items),
    }


__all__ = [
    "SplitEntry",
    "build_class_map",
    "discover_class_folder_items",
    "load_class_map",
    "load_split",
    "make_soil_cross_region_split",
    "save_split",
    "stratified_split",
]
