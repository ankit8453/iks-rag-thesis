"""Per-dataset adapters: where each dataset's images live + how labels are read.

Each adapter returns a list of ``(image_path, label_str)`` tuples ready
to feed into :mod:`src.utils.data_splits`. Layouts differ a lot — see
``PHASE4_SUMMARY.md`` for the discovered structure of each.

Conventions:
- Image paths are absolute (under the project's ``DATA_*`` constants).
- Labels are strings; ``build_class_map`` later assigns indices.
- For datasets whose Kaggle/Zenodo packaging contains extra variants
  we ignore the ones not in scope (e.g. Phantom-fs ``CyAUG-Dataset``).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.utils.data_splits import discover_class_folder_items
from src.utils.paths import DATA_DIR, DATA_PLANT_DISEASE_DIR, DATA_SOIL_DIR


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing where a dataset lives and how to enumerate it."""

    name: str
    role: str
    raw_root: Path
    image_size: int
    discover_fn: Callable[[Path, set[str] | None], list[tuple[Path, str]]]
    notes: str = ""


# --------------------------------------------------------------------------- #
# Per-dataset discovery functions
# --------------------------------------------------------------------------- #


def _discover_plantvillage(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """PlantVillage: drill into ``plantvillage dataset/color/<class>/*.jpg``.

    The Kaggle archive contains three variants: ``color``, ``grayscale``,
    and ``segmented``. We use ``color`` (RGB, full resolution).
    """
    color_root = raw_root / "plantvillage dataset" / "color"
    if not color_root.is_dir():
        raise FileNotFoundError(f"Expected PlantVillage color/ under {color_root}")
    return discover_class_folder_items(color_root, exclude_paths=exclude)


def _discover_plantdoc(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """PlantDoc: per-class folders at the root after our download merged train+test."""
    return discover_class_folder_items(raw_root, exclude_paths=exclude)


def _discover_paddy_doctor(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """Paddy Doctor: train_images/<class>/<image>.

    The Kaggle competition zip puts class folders under ``train_images``;
    ``train.csv`` carries the same labels plus variety/age metadata
    (which we ignore for Phase 4).
    """
    train_dir = raw_root / "train_images"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected Paddy Doctor train_images/ under {train_dir}")
    return discover_class_folder_items(train_dir, exclude_paths=exclude)


def _discover_phantomfs(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """Phantom-fs: drill into ``Orignal-Dataset/<class>/*`` (CyAUG ignored).

    Note the upstream typo: ``Orignal-Dataset`` (not "Original"). We use
    only the original variant for Phase 4 per the locked decisions; the
    CyAUG augmented variant is deferred to Phase 6.
    """
    original_root = raw_root / "Orignal-Dataset"
    if not original_root.is_dir():
        raise FileNotFoundError(f"Expected Phantom-fs Orignal-Dataset/ under {original_root}")
    return discover_class_folder_items(original_root, exclude_paths=exclude)


def _discover_irsid(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """IRSID Kaggle mirror: flat directory + ``Practical_Reading.csv``.

    The CSV is tab-separated with columns ``Sample, Sand, Silt, Clay, Type``.
    Per master reference §14, only the categorical ``Type`` column is
    exposed as a label; ``Sand``, ``Silt``, ``Clay`` are sieve-analysis
    numbers and must NOT be exposed as model targets.
    """
    csv_path = raw_root / "Practical_Reading.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected IRSID CSV at {csv_path}")
    exclude = exclude or set()

    items: list[tuple[Path, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        # The CSV uses tabs; csv.Sniffer is sometimes wrong, so we set it.
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sample_id = row["Sample"].strip()
            label = row["Type"].strip()
            image_path = raw_root / f"Sample{sample_id}.jpg"
            if not image_path.is_file():
                continue
            rel = image_path.relative_to(raw_root).as_posix()
            if rel in exclude:
                continue
            items.append((image_path, label))
    return items


def _discover_olid_i(raw_root: Path, exclude: set[str] | None) -> list[tuple[Path, str]]:
    """OLID I (smoke sample, Phase 4): per-folder labels.

    Phase 4 ships only the bottle-gourd archive; folders are
    ``bottle_gourd__DM``, ``bottle_gourd__JAS``, ``bottle_gourd__healthy``.
    Multi-label semantics are imposed in Phase 11 by the dataset class
    (one folder maps to one disease label here; multiple labels per
    image arrive with the full Zenodo download).
    """
    exclude = exclude or set()
    items: list[tuple[Path, str]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for class_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for image in sorted(class_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in image_exts:
                continue
            rel = image.relative_to(raw_root).as_posix()
            if rel in exclude:
                continue
            items.append((image, label))
    return items


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="plantvillage",
        role="disease pretraining",
        raw_root=DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw",
        image_size=380,
        discover_fn=_discover_plantvillage,
        notes="38 classes, color variant only.",
    ),
    DatasetSpec(
        name="plantdoc",
        role="disease real-field eval",
        raw_root=DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw",
        image_size=380,
        discover_fn=_discover_plantdoc,
        notes="train + test merged into class folders at raw/.",
    ),
    DatasetSpec(
        name="paddy_doctor",
        role="Indian rice disease",
        raw_root=DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw",
        image_size=380,
        discover_fn=_discover_paddy_doctor,
        notes="Kaggle competition variant: 10 classes, 10,407 train images.",
    ),
    DatasetSpec(
        name="phantomfs",
        role="soil primary",
        raw_root=DATA_SOIL_DIR / "phantomfs" / "raw",
        image_size=224,
        discover_fn=_discover_phantomfs,
        notes="Orignal-Dataset (sic) only; CyAUG deferred to Phase 6.",
    ),
    DatasetSpec(
        name="irsid",
        role="soil cross-region (§14)",
        raw_root=DATA_SOIL_DIR / "irsid" / "raw",
        image_size=224,
        discover_fn=_discover_irsid,
        notes=(
            "Kaggle mirror: only 16 samples + Practical_Reading.csv. "
            "Sand/Silt/Clay sieve numbers are NOT used; Type column only "
            "per §14. Full IEEE DataPort version pending — see TODO[IEEE]."
        ),
    ),
    DatasetSpec(
        name="olid_i",
        role="causation (C5)",
        raw_root=DATA_DIR / "causation" / "olid_i" / "raw",
        image_size=380,
        discover_fn=_discover_olid_i,
        notes=(
            "Phase 4 smoke sample (bottle_gourd only, ~448 MB). Full ~14 GB "
            "Zenodo download deferred to Phase 11; multi-label semantics "
            "applied by the dataset class in src/integration/causation_dataset.py."
        ),
    ),
]


def get_spec(name: str) -> DatasetSpec:
    for spec in DATASET_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown dataset spec: {name}")
