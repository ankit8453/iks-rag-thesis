"""Generate stratified 80/10/10 train/val/test JSON splits for every
non-cross-region dataset.

Drives :func:`stratified_split` across the registered Phase-4 datasets
(excluding IRSID, whose role is the cross-region soil eval — that split
is built by ``scripts/build_soil_cross_region.py``).

Reads any existing corrupt-file lists under ``results/`` and excludes
those paths from the items being split. Writes JSON splits into
``data/splits/<dataset>/`` plus a ``class_map.json`` per dataset.

Idempotent: re-runs overwrite the prior JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._dataset_specs import DATASET_SPECS  # noqa: E402
from src.utils.data_splits import (  # noqa: E402
    build_class_map,
    save_split,
    stratified_split,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Datasets that get a standard 80/10/10 split.
DATASETS_WITH_STANDARD_SPLIT = {
    "plantvillage",
    "plantdoc",
    "paddy_doctor",
    "phantomfs",
    "sirajganj_moisture",
    # olid_i deliberately omitted: it's multi-label and uses
    # scripts/build_olid_artifacts.py (MultilabelStratifiedShuffleSplit)
    # instead of the standard single-label stratified split.
}


def _load_exclude_list(dataset_name: str) -> set[str]:
    """Read ``results/corrupt_files_<dataset>.txt`` if it exists."""
    path = RESULTS_DIR / f"corrupt_files_{dataset_name}.txt"
    if not path.is_file():
        return set()
    excluded: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rel = line.split("\t", 1)[0].strip()
            if rel:
                excluded.add(rel)
    return excluded


def _build_one(spec_name: str) -> int:
    spec = next(s for s in DATASET_SPECS if s.name == spec_name)
    if not spec.raw_root.is_dir():
        _LOGGER.warning("Skipping %s — raw root missing.", spec.name)
        return 1

    exclude = _load_exclude_list(spec.name)
    items = spec.discover_fn(spec.raw_root, exclude)
    if not items:
        _LOGGER.warning("Skipping %s — no items discovered.", spec.name)
        return 1

    labels = sorted({lab for _, lab in items})
    _LOGGER.info(
        "%s: %d items across %d classes -> splitting 80/10/10.",
        spec.name,
        len(items),
        len(labels),
    )

    split = stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=42)
    class_map = build_class_map(labels)
    output_dir = SPLITS_DIR / spec.name
    save_split(split, output_dir, class_map, raw_root=spec.raw_root)
    return 0


def main() -> int:
    rc = 0
    for spec in DATASET_SPECS:
        if spec.name not in DATASETS_WITH_STANDARD_SPLIT:
            continue
        rc |= _build_one(spec.name)
    if rc:
        _LOGGER.warning("One or more datasets were skipped — see warnings above.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
