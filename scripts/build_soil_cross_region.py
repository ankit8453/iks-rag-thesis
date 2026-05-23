"""Build the §14-mandated soil cross-region split.

- Train + val: Phantom-fs (90/10 stratified, seed=42).
- Test: 100% of the IRSID Kaggle mirror.

Output: ``data/splits/soil_cross_region/{train,val,test}.json`` plus a
``soil_cross_region_meta.json`` carrying the label-space note (deposit
labels vs texture labels — these are categorically different, see the
adjacent ``README.md``).

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._dataset_specs import get_spec  # noqa: E402
from src.utils.data_splits import (  # noqa: E402
    SplitEntry,
    build_class_map,
    make_soil_cross_region_split,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

_LOGGER = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "soil_cross_region"


def _write_split(items: list[tuple[Path, str]], path: Path, class_map: dict[str, int],
                 raw_roots: dict[str, Path]) -> None:
    entries: list[dict[str, object]] = []
    for image_path, label in items:
        # Pick whichever raw_root the path lives under.
        for tag, root in raw_roots.items():
            try:
                rel = image_path.resolve().relative_to(root.resolve())
                rel_str = f"{tag}/{rel.as_posix()}"
                break
            except ValueError:
                continue
        else:
            rel_str = image_path.as_posix()
        entries.append(
            SplitEntry(path=rel_str, label=label, label_idx=class_map[label]).model_dump()
        )
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    _LOGGER.info("Saved %s (%d entries)", path, len(entries))


def main() -> int:
    phantom_spec = get_spec("phantomfs")
    irsid_spec = get_spec("irsid")

    if not phantom_spec.raw_root.is_dir() or not irsid_spec.raw_root.is_dir():
        _LOGGER.error("Phantom-fs or IRSID raw root missing — cannot build cross-region split.")
        return 1

    phantom_items = phantom_spec.discover_fn(phantom_spec.raw_root, None)
    irsid_items = irsid_spec.discover_fn(irsid_spec.raw_root, None)

    split = make_soil_cross_region_split(
        phantom_items, irsid_items, val_frac=0.1, seed=42
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Two class maps, one per label space (deposit vs texture).
    phantom_labels = sorted({lab for _, lab in phantom_items})
    irsid_labels = sorted({lab for _, lab in irsid_items})
    phantom_map = build_class_map(phantom_labels)
    irsid_map = build_class_map(irsid_labels)

    raw_roots = {
        "phantomfs": phantom_spec.raw_root,
        "irsid": irsid_spec.raw_root,
    }

    _write_split(split["train"], OUTPUT_DIR / "train.json", phantom_map, raw_roots)
    _write_split(split["val"], OUTPUT_DIR / "val.json", phantom_map, raw_roots)
    _write_split(split["test"], OUTPUT_DIR / "test.json", irsid_map, raw_roots)

    meta = {
        "harmonisation_note": (
            "Phantom-fs labels are deposit names (Alluvial / Arid / Black / "
            "Laterite / Mountain / Red / Yellow); IRSID labels are texture "
            "names (Clay / Sand / Silt). These are different label spaces "
            "and must NOT be merged. Phase 6 evaluation reports two "
            "numbers: same-distribution accuracy on Phantom-fs val, and a "
            "transfer-learning style stress test on IRSID test."
        ),
        "train_source": "phantomfs",
        "val_source": "phantomfs",
        "test_source": "irsid",
        "train_class_map": phantom_map,
        "test_class_map": irsid_map,
        "train_size": len(split["train"]),
        "val_size": len(split["val"]),
        "test_size": len(split["test"]),
    }
    with (OUTPUT_DIR / "soil_cross_region_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    _LOGGER.info("Saved soil_cross_region_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
