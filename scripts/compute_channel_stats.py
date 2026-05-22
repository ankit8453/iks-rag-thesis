"""Compute per-dataset RGB channel mean/std from each training split.

Writes ``configs/data/<dataset>_norm.yaml`` for every dataset. For very
large training sets (PlantVillage, Paddy Doctor) we subsample to keep
the run under a few minutes; the subsample seed is fixed (42) so the
result is reproducible.

Per master reference §22: we use dataset-computed normalisation stats
rather than blanket ImageNet defaults, because agriculture imagery has
different colour distributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._dataset_specs import DATASET_SPECS  # noqa: E402
from src.utils.data_stats import (  # noqa: E402
    compute_channel_stats,
    compute_channel_stats_from_paths,
    save_channel_stats,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT  # noqa: E402

_LOGGER = get_logger(__name__)

SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
NORM_OUTPUT_ROOT = CONFIGS_DIR / "data"

# Subsample caps — chosen so each dataset finishes within a few minutes.
# Statistics converge well before the full pass on large datasets.
MAX_IMAGES: dict[str, int | None] = {
    "plantvillage": 4000,
    "plantdoc": None,
    "paddy_doctor": 4000,
    "phantomfs": None,
    "irsid": None,
    "olid_i": None,
}


def main() -> int:
    NORM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rc = 0
    for spec in DATASET_SPECS:
        cap = MAX_IMAGES.get(spec.name)
        _LOGGER.info(
            "Computing channel stats for %s (size=%d, cap=%s) ...",
            spec.name,
            spec.image_size,
            cap,
        )

        # IRSID has no standard 80/10/10 split (it's test-only under §D).
        # Discover its items directly and feed the path list in.
        if spec.name == "irsid":
            items = spec.discover_fn(spec.raw_root, None)
            paths = [p for p, _ in items]
            stats = compute_channel_stats_from_paths(
                image_paths=paths,
                image_size=spec.image_size,
                max_images=cap,
                seed=42,
            )
            out_path = NORM_OUTPUT_ROOT / f"{spec.name}_norm.yaml"
            save_channel_stats(stats, out_path)
            _LOGGER.info(
                "%s: n=%d, mean=%s, std=%s",
                spec.name, stats.n_images_sampled, stats.mean, stats.std,
            )
            continue

        split_path = SPLITS_ROOT / spec.name / "train.json"
        if not split_path.is_file():
            _LOGGER.warning("Skipping %s — no split JSON at %s", spec.name, split_path)
            rc = 1
            continue
        stats = compute_channel_stats(
            split_path=split_path,
            raw_root=spec.raw_root,
            image_size=spec.image_size,
            max_images=cap,
            seed=42,
        )
        out_path = NORM_OUTPUT_ROOT / f"{spec.name}_norm.yaml"
        save_channel_stats(stats, out_path)
        _LOGGER.info(
            "%s: n=%d, mean=%s, std=%s",
            spec.name,
            stats.n_images_sampled,
            stats.mean,
            stats.std,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
