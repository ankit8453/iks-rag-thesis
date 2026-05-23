"""Run image-integrity validation across every Phase 4 dataset.

Walks each dataset's raw/ directory via :func:`validate_image_directory`,
prints a per-dataset summary, and writes a ``results/corrupt_files_*.txt``
file for any dataset that has corrupt or unreadable images.

Idempotent — safe to re-run after re-downloading any dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._dataset_specs import DATASET_SPECS  # noqa: E402
from src.utils.data_validators import (  # noqa: E402
    validate_image_directory,
    write_corrupt_list,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import RESULTS_DIR  # noqa: E402

_LOGGER = get_logger(__name__)


def main() -> int:
    summary: list[tuple[str, int, int, int]] = []  # (name, total, valid, corrupt)
    failed = 0
    for spec in DATASET_SPECS:
        if not spec.raw_root.is_dir():
            _LOGGER.warning(
                "Dataset %s: raw root %s missing — skipping. "
                "Run the corresponding scripts/download_*.py first.",
                spec.name,
                spec.raw_root,
            )
            failed += 1
            continue
        report = validate_image_directory(spec.raw_root, spec.name)
        write_corrupt_list(report, RESULTS_DIR)
        summary.append(
            (
                spec.name,
                report.total_files,
                report.valid_files,
                len(report.corrupt_files),
            )
        )

    _LOGGER.info("==== validation summary ====")
    _LOGGER.info("  %-15s %8s %8s %8s", "dataset", "total", "valid", "corrupt")
    for name, total, valid, corrupt in summary:
        _LOGGER.info("  %-15s %8d %8d %8d", name, total, valid, corrupt)
    if failed:
        _LOGGER.warning("%d dataset(s) missing on disk.", failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
