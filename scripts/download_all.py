"""Orchestrate all six Phase 4 dataset downloads.

Runs each per-dataset script in sequence (never in parallel — Kaggle
rate-limits and parallel requests get the API key throttled). Logs a
total time estimate at start and per-dataset wall-clock at the end.

If any one download fails (non-zero exit, raised exception), the
orchestrator logs the failure and continues with the next dataset so we
can proceed on whatever data is available. Failed datasets are listed
at the end and should be re-attempted by running their script directly.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

# (module name, friendly name, rough size estimate in GB for logging)
DOWNLOAD_PLAN: list[tuple[str, str, float]] = [
    ("scripts.download_plantvillage", "PlantVillage", 2.0),
    ("scripts.download_plantdoc", "PlantDoc", 0.3),
    ("scripts.download_paddy_doctor", "Paddy Doctor", 1.5),
    ("scripts.download_phantomfs_soil", "Phantom-fs Soil", 0.5),
    ("scripts.download_irsid", "IRSID", 0.3),
    ("scripts.download_olid_i", "OLID I", 0.8),
]


def main() -> int:
    total_estimate_gb = sum(g for _, _, g in DOWNLOAD_PLAN)
    _LOGGER.info(
        "Phase 4 dataset acquisition: %d datasets, ~%.1f GB total estimate.",
        len(DOWNLOAD_PLAN),
        total_estimate_gb,
    )

    results: list[tuple[str, str, float]] = []  # (name, status, elapsed_s)
    overall_start = time.monotonic()

    for module_name, friendly_name, _gb in DOWNLOAD_PLAN:
        _LOGGER.info("==== %s ====", friendly_name)
        start = time.monotonic()
        try:
            mod = importlib.import_module(module_name)
            rc = mod.main()
            status = "ok" if rc == 0 else f"non-zero exit ({rc})"
        except Exception as exc:  # noqa: BLE001
            status = f"failed: {exc}"
            _LOGGER.exception("Download %s raised", friendly_name)
        elapsed = time.monotonic() - start
        results.append((friendly_name, status, elapsed))
        _LOGGER.info("%s -> %s in %.1fs", friendly_name, status, elapsed)

    overall = time.monotonic() - overall_start
    _LOGGER.info("==== summary ====")
    for name, status, elapsed in results:
        _LOGGER.info("  %-18s %-30s %6.1fs", name, status, elapsed)
    _LOGGER.info("Total elapsed: %.1f minutes", overall / 60)

    n_failed = sum(1 for _, status, _ in results if status != "ok")
    return 0 if n_failed == 0 else n_failed


if __name__ == "__main__":
    raise SystemExit(main())
