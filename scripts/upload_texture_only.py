"""Upload ONLY the texture (IRSID+VIT merged) dataset to HF Hub.

Thin wrapper around :mod:`scripts.prepare_soil_hf_datasets` for the
case where the operator wants to run uploads one repo at a time. The
sirajganj moisture push is intentionally NOT triggered here — run
``python scripts/prepare_soil_hf_datasets.py`` (which auto-skips repos
already on Hub) once you're ready to push sirajganj.

Run from the project root:

    python scripts/upload_texture_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_soil_hf_datasets import (  # noqa: E402
    REPO_TEXTURE,
    TEXTURE_CLASSES_PATH,
    _load_yaml,
    _preflight_auth,
    _repo_done,
    prepare_texture_merged,
    push_to_hf,
)
from src.utils.logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)


def main() -> int:
    _preflight_auth()

    if _repo_done(REPO_TEXTURE):
        _LOGGER.info("Skip texture — Hub already has parquet shards.")
        return 0

    tx_cfg = _load_yaml(TEXTURE_CLASSES_PATH)
    tx_train, tx_val, tx_test = prepare_texture_merged()
    push_to_hf(
        tx_train, tx_val, tx_test,
        repo_name=REPO_TEXTURE,
        head="texture",
        classes={int(k): v for k, v in tx_cfg["classes"].items()},
        description=(
            "IRSID + VIT latha-soil merged, USDA-collapsed to "
            "{coarse, fine, mixed}. The `source` column ('irsid' / 'vit') "
            "supports per-source ablation."
        ),
    )
    print()
    print(
        f"Texture push complete: {REPO_TEXTURE} "
        f"(train={len(tx_train)}, val={len(tx_val)}, test={len(tx_test)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
