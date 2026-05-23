"""Upload PlantVillage splits to the HF Hub as ``ankit-iiitdmj/iks-plantvillage``.

Source citation: Hughes D.P. & Salathé M. (2015). "An open access
repository of images on plant health to enable the development of
mobile disease diagnostics." arXiv:1511.08060. License: CC0.

The upload uses **only the ``color`` variant** (54,305 RGB images,
38 classes). The Kaggle archive ships ``color/grayscale/segmented``
variants — the latter two are redundant for our CNN backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hf_upload import HFDatasetUploader  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR, PROJECT_ROOT  # noqa: E402

_LOGGER = get_logger(__name__)

HUB_REPO_ID = "ankit-iiitdmj/iks-plantvillage"

# Split JSONs encode paths like
# ``plantvillage dataset/color/<class>/<file>`` relative to the dataset's
# raw/ directory (per scripts/build_splits.py +
# scripts/_dataset_specs.py::_discover_plantvillage). Only entries
# under the ``color`` variant are referenced; ``grayscale`` and
# ``segmented`` on disk are ignored because the split itself filters.
RAW_ROOT = DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "plantvillage"
CLASS_MAP = SPLITS_DIR / "class_map.json"

DATASET_CARD_BODY = """\
Stratified 80/10/10 train/val/test split of the PlantVillage ``color``
variant. 54,305 RGB leaf images across 38 disease + healthy classes.

Source: Hughes D.P. & Salathé M. (2015). "An open access repository
of images on plant health to enable the development of mobile disease
diagnostics." arXiv:1511.08060. License: CC0 (public domain).

The Kaggle archive ``abdallahalidev/plantvillage-dataset`` ships three
variants — ``color``, ``grayscale``, ``segmented``. **This release
contains the ``color`` variant only**; grayscale and segmented are
redundant for our CNN backbone and would triple the storage.

Pre-built for thesis pipeline use at IIITDM Jabalpur (Author: Ankit
Pawar, Supervisor: Dr. Akshay Pandey).
"""


def main() -> int:
    if not RAW_ROOT.is_dir():
        _LOGGER.error(
            "Raw color/ variant missing at %s. Run scripts/download_plantvillage.py first.",
            RAW_ROOT,
        )
        return 1
    if not (SPLITS_DIR / "train.json").is_file():
        _LOGGER.error("Split JSONs missing in %s. Run scripts/build_splits.py first.", SPLITS_DIR)
        return 1

    uploader = HFDatasetUploader()
    result = uploader.upload_image_classification_dataset(
        local_root=RAW_ROOT,
        splits_dir=SPLITS_DIR,
        class_map_path=CLASS_MAP,
        hub_repo_id=HUB_REPO_ID,
        private=True,
        license_str="cc0-1.0",
        dataset_card_body=DATASET_CARD_BODY,
    )
    _LOGGER.info(
        "PlantVillage uploaded -> %s (%.1fs, train=%d, val=%d, test=%d, classes=%d)",
        result.hub_url, result.elapsed_seconds,
        result.n_train, result.n_val, result.n_test, result.n_classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
