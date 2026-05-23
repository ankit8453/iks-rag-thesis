"""Upload Paddy Doctor splits to the HF Hub as ``ankit-iiitdmj/iks-paddy-doctor``.

Thin per-dataset wrapper around :class:`HFDatasetUploader`. Run after
``scripts/download_paddy_doctor.py`` + ``scripts/build_splits.py``.

Source citation: Petchiammal A., Briskline Kiruba S., Pandarasamy A.,
Dhandapani M. (2023). "Paddy Doctor: A Visual Image Dataset for
Automated Paddy Disease Classification and Benchmarking." CODS-COMAD
2023. (Kaggle competition ``paddy-disease-classification``.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hf_upload import HFDatasetUploader  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR, PROJECT_ROOT  # noqa: E402

_LOGGER = get_logger(__name__)

HUB_REPO_ID = "ankit-iiitdmj/iks-paddy-doctor"
# Split JSONs encode paths like ``train_images/<class>/<file>`` relative
# to the dataset's raw/ directory (per scripts/build_splits.py +
# scripts/_dataset_specs.py::_discover_paddy_doctor).
RAW_ROOT = DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "paddy_doctor"
CLASS_MAP = SPLITS_DIR / "class_map.json"

DATASET_CARD_BODY = """\
Stratified 80/10/10 train/val/test split of the labelled training set
from the Kaggle ``paddy-disease-classification`` competition
(Petchiammal A., Briskline Kiruba S., Pandarasamy A., Dhandapani M.
(2023). "Paddy Doctor: A Visual Image Dataset for Automated Paddy
Disease Classification and Benchmarking." CODS-COMAD 2023). 10 classes
(9 disease + healthy).

**Not** the competition's test set — that set is unlabelled
(leaderboard). All entries here come from the labelled training
images; the 80/10/10 split is internal to this project and seeded at
42 for reproducibility.

Pre-built for thesis pipeline use at IIITDM Jabalpur (Author: Ankit
Pawar, Supervisor: Dr. Akshay Pandey).
"""


def main() -> int:
    if not RAW_ROOT.is_dir():
        _LOGGER.error(
            "Raw root missing at %s. Run scripts/download_paddy_doctor.py first.",
            RAW_ROOT,
        )
        return 1
    if not (SPLITS_DIR / "train.json").is_file():
        _LOGGER.error(
            "Split JSONs missing in %s. Run scripts/build_splits.py first.",
            SPLITS_DIR,
        )
        return 1

    uploader = HFDatasetUploader()
    result = uploader.upload_image_classification_dataset(
        local_root=RAW_ROOT,
        splits_dir=SPLITS_DIR,
        class_map_path=CLASS_MAP,
        hub_repo_id=HUB_REPO_ID,
        private=True,
        license_str="cc-by-4.0",
        dataset_card_body=DATASET_CARD_BODY,
    )
    _LOGGER.info(
        "Paddy Doctor uploaded -> %s (%.1fs, train=%d, val=%d, test=%d, classes=%d)",
        result.hub_url,
        result.elapsed_seconds,
        result.n_train,
        result.n_val,
        result.n_test,
        result.n_classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
