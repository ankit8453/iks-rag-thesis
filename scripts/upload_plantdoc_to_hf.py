"""Upload PlantDoc splits to the HF Hub as ``ankit-iiitdmj/iks-plantdoc``.

Source citation: Singh D., Jain N., Jain P., Kayal P., Kumawat S.,
Batra N. (2020). "PlantDoc: A Dataset for Visual Plant Disease
Detection." CODS-COMAD 2020.

The upload's canonical class count is **27** (Singh et al. 2020), not
the upstream GitHub repo's 28. The vestigial ``Tomato two spotted
spider mites leaf`` folder (2 images) was merged into ``Tomato leaf``
in Phase 5 §A — see commit ``Phase 5 §A: merge PlantDoc spider-mites
2-image class into Tomato leaf``. The two merged files retain the
filename prefix ``was-spider-mites-`` for audit traceability.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hf_upload import HFDatasetUploader  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_PLANT_DISEASE_DIR, PROJECT_ROOT  # noqa: E402

_LOGGER = get_logger(__name__)

HUB_REPO_ID = "ankit-iiitdmj/iks-plantdoc"
RAW_ROOT = DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "plantdoc"
CLASS_MAP = SPLITS_DIR / "class_map.json"

DATASET_CARD_BODY = """\
Pre-cropped, real-field plant-disease leaves with stratified 80/10/10
splits. Source: Singh D., Jain N., Jain P., Kayal P., Kumawat S.,
Batra N. (2020). "PlantDoc: A Dataset for Visual Plant Disease
Detection." CODS-COMAD 2020.

This release has **27 classes**, matching the published canonical
taxonomy. The upstream GitHub repo currently has 28 top-level
folders; the extra folder ``Tomato two spotted spider mites leaf``
contained only 2 images and was identified as a curation remnant of
the original collection process, not one of the 27 published
classes. Those 2 images were merged into ``Tomato leaf`` for this
release with the filename prefix ``was-spider-mites-`` so audits can
trace them. Supervisor sign-off (Dr. Akshay Pandey) was received
before the merge.

Pre-built for thesis pipeline use at IIITDM Jabalpur (Author: Ankit
Pawar).
"""


def main() -> int:
    if not RAW_ROOT.is_dir():
        _LOGGER.error("Raw root missing at %s. Run scripts/download_plantdoc.py first.", RAW_ROOT)
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
        license_str="cc-by-sa-4.0",
        dataset_card_body=DATASET_CARD_BODY,
    )
    _LOGGER.info(
        "PlantDoc uploaded -> %s (%.1fs, train=%d, val=%d, test=%d, classes=%d)",
        result.hub_url, result.elapsed_seconds,
        result.n_train, result.n_val, result.n_test, result.n_classes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
