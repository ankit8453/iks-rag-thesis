"""Hugging Face Hub upload utilities for image-classification datasets.

Wraps ``huggingface_hub`` + ``datasets`` so each per-dataset upload
script in ``scripts/`` stays short. Used by Phase 5 §C to push
PlantVillage / PlantDoc / Paddy Doctor to ``ankit-iiitdmj/iks-*``
under private visibility.

Convention: every dataset we push to HF Hub has been pre-split into
``data/splits/<name>/{train,val,test}.json`` + ``class_map.json`` by
:mod:`scripts.build_splits`. The uploader reads those, resolves each
entry's image under ``local_root``, and constructs a
``datasets.DatasetDict`` with image / label / label_idx columns.

Pre-flight auth check
---------------------
``HFDatasetUploader.__init__`` calls
``HfApi().whoami()`` and refuses to proceed unless the authenticated
user matches the expected account (``ankit-iiitdmj`` by default). This
catches misconfigured or expired tokens before any bandwidth is spent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from huggingface_hub import HfApi  # noqa: F401

_LOGGER = get_logger(__name__)

EXPECTED_HF_USERNAME = "ankit-iiitdmj"


@dataclass
class HFDatasetUploadResult:
    """Outcome of a single dataset upload."""

    hub_repo_id: str
    hub_url: str
    n_train: int
    n_val: int
    n_test: int
    n_classes: int
    elapsed_seconds: float


class HFDatasetUploader:
    """Upload one Phase-4-split-JSON dataset to the Hugging Face Hub.

    Parameters
    ----------
    expected_username : str
        Username the HF Hub token must belong to. The constructor
        raises immediately if ``whoami()['name']`` doesn't match.

    Notes
    -----
    Imports of ``huggingface_hub`` and ``datasets`` are lazy so this
    module stays importable in environments without those deps.
    """

    def __init__(self, expected_username: str = EXPECTED_HF_USERNAME) -> None:
        # Lazy import — keeps the module import-safe.
        from huggingface_hub import HfApi  # noqa: PLC0415

        self._api = HfApi()
        info = self._api.whoami()
        actual = info.get("name")
        if actual != expected_username:
            raise PermissionError(
                f"HF Hub token belongs to '{actual}', expected "
                f"'{expected_username}'. Run `huggingface-cli login` with "
                f"the correct Write token, then retry."
            )
        token = info.get("auth", {}).get("accessToken", {})
        if token.get("role") != "write":
            raise PermissionError(
                f"HF Hub token role is '{token.get('role')}', need 'write'. "
                f"Generate a Write token at https://huggingface.co/settings/tokens"
                f" and re-login via `huggingface-cli login`."
            )
        _LOGGER.info(
            "HF Hub pre-flight ok: user=%s, token role=write",
            expected_username,
        )

    # ----- helpers -------------------------------------------------- #

    def _load_split_entries(
        self, splits_dir: Path, local_root: Path
    ) -> dict[str, list[dict[str, Any]]]:
        """Read train/val/test JSON, resolve paths, return Datasets-ready rows."""
        rows: dict[str, list[dict[str, Any]]] = {}
        for split_name in ("train", "val", "test"):
            split_path = splits_dir / f"{split_name}.json"
            if not split_path.is_file():
                _LOGGER.info("No %s split at %s — skipping.", split_name, split_path)
                continue
            with split_path.open(encoding="utf-8") as fh:
                entries = json.load(fh)
            split_rows = []
            for entry in entries:
                full_path = (local_root / entry["path"]).resolve()
                if not full_path.is_file():
                    raise FileNotFoundError(
                        f"Split entry {entry['path']} resolves to missing file "
                        f"{full_path}. Did you re-run scripts/build_splits.py?"
                    )
                split_rows.append(
                    {
                        "image": str(full_path),
                        "label": entry["label"],
                        "label_idx": int(entry["label_idx"]),
                    }
                )
            rows[split_name] = split_rows
            _LOGGER.info("Loaded %s/%s: %d entries", splits_dir.name, split_name, len(split_rows))
        return rows

    def _build_dataset_dict(
        self,
        split_rows: dict[str, list[dict[str, Any]]],
    ) -> Any:
        from datasets import Dataset, DatasetDict, Image  # noqa: PLC0415

        dsd: dict[str, Dataset] = {}
        for split_name, rows in split_rows.items():
            ds = Dataset.from_list(rows).cast_column("image", Image())
            dsd[split_name] = ds
        return DatasetDict(dsd)

    def _ensure_repo(self, hub_repo_id: str, private: bool) -> None:
        self._api.create_repo(
            repo_id=hub_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

    def _write_dataset_card(
        self,
        hub_repo_id: str,
        class_map: dict[str, int],
        n_train: int,
        n_val: int,
        n_test: int,
        license_str: str | None,
        body: str | None,
    ) -> None:
        """Render and push a README.md (HF dataset card) for the repo."""
        from huggingface_hub import HfApi  # noqa: PLC0415

        header_lines = ["---"]
        if license_str:
            header_lines.append(f"license: {license_str}")
        header_lines.append("task_categories:")
        header_lines.append("  - image-classification")
        header_lines.append("size_categories:")
        n_total = n_train + n_val + n_test
        if n_total < 1_000:
            header_lines.append("  - n<1K")
        elif n_total < 10_000:
            header_lines.append("  - 1K<n<10K")
        elif n_total < 100_000:
            header_lines.append("  - 10K<n<100K")
        else:
            header_lines.append("  - 100K<n<1M")
        header_lines.append("---")
        header = "\n".join(header_lines)

        sorted_classes = sorted(class_map.items(), key=lambda kv: kv[1])
        class_list_md = "\n".join(f"- `{cls}` (idx {idx})" for cls, idx in sorted_classes)
        sizes_md = (
            f"- train: {n_train}\n"
            f"- val: {n_val}\n"
            f"- test: {n_test}\n"
            f"- total: {n_total}\n"
            f"- classes: {len(class_map)}"
        )
        card = (
            f"{header}\n\n"
            f"# {hub_repo_id}\n\n"
            f"{body or 'Phase 4 split of an image-classification dataset for the IIITDM Jabalpur thesis pipeline.'}\n\n"
            f"## Splits\n\n{sizes_md}\n\n"
            f"## Classes\n\n{class_list_md}\n\n"
            f"## Preprocessing\n\n"
            f"- 80/10/10 stratified train/val/test split (`sklearn.model_selection.train_test_split`, seed=42).\n"
            f"- Channel normalisation stats computed on this dataset's train split (see `configs/data/<name>_norm.yaml` in the upstream repo).\n"
            f"- Image-integrity verified via PIL `verify()` + load (see `src/utils/data_validators.py`).\n"
        )

        api = HfApi()
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=hub_repo_id,
            repo_type="dataset",
        )
        _LOGGER.info("Dataset card pushed to %s/README.md", hub_repo_id)

    # ----- public API ----------------------------------------------- #

    def upload_image_classification_dataset(
        self,
        local_root: Path,
        splits_dir: Path,
        class_map_path: Path,
        hub_repo_id: str,
        *,
        private: bool = True,
        license_str: str | None = None,
        dataset_card_body: str | None = None,
    ) -> HFDatasetUploadResult:
        """Push a Phase-4-split dataset to the Hub. Returns the URL + counts."""
        with class_map_path.open(encoding="utf-8") as fh:
            class_map: dict[str, int] = json.load(fh)

        split_rows = self._load_split_entries(splits_dir, local_root)
        n_train = len(split_rows.get("train", []))
        n_val = len(split_rows.get("val", []))
        n_test = len(split_rows.get("test", []))

        _LOGGER.info(
            "Upload plan: %s -> %s (train=%d, val=%d, test=%d, classes=%d, private=%s)",
            local_root,
            hub_repo_id,
            n_train,
            n_val,
            n_test,
            len(class_map),
            private,
        )

        self._ensure_repo(hub_repo_id, private)
        dsd = self._build_dataset_dict(split_rows)

        start = time.monotonic()
        dsd.push_to_hub(hub_repo_id, private=private)
        elapsed = time.monotonic() - start

        # Dataset card last so it overlays the parquet auto-card.
        self._write_dataset_card(
            hub_repo_id,
            class_map,
            n_train,
            n_val,
            n_test,
            license_str,
            dataset_card_body,
        )

        hub_url = f"https://huggingface.co/datasets/{hub_repo_id}"
        return HFDatasetUploadResult(
            hub_repo_id=hub_repo_id,
            hub_url=hub_url,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_classes=len(class_map),
            elapsed_seconds=elapsed,
        )


def verify_dataset_uploaded(hub_repo_id: str) -> dict[str, Any]:
    """Programmatic post-upload check — returns the API's info() dict."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().dataset_info(hub_repo_id)
    return {
        "id": info.id,
        "siblings": len(info.siblings or []),
        "tags": info.tags,
        "private": info.private,
    }


__all__ = [
    "EXPECTED_HF_USERNAME",
    "HFDatasetUploadResult",
    "HFDatasetUploader",
    "verify_dataset_uploaded",
]
