"""Push three Phase 6 soil-module dataset repos to Hugging Face Hub.

Three private dataset repos under ``ankit-iiitdmj/`` get populated with
80/10/10 stratified splits keyed by the multi-task soil heads:

================================  ===========================  =============
Repo                              Head supervised              Source
================================  ===========================  =============
iks-soil-phantomfs                soil_type (7 classes)        Phantom-fs Kaggle
iks-soil-sirajganj-moisture       moisture_appearance (3)      Sirajganj 2025
iks-soil-texture-irsid-vit        texture (3, USDA-collapsed)  IRSID + VIT merged
================================  ===========================  =============

Each row schema (after ``cast_column("image", Image())``):

- ``image``      decoded PIL.Image (HF auto-encodes on push, auto-decodes on load)
- ``label_idx``  int — head-local class index
- ``class_name`` str — canonical class name
- ``source``     str — provenance within the repo

We pass **file paths** as the image column and let ``datasets`` /
``huggingface_hub`` handle the encoding-and-upload pipeline. This is the
same pattern that worked for the Phase 5 disease uploads (PlantVillage
54k images / 6.6 GB) and avoids the pitfalls of an earlier draft that
pre-encoded every image into Python memory and then called
``api.upload_file`` on a single 1.67 GB parquet — that approach OOM'd
on phantomfs at PNG quality and hung on sirajganj LFS commit at JPEG
quality. ``DatasetDict.push_to_hub`` shards automatically and uses
chunked LFS uploads with resume, so even a multi-GB dataset survives
flaky connections.

This prompt only handles single-head dataset uploads. The multi-task
fusion (filling ``-1`` for non-supervised heads) happens at training
time in the DataLoader (Phase 6 Prompt 2).
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image as PILImage, UnidentifiedImageError  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import CONFIGS_DIR, DATA_SOIL_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

EXPECTED_HF_USERNAME = "ankit-iiitdmj"
SEED = 42

# Resize images so max(W,H) == this before storing into parquet. The
# training pipeline crops to 224x224 with augmentation, so 768 gives
# >3x area headroom over the train resolution while keeping each row
# small (~100 KB). Without resize, full-res phone photos (~1.8 MB
# each from Sirajganj) make sirajganj's single parquet shard hit
# ~450 MB on its own — large enough that the LFS upload reliably
# dies mid-transfer on this machine (3 reproducible crashes between
# 80-280 MB across runs).
RESIZE_MAX_DIM = 768
JPEG_QUALITY = 90

_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

PHANTOMFS_ROOT = DATA_SOIL_DIR / "phantomfs" / "raw" / "Orignal-Dataset"
SIRAJGANJ_ROOT = (
    DATA_SOIL_DIR / "sirajganj_moisture" / "raw"
    / "Soil_Moisture_Dataset" / "Before Augmentation"
)
IRSID_ROOT = DATA_SOIL_DIR / "irsid" / "raw"
VIT_ROOT = DATA_SOIL_DIR / "vit_texture" / "raw"

LABEL_MAPPING_PATH = CONFIGS_DIR / "data" / "soil_texture_label_mapping.yaml"

PHANTOMFS_CLASSES_PATH = CONFIGS_DIR / "data" / "soil_soil_type_classes.yaml"
MOISTURE_CLASSES_PATH = CONFIGS_DIR / "data" / "soil_moisture_classes.yaml"
TEXTURE_CLASSES_PATH = CONFIGS_DIR / "data" / "soil_texture_classes.yaml"

REPO_PHANTOMFS = f"{EXPECTED_HF_USERNAME}/iks-soil-phantomfs"
REPO_SIRAJGANJ = f"{EXPECTED_HF_USERNAME}/iks-soil-sirajganj-moisture"
REPO_TEXTURE = f"{EXPECTED_HF_USERNAME}/iks-soil-texture-irsid-vit"


# --------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------- #


def _preflight_auth() -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().whoami()
    actual = info.get("name")
    if actual != EXPECTED_HF_USERNAME:
        raise PermissionError(
            f"HF Hub token belongs to '{actual}', expected "
            f"'{EXPECTED_HF_USERNAME}'. Run `huggingface-cli login` "
            f"with a Write token before re-running."
        )
    token = info.get("auth", {}).get("accessToken", {})
    if token.get("role") != "write":
        raise PermissionError(
            f"HF Hub token role is '{token.get('role')}', need 'write'."
        )
    _LOGGER.info("HF Hub pre-flight ok: user=%s, role=write", actual)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


@dataclass
class _Row:
    image_bytes: bytes   # pre-resized JPEG bytes
    label_idx: int
    class_name: str
    source: str


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _image_ok(image_path: Path) -> bool:
    """``PIL.verify`` round-trip; logs and rejects unreadable files."""
    try:
        with PILImage.open(image_path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        _LOGGER.warning("PIL rejected %s: %s", image_path.name, exc)
        return False


def _resized_jpeg_bytes(image_path: Path) -> bytes | None:
    """Open + RGB-convert + resize-so-max-dim==RESIZE_MAX_DIM + JPEG-encode."""
    try:
        with PILImage.open(image_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            if max(w, h) > RESIZE_MAX_DIM:
                scale = RESIZE_MAX_DIM / max(w, h)
                rgb = rgb.resize(
                    (int(round(w * scale)), int(round(h * scale))),
                    PILImage.LANCZOS,
                )
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return buf.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        _LOGGER.warning("PIL re-encode failed for %s: %s", image_path.name, exc)
        return None


def _stratified_split(
    rows: list[_Row],
) -> tuple[list[_Row], list[_Row], list[_Row]]:
    """80/10/10 stratified split keyed by ``label_idx``, seed=42.

    Operates on plain Python lists — does NOT pre-load image bytes. The
    rows carry only file paths + small metadata, so the split is cheap
    even for thousand-image datasets.
    """
    df = pd.DataFrame(
        {
            "image_bytes": [r.image_bytes for r in rows],
            "label_idx": [r.label_idx for r in rows],
            "class_name": [r.class_name for r in rows],
            "source": [r.source for r in rows],
        }
    )
    train_df, holdout_df = train_test_split(
        df, test_size=0.20, stratify=df["label_idx"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=0.50, stratify=holdout_df["label_idx"], random_state=SEED
    )

    for df_part, name in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        per_class = df_part["label_idx"].value_counts().sort_index().to_dict()
        empty = [c for c, n in per_class.items() if n == 0]
        if empty:
            raise RuntimeError(
                f"Stratified split produced empty classes in {name}: {empty}. "
                f"Class counts: {per_class}"
            )

    return _df_to_rows(train_df), _df_to_rows(val_df), _df_to_rows(test_df)


def _df_to_rows(df: pd.DataFrame) -> list[_Row]:
    return [
        _Row(
            image_bytes=bytes(rec.image_bytes),
            label_idx=int(rec.label_idx),
            class_name=str(rec.class_name),
            source=str(rec.source),
        )
        for rec in df.itertuples(index=False)
    ]


# --------------------------------------------------------------------- #
# Per-dataset preparation (return path-only rows, NO image bytes)
# --------------------------------------------------------------------- #


def prepare_phantomfs() -> tuple[list[_Row], list[_Row], list[_Row]]:
    """Phantom-fs (soil_type head). Stratified 80/10/10 over the 7 classes."""
    cfg = _load_yaml(PHANTOMFS_CLASSES_PATH)
    folder_to_class: dict[str, str] = cfg["original_folder_mapping"]
    class_to_idx: dict[str, int] = {name: idx for idx, name in cfg["classes"].items()}

    rows: list[_Row] = []
    for folder_name, class_name in folder_to_class.items():
        class_dir = PHANTOMFS_ROOT / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Phantom-fs class dir missing: {class_dir}")
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            blob = _resized_jpeg_bytes(img_path)
            if blob is None:
                continue
            rows.append(
                _Row(
                    image_bytes=blob,
                    label_idx=class_to_idx[class_name],
                    class_name=class_name,
                    source="phantomfs",
                )
            )
    _LOGGER.info("Phantom-fs: %d valid rows ready for split.", len(rows))
    return _stratified_split(rows)


def prepare_sirajganj_moisture() -> tuple[list[_Row], list[_Row], list[_Row]]:
    """Sirajganj (moisture_appearance head). Wet/Moderate/Dry → 0/1/2."""
    cfg = _load_yaml(MOISTURE_CLASSES_PATH)
    folder_to_class: dict[str, str] = cfg["original_folder_mapping"]
    class_to_idx: dict[str, int] = {name: idx for idx, name in cfg["classes"].items()}

    rows: list[_Row] = []
    for folder_name, class_name in folder_to_class.items():
        class_dir = SIRAJGANJ_ROOT / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Sirajganj class dir missing: {class_dir}")
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            blob = _resized_jpeg_bytes(img_path)
            if blob is None:
                continue
            rows.append(
                _Row(
                    image_bytes=blob,
                    label_idx=class_to_idx[class_name],
                    class_name=class_name,
                    source="sirajganj",
                )
            )
    _LOGGER.info("Sirajganj: %d valid rows ready for split.", len(rows))
    return _stratified_split(rows)


def _read_irsid_label_map() -> dict[str, str]:
    """``Sample N`` -> raw Type string from IRSID's Practical_Reading.csv."""
    csv_path = IRSID_ROOT / "Practical_Reading.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"IRSID labels CSV missing: {csv_path}")
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["Type"] = df["Type"].astype(str).str.strip()
    return {f"Sample{int(r.Sample)}": r.Type for r in df.itertuples(index=False)}


def prepare_texture_merged() -> tuple[list[_Row], list[_Row], list[_Row]]:
    """IRSID + VIT, USDA-collapsed to {coarse, fine, mixed}. Stratified 80/10/10."""
    import re  # noqa: PLC0415

    cfg = _load_yaml(TEXTURE_CLASSES_PATH)
    class_to_idx: dict[str, int] = {name: idx for idx, name in cfg["classes"].items()}

    mapping = _load_yaml(LABEL_MAPPING_PATH)
    irsid_lookup = {
        re.sub(r"\s+", "_", k.strip().lower()): v
        for k, v in mapping["texture_mapping"].items()
    }
    vit_lookup = dict(mapping["vit_texture"])

    rows: list[_Row] = []

    # ---- IRSID ----
    irsid_labels = _read_irsid_label_map()
    for sample_name, raw_type in irsid_labels.items():
        canon = re.sub(r"\s+", "_", raw_type.strip().lower())
        coarse_fine_mixed = irsid_lookup.get(canon)
        if coarse_fine_mixed is None:
            raise RuntimeError(
                f"IRSID class {raw_type!r} (canon={canon!r}) not in "
                f"texture_mapping: {sorted(irsid_lookup)}"
            )
        img_path = IRSID_ROOT / f"{sample_name}.jpg"
        if not img_path.is_file():
            _LOGGER.warning("IRSID image missing: %s — skipping.", img_path.name)
            continue
        blob = _resized_jpeg_bytes(img_path)
        if blob is None:
            continue
        rows.append(
            _Row(
                image_bytes=blob,
                label_idx=class_to_idx[coarse_fine_mixed],
                class_name=coarse_fine_mixed,
                source="irsid",
            )
        )

    # ---- VIT ----
    for class_dir in sorted(p for p in VIT_ROOT.iterdir() if p.is_dir()):
        raw_class = class_dir.name
        coarse_fine_mixed = vit_lookup.get(raw_class)
        if coarse_fine_mixed is None:
            raise RuntimeError(
                f"VIT class {raw_class!r} not in vit_texture mapping: "
                f"{sorted(vit_lookup)}"
            )
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            blob = _resized_jpeg_bytes(img_path)
            if blob is None:
                continue
            rows.append(
                _Row(
                    image_bytes=blob,
                    label_idx=class_to_idx[coarse_fine_mixed],
                    class_name=coarse_fine_mixed,
                    source="vit",
                )
            )

    _LOGGER.info("Texture (IRSID+VIT): %d valid rows ready for split.", len(rows))
    return _stratified_split(rows)


# --------------------------------------------------------------------- #
# HF Hub push (uses datasets.Dataset.push_to_hub — auto-shards, resumes)
# --------------------------------------------------------------------- #


def _rows_to_dataset(rows: list[_Row]):
    """Build a ``datasets.Dataset`` with an ``Image()`` column from pre-resized JPEG bytes.

    Bytes are pre-resized so the resulting parquet shards stay small
    (each row ~100 KB) — this keeps single-shard upload time well below
    the 3-min window that triggered reproducible silent crashes on
    larger sirajganj transfers.
    """
    from datasets import Dataset, Image  # noqa: PLC0415

    payload = [
        {
            "image": {"bytes": r.image_bytes, "path": None},
            "label_idx": r.label_idx,
            "class_name": r.class_name,
            "source": r.source,
        }
        for r in rows
    ]
    return Dataset.from_list(payload).cast_column("image", Image())


def _dataset_card(
    *,
    repo_id: str,
    head: str,
    classes: dict[int, str],
    n_train: int,
    n_val: int,
    n_test: int,
    description: str,
) -> str:
    total = n_train + n_val + n_test
    if total < 1_000:
        size_cat = "n<1K"
    elif total < 10_000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    class_md = "\n".join(
        f"- `{name}` (idx {idx})" for idx, name in sorted(classes.items())
    )
    return (
        f"---\n"
        f"task_categories:\n  - image-classification\n"
        f"size_categories:\n  - {size_cat}\n"
        f"---\n\n"
        f"# {repo_id}\n\n"
        f"{description}\n\n"
        f"## Head supervised\n\n`{head}`\n\n"
        f"## Splits\n\n"
        f"- train: {n_train}\n- val: {n_val}\n- test: {n_test}\n- total: {total}\n"
        f"- classes: {len(classes)}\n\n"
        f"## Classes\n\n{class_md}\n\n"
        f"## Schema\n\n"
        f"- `image` — HF `Image()` column (lazy PIL decode)\n"
        f"- `label_idx` — int, head-local class index\n"
        f"- `class_name` — canonical class name\n"
        f"- `source` — dataset-source identifier within this repo\n\n"
        f"## Preprocessing\n\n"
        f"- 80/10/10 stratified split (`sklearn.train_test_split`, seed=42).\n"
        f"- Per-sample loss masking handles multi-task supervision at training time.\n"
        f"- See `configs/data/soil_*_classes.yaml` in the upstream repo for class mapping.\n"
    )


def push_to_hf(
    train_rows: list[_Row],
    val_rows: list[_Row],
    test_rows: list[_Row],
    *,
    repo_name: str,
    head: str,
    classes: dict[int, str],
    description: str,
    private: bool = True,
) -> dict[str, int]:
    """Push a DatasetDict to a private HF Hub repo. Auto-shards + chunked LFS."""
    from datasets import DatasetDict  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(
        repo_id=repo_name, repo_type="dataset", private=private, exist_ok=True,
    )

    dsd = DatasetDict(
        {
            "train": _rows_to_dataset(train_rows),
            "val": _rows_to_dataset(val_rows),
            "test": _rows_to_dataset(test_rows),
        }
    )
    counts = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
    _LOGGER.info(
        "Pushing %s (train=%d, val=%d, test=%d) ...",
        repo_name, counts["train"], counts["val"], counts["test"],
    )
    dsd.push_to_hub(repo_name, private=private)
    _LOGGER.info("Parquet shards pushed to %s.", repo_name)

    card = _dataset_card(
        repo_id=repo_name, head=head, classes=classes,
        n_train=counts["train"], n_val=counts["val"], n_test=counts["test"],
        description=description,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    _LOGGER.info("README pushed to %s.", repo_name)
    return counts


# --------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------- #


def _job(label: str, fn) -> tuple[list[_Row], list[_Row], list[_Row]]:
    start = time.monotonic()
    _LOGGER.info("===== preparing %s =====", label)
    out = fn()
    elapsed = time.monotonic() - start
    _LOGGER.info("%s prepared in %.1fs (train=%d, val=%d, test=%d)",
                 label, elapsed, len(out[0]), len(out[1]), len(out[2]))
    return out


def _repo_done(repo: str) -> bool:
    """True if a Hub dataset repo already has parquet shards (skip-flag for re-runs)."""
    from huggingface_hub import HfApi  # noqa: PLC0415
    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        return any(f.endswith(".parquet") for f in files)
    except Exception:
        return False


def main() -> int:
    _preflight_auth()

    moist_cfg = _load_yaml(MOISTURE_CLASSES_PATH)
    tex_cfg = _load_yaml(TEXTURE_CLASSES_PATH)
    pf_cfg = _load_yaml(PHANTOMFS_CLASSES_PATH)

    # Skip any repo that already has parquet shards on Hub — keeps re-runs cheap
    # after a partial failure (e.g. sirajganj crash mid-upload).
    if _repo_done(REPO_PHANTOMFS):
        _LOGGER.info("Skip phantomfs — Hub already has parquet shards.")
    else:
        pf_train, pf_val, pf_test = _job("phantomfs", prepare_phantomfs)
        push_to_hf(
            pf_train, pf_val, pf_test,
            repo_name=REPO_PHANTOMFS,
            head="soil_type",
            classes={int(k): v for k, v in pf_cfg["classes"].items()},
            description=(
                "Phantom-fs Indian-deposit soil-type supervision for the IKS "
                "agricultural advisory thesis (Phase 6 soil module)."
            ),
        )

    if _repo_done(REPO_SIRAJGANJ):
        _LOGGER.info("Skip sirajganj — Hub already has parquet shards.")
    else:
        mo_train, mo_val, mo_test = _job("sirajganj_moisture", prepare_sirajganj_moisture)
        push_to_hf(
            mo_train, mo_val, mo_test,
            repo_name=REPO_SIRAJGANJ,
            head="moisture_appearance",
            classes={int(k): v for k, v in moist_cfg["classes"].items()},
            description=(
                "Sirajganj 2025 visual-moisture supervision for the IKS "
                "agricultural advisory thesis (Phase 6 soil module)."
            ),
        )

    if _repo_done(REPO_TEXTURE):
        _LOGGER.info("Skip texture — Hub already has parquet shards.")
    else:
        tx_train, tx_val, tx_test = _job("texture_merged", prepare_texture_merged)
        push_to_hf(
            tx_train, tx_val, tx_test,
            repo_name=REPO_TEXTURE,
            head="texture",
            classes={int(k): v for k, v in tex_cfg["classes"].items()},
            description=(
                "IRSID + VIT latha-soil merged, USDA-collapsed to "
                "{coarse, fine, mixed}. The `source` column ('irsid' / 'vit') "
                "supports per-source ablation."
            ),
        )

    print()
    print("Soil HF Hub dataset push complete (or skipped where Hub already had data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
