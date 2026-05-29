"""Build the tiled texture dataset (Phase 6 V3-tiling Part 1).

Pipeline:

1. Load the existing private dataset
   ``ankit-iiitdmj/iks-soil-texture-irsid-vit`` from the HF Hub
   (train/val/test = 223/28/28 source images).
2. For each split independently:
   a. Tag every source image with a unique ``source_id`` of the form
      ``f"{split}_{idx:04d}"``. The split prefix guarantees that source
      ids in train, val, and test are pairwise disjoint by construction.
   b. Tile every source image into ``grid × grid`` patches (default
      ``grid=4`` → 16 patches per image). Patches inherit the source's
      label and source_id.
3. Assert source_id disjointness across splits — this is the formal
   leakage guard.
4. Run a resolution sanity check on the source images; if patches at
   the requested grid would fall below 120 px pre-resize, log a loud
   warning recommending ``grid=3`` and refuse to proceed unless the
   ``--force`` flag is passed.
5. Push the tiled patches as a ``DatasetDict{train, val, test}`` to
   ``ankit-iiitdmj/iks-soil-texture-tiled`` (private) using the
   proven ``Dataset.push_to_hub`` streaming pattern (rows store
   ``{image: {bytes, path: None}, label_idx, class_name, source}``
   so HF auto-shards and auto-resumes the LFS upload). See
   ``feedback_hf_dataset_uploads.md`` for the failure history that
   established this pattern.
6. Write ``data/soil/tiled_texture_audit.json`` with the per-split
   patch counts, the per-class breakdown, and the source_id -> split
   mapping for auditability.

Run::

    python scripts/build_tiled_texture_dataset.py
    python scripts/build_tiled_texture_dataset.py --grid 3
    python scripts/build_tiled_texture_dataset.py --force  # ignore reso warning
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.soil.tiling import (  # noqa: E402
    PATCH_MIN_PIXELS,
    build_tiled_split,
    check_patch_resolution,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import DATA_SOIL_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

EXPECTED_HF_USERNAME = "ankit-iiitdmj"
SOURCE_REPO = f"{EXPECTED_HF_USERNAME}/iks-soil-texture-irsid-vit"
TARGET_REPO = f"{EXPECTED_HF_USERNAME}/iks-soil-texture-tiled"
AUDIT_PATH = DATA_SOIL_DIR / "tiled_texture_audit.json"
DEFAULT_GRID = 4
JPEG_QUALITY = 90


# --------------------------------------------------------------------- #
# HF Hub pre-flight + helpers
# --------------------------------------------------------------------- #


def _preflight_auth() -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().whoami()
    actual = info.get("name")
    if actual != EXPECTED_HF_USERNAME:
        raise PermissionError(
            f"HF Hub token belongs to '{actual}', expected "
            f"'{EXPECTED_HF_USERNAME}'. Run `huggingface-cli login` with "
            f"the correct Write token before re-running."
        )
    token = info.get("auth", {}).get("accessToken", {})
    if token.get("role") != "write":
        raise PermissionError(
            f"HF Hub token role is '{token.get('role')}', need 'write'."
        )
    _LOGGER.info("HF Hub pre-flight ok: user=%s role=write", actual)


def _pil_to_jpeg_bytes(img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------- #
# Split build
# --------------------------------------------------------------------- #


def _load_class_names(dataset_dict) -> dict[int, str]:
    """Best-effort idx → class_name map from the source dataset's train split."""
    class_names: dict[int, str] = {}
    for row in dataset_dict["train"]:
        idx = int(row["label_idx"])
        if idx not in class_names:
            class_names[idx] = str(row.get("class_name", str(idx)))
    return class_names


def _collect_source_items(
    hf_split, split_name: str,
) -> list[tuple[str, object, int, str, str]]:
    """Returns ``[(source_id, image, label_idx, class_name, source), ...]``.

    ``source`` is the per-row provenance string ('irsid' / 'vit') already
    on the parquet rows from the V1 prep — we preserve it through the
    tiled rows for ablation.
    """
    out = []
    for idx, row in enumerate(hf_split):
        source_id = f"{split_name}_{idx:04d}"
        out.append(
            (
                source_id,
                row["image"],
                int(row["label_idx"]),
                str(row.get("class_name", str(int(row["label_idx"])))),
                str(row.get("source", "unknown")),
            )
        )
    return out


def _tile_split(
    hf_split, split_name: str, grid: int,
) -> tuple[list[dict], dict[str, str]]:
    """Tile every row in one HF split, return ``(rows, source_id_to_split)``."""
    items = _collect_source_items(hf_split, split_name)
    _LOGGER.info(
        "Tiling split %s: %d source images at grid=%d ...",
        split_name, len(items), grid,
    )

    # Run resolution check on this split's images.
    check_patch_resolution([item[1] for item in items], grid)

    # Build (source_id, image, label) triples for build_tiled_split, but we
    # also need to carry class_name + source — re-emit those alongside.
    rows_out: list[dict] = []
    source_id_to_split: dict[str, str] = {}
    for source_id, img, label_idx, class_name, source in items:
        source_id_to_split[source_id] = split_name
        patches = build_tiled_split([(source_id, img, label_idx)], grid)
        for patch_img, _label, _sid in patches:
            rows_out.append(
                {
                    "image": {"bytes": _pil_to_jpeg_bytes(patch_img), "path": None},
                    "label_idx": int(label_idx),
                    "class_name": class_name,
                    "source": source,
                    "source_id": source_id,
                }
            )
    return rows_out, source_id_to_split


def _assert_split_disjointness(
    per_split_source_ids: dict[str, set[str]],
) -> None:
    """Raise ``RuntimeError`` if any pair of splits shares a source_id."""
    splits = list(per_split_source_ids)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = (
                per_split_source_ids[splits[i]]
                & per_split_source_ids[splits[j]]
            )
            if overlap:
                raise RuntimeError(
                    f"Source-id leakage detected between splits "
                    f"{splits[i]!r} and {splits[j]!r}: {sorted(overlap)[:5]}"
                    f" (total {len(overlap)})."
                )
    _LOGGER.info("Disjointness assertion passed across %d splits.", len(splits))


# --------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------- #


def _rows_to_dataset(rows: list[dict]):
    """Build a ``datasets.Dataset`` with an ``Image()`` column from JPEG bytes."""
    from datasets import Dataset, Image  # noqa: PLC0415

    ds = Dataset.from_list(rows).cast_column("image", Image())
    return ds


def _dataset_card(
    *,
    repo_id: str,
    grid: int,
    counts: dict[str, int],
    classes: dict[int, str],
) -> str:
    total = sum(counts.values())
    if total < 1_000:
        size_cat = "n<1K"
    elif total < 10_000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    class_md = "\n".join(f"- `{name}` (idx {idx})" for idx, name in sorted(classes.items()))
    return (
        f"---\n"
        f"task_categories:\n  - image-classification\n"
        f"size_categories:\n  - {size_cat}\n"
        f"---\n\n"
        f"# {repo_id}\n\n"
        f"Tiled-patch expansion of `ankit-iiitdmj/iks-soil-texture-irsid-vit`.\n"
        f"Each source image is split into a {grid}x{grid} grid of "
        f"non-overlapping patches; every patch is resized to 224x224 and "
        f"keeps the source image's label and a `source_id` for leakage "
        f"audits.\n\n"
        f"## Splits\n\n"
        f"- train: {counts.get('train', 0)}\n"
        f"- val:   {counts.get('val', 0)}\n"
        f"- test:  {counts.get('test', 0)}\n"
        f"- total: {total}\n"
        f"- classes: {len(classes)}\n\n"
        f"## Classes\n\n{class_md}\n\n"
        f"## Schema\n\n"
        f"- `image` — HF `Image()` column (224x224 JPEG patches, lazy PIL decode)\n"
        f"- `label_idx` — int, USDA-collapsed class index (0=coarse, 1=fine, 2=mixed)\n"
        f"- `class_name` — canonical class name string\n"
        f"- `source` — `irsid` / `vit` — source within the merged texture dataset\n"
        f"- `source_id` — unique identifier of the parent source image\n\n"
        f"## Leakage guarantee\n\n"
        f"Source images were assigned to train/val/test by the existing "
        f"upstream split. Patches inherit their parent's split, so source "
        f"images never cross split boundaries. The build script's\n"
        f"disjointness assertion blocks the upload otherwise; see "
        f"`scripts/build_tiled_texture_dataset.py`.\n"
    )


def _push_to_hub(
    rows_per_split: dict[str, list[dict]],
    *,
    grid: int,
    classes: dict[int, str],
) -> dict[str, int]:
    from datasets import DatasetDict  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(
        repo_id=TARGET_REPO, repo_type="dataset", private=True, exist_ok=True,
    )

    dsd = DatasetDict(
        {split: _rows_to_dataset(rows) for split, rows in rows_per_split.items()}
    )
    counts = {split: len(rows) for split, rows in rows_per_split.items()}
    _LOGGER.info(
        "Pushing %s with counts %s ...", TARGET_REPO, counts,
    )
    dsd.push_to_hub(TARGET_REPO, private=True)
    _LOGGER.info("Parquet shards pushed to %s.", TARGET_REPO)

    card = _dataset_card(repo_id=TARGET_REPO, grid=grid, counts=counts, classes=classes)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=TARGET_REPO,
        repo_type="dataset",
    )
    _LOGGER.info("README pushed to %s.", TARGET_REPO)
    return counts


# --------------------------------------------------------------------- #
# Audit JSON
# --------------------------------------------------------------------- #


def _write_audit(
    grid: int,
    counts: dict[str, int],
    per_split_source_ids: dict[str, set[str]],
    rows_per_split: dict[str, list[dict]],
) -> None:
    # Per-class counts per split.
    per_split_class_counts: dict[str, dict[str, int]] = {}
    for split, rows in rows_per_split.items():
        per_class: dict[str, int] = {}
        for row in rows:
            per_class[row["class_name"]] = per_class.get(row["class_name"], 0) + 1
        per_split_class_counts[split] = per_class

    source_id_to_split = {
        sid: split for split, ids in per_split_source_ids.items() for sid in ids
    }

    payload = {
        "source_dataset": SOURCE_REPO,
        "target_dataset": TARGET_REPO,
        "tile_grid": int(grid),
        "patch_output_size": 224,
        "patches_per_source_image": grid * grid,
        "split_counts": {split: int(n) for split, n in counts.items()},
        "split_class_counts": per_split_class_counts,
        "source_ids_per_split": {split: sorted(ids) for split, ids in per_split_source_ids.items()},
        "source_id_to_split": source_id_to_split,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote audit JSON to %s", AUDIT_PATH)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the tiled texture dataset.")
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID,
        help=f"Grid size per side (default {DEFAULT_GRID}; tune to 3 if the resolution guard fires).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore the resolution-guard warning and proceed anyway.",
    )
    args = parser.parse_args(argv)

    _preflight_auth()

    from datasets import load_dataset  # noqa: PLC0415

    _LOGGER.info("Loading source dataset %s ...", SOURCE_REPO)
    dsd = load_dataset(SOURCE_REPO)
    classes = _load_class_names(dsd)

    # Resolution guard — run on TRAIN only, but the train images are
    # representative of the others (same upstream pipeline).
    sample_images = [row["image"] for row in dsd["train"]]
    guard = check_patch_resolution(sample_images, args.grid)
    if guard["warning"] and not args.force:
        banner = "!" * 72
        _LOGGER.error(banner)
        _LOGGER.error(
            "RESOLUTION GUARD: patches at grid=%d would be %dpx (min %dpx).",
            args.grid, guard["patch_min_pre_resize"], PATCH_MIN_PIXELS,
        )
        _LOGGER.error(
            "Recommended grid=%d. Pass --force to override and proceed anyway.",
            guard["recommended_grid"],
        )
        _LOGGER.error(banner)
        return 2

    rows_per_split: dict[str, list[dict]] = {}
    per_split_source_ids: dict[str, set[str]] = {}
    t0 = time.monotonic()
    for split in ("train", "val", "test"):
        rows, source_id_to_split = _tile_split(dsd[split], split, args.grid)
        rows_per_split[split] = rows
        per_split_source_ids[split] = set(source_id_to_split.keys())
        _LOGGER.info(
            "Split %s: %d source images -> %d patches",
            split, len(source_id_to_split), len(rows),
        )

    # Source-id disjointness assertion.
    _assert_split_disjointness(per_split_source_ids)

    counts = _push_to_hub(
        rows_per_split, grid=args.grid, classes=classes,
    )
    elapsed = time.monotonic() - t0
    _write_audit(args.grid, counts, per_split_source_ids, rows_per_split)

    print()
    print(f"Tiled texture dataset built in {elapsed:.1f}s and pushed to {TARGET_REPO}:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[split]} patches from {len(per_split_source_ids[split])} source images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
