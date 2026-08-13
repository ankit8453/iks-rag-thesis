"""Unify our disease datasets into a single CROP-AGNOSTIC, disease-type dataset.

Reads any number of source image trees (PlantVillage, PlantDoc, Paddy Doctor,
Dr. Pandey's Brazilian multi-crop set), maps every image to a canonical disease
TYPE (via :mod:`src.disease.disease_type_map`), removes exact duplicates, resizes,
and writes a stratified train/val/test split plus a manifest.

Why derive the type from the DEEPEST disease-naming folder
---------------------------------------------------------
Dataset 1 nests inconsistently — e.g. a correctly-labelled ``.../Café (Coffee) -
Ferrugem (Rust) - 1/1/img.jpg`` sits under class folders, and some images sit
under a mislabelled outer ``... - Cropped`` parent. Walking the path from the
image OUTWARD and taking the first folder that names a real disease picks the
specific (inner) label over a misleading outer one.

Usage (Colab, where the big datasets live)::

    python scripts/build_disease_type_dataset.py \
        --source plantvillage=/content/plantvillage/color \
        --source plantdoc=/content/plantdoc \
        --source paddy=/content/paddy/train_images \
        --source brazil="05-11-2020_256X256_IDEAL_PRACTICAL" \
        --out data/disease_type --size 256 --min-per-class 40

Missing sources are skipped with a warning, so it runs on a laptop with only the
Brazilian set present too.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# allow running as a plain script (python scripts/build_disease_type_dataset.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.disease.disease_type_map import to_disease_type  # noqa: E402

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def disease_type_from_path(path: Path, root: Path) -> str:
    """Map an image to a disease type using its DEEPEST disease-naming folder.

    Returns ``"other"`` if no folder between ``root`` and the image names a
    recognised disease.
    """
    rel_parts = path.relative_to(root).parts[:-1]   # folder segments only
    for seg in reversed(rel_parts):                 # deepest first
        dtype = to_disease_type(seg)
        if dtype != "other":
            return dtype
    # fall back to the joined path (handles healthy "<crop> leaf" style folders)
    return to_disease_type(" ".join(rel_parts))


def stratified_split(
    keys: list[str], ratios: tuple[float, float, float], seed: int,
) -> dict[str, str]:
    """Assign each key to train/val/test deterministically.

    Splitting per-class (the caller groups by class first) keeps every split's
    class distribution the same. A fixed seed makes the split reproducible.
    """
    rng = random.Random(seed)
    order = list(keys)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    out: dict[str, str] = {}
    for i, k in enumerate(order):
        out[k] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def _iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXT and not p.name.startswith("~$"):
            yield p


def build(
    sources: dict[str, Path], out_dir: Path, *, size: int = 256, seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    keep_pest: bool = False, min_per_class: int = 1, crop_leaf: bool = False,
) -> dict:
    """Do the unification. Returns a summary dict (also printed by ``main``).

    ``crop_leaf`` runs the pretrained YOLO :class:`LeafCropper` on every image
    before resize, so the model trains on LEAF crops rather than whole scenes —
    the same fix that made C-PD leaf-attentive. Images where no leaf is detected
    fall back to the full frame (never dropped).
    """
    from PIL import Image  # noqa: PLC0415 - heavy import kept local

    cropper = None
    if crop_leaf:
        from src.disease.leaf_detect import LeafCropper  # noqa: PLC0415
        cropper = LeafCropper()

    # 1) collect (type, source, path) for every image, deduping by content hash
    seen: set[str] = set()
    records: list[tuple[str, str, Path, Path]] = []   # (dtype, source, path, root)
    per_type_raw: Counter = Counter()
    dupes = 0
    for name, root in sources.items():
        if not root.exists():
            print(f"  ! source {name!r} not found at {root} — skipping", file=sys.stderr)
            continue
        for p in _iter_images(root):
            dtype = disease_type_from_path(p, root)
            if dtype == "other" or (dtype == "pest_damage" and not keep_pest):
                continue
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
            except Exception:
                continue
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
            records.append((dtype, name, p, root))
            per_type_raw[dtype] += 1

    # 2) drop under-populated types
    kept_types = {t for t, c in per_type_raw.items() if c >= min_per_class}
    records = [r for r in records if r[0] in kept_types]

    # 3) split per type, then write
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, (dtype, *_rest) in enumerate(records):
        by_type[dtype].append(i)

    split_of: dict[int, str] = {}
    for dtype, idxs in by_type.items():
        assign = stratified_split([str(i) for i in idxs], ratios, seed)
        for i in idxs:
            split_of[i] = assign[str(i)]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    counts: Counter = Counter()
    n_cropped = 0
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "disease_type", "source", "orig_class", "out_path"])
        for i, (dtype, source, path, root) in enumerate(records):
            split = split_of[i]
            dst_dir = out_dir / split / dtype
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{source}_{i:06d}.jpg"
            try:
                im = Image.open(path).convert("RGB")
                if cropper is not None:
                    im, found = cropper.crop(im)   # YOLO leaf crop (full frame if none)
                    n_cropped += int(found)
                im.resize((size, size)).save(dst, quality=90)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed {path}: {exc}", file=sys.stderr)
                continue
            orig = path.relative_to(root).parts[-2] if len(path.relative_to(root).parts) > 1 else ""
            w.writerow([split, dtype, source, orig, str(dst.relative_to(out_dir))])
            counts[(split, dtype)] += 1

    summary = {
        "sources": {k: str(v) for k, v in sources.items()},
        "total_images": len(records), "duplicates_skipped": dupes,
        "leaf_cropped": (n_cropped if crop_leaf else None),
        "types": sorted(kept_types),
        "dropped_small_types": sorted(set(per_type_raw) - kept_types),
        "per_type": {t: per_type_raw[t] for t in sorted(kept_types)},
        "manifest": str(manifest),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=[], metavar="name=path",
                    help="a source tree, e.g. plantdoc=/content/plantdoc (repeatable)")
    ap.add_argument("--out", default="data/disease_type")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=1)
    ap.add_argument("--keep-pest", action="store_true",
                    help="keep insect/mite damage as a class (default: drop)")
    ap.add_argument("--crop-leaf", action="store_true",
                    help="YOLO-crop the leaf from each image before resize (the "
                         "C-PD leaf-focus fix; needs ultralytics + GPU)")
    args = ap.parse_args(argv)

    sources: dict[str, Path] = {}
    for s in args.source:
        if "=" not in s:
            ap.error(f"--source must be name=path, got {s!r}")
        name, path = s.split("=", 1)
        sources[name.strip()] = Path(path.strip())
    if not sources:
        ap.error("give at least one --source name=path")

    summary = build(sources, Path(args.out), size=args.size, seed=args.seed,
                    keep_pest=args.keep_pest, min_per_class=args.min_per_class,
                    crop_leaf=args.crop_leaf)

    print("\n=== disease-type dataset built ===")
    print(f"  total images: {summary['total_images']}  (dupes skipped: {summary['duplicates_skipped']})")
    if summary.get("leaf_cropped") is not None:
        print(f"  leaf-cropped by YOLO: {summary['leaf_cropped']} (rest kept full frame)")
    print(f"  dropped small types: {summary['dropped_small_types'] or 'none'}")
    print("  per type:")
    for t in summary["types"]:
        print(f"     {t:16} {summary['per_type'][t]}")
    print(f"  manifest -> {summary['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
