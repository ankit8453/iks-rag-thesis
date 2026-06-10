"""Phase 5-R Part 1 QC sampler — segment + composite ~10 images per disease
dataset, save 3-panel contact sheets per dataset + a background-pool sheet.

Sample only. Does NOT process full datasets. Does NOT train. The point is
for the human reviewer (Ankit + Dr. Pandey) to look at the masks before
the Phase 5-R Part 2 retrain commits to them.

Outputs:

- ``docs/phase5r_qc/plantvillage_qc.png`` — 10 rows × 3 panels
  (original | mask overlay | composited onto random soil bg)
- ``docs/phase5r_qc/paddy_qc.png``        — same shape
- ``docs/phase5r_qc/plantdoc_qc.png``     — same shape (rembg path)
- ``docs/phase5r_qc/background_pool.png`` — 8 random pool samples

Also prints a short honest per-dataset verdict:

- mask quality (clean / acceptable / poor),
- % of samples flagged by the quality guard,
- and whether rembg copes with PlantDoc field images.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.disease.backgrounds import (
    build_background_pool,
    composite_leaf_on_bg,
    pool_size_by_source,
)
from src.disease.segment import SegmentResult, segment
from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

_LOGGER = get_logger(__name__)

QC_OUT = PROJECT_ROOT / "docs" / "phase5r_qc"
QC_OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SEED = 42
N_PER_DATASET = 10
N_BG_GRID = 8

PLANTVILLAGE_ROOT = (
    PROJECT_ROOT / "data" / "plant_disease" / "plantvillage" / "raw"
    / "plantvillage dataset" / "color"
)
PADDY_ROOT = PROJECT_ROOT / "data" / "plant_disease" / "paddy_doctor" / "raw"
PLANTDOC_ROOT = PROJECT_ROOT / "data" / "plant_disease" / "plantdoc" / "raw"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class QCRow:
    dataset: str
    style: str        # "lab" or "field"
    class_label: str
    image_path: Path
    result: SegmentResult
    bg_path: Path | None


def _pick_samples(
    root: Path, n: int, rng: random.Random,
) -> list[tuple[str, Path]]:
    """Pick ~n samples spanning several class folders so the QC sheet
    isn't monolithic. Returns (class_label, file_path)."""
    if not root.is_dir():
        _LOGGER.warning("dataset root missing: %s", root)
        return []
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not class_dirs:
        return []

    # Pick ~n classes (with replacement if there are fewer than n).
    if len(class_dirs) >= n:
        chosen_classes = rng.sample(class_dirs, n)
    else:
        chosen_classes = list(class_dirs)
        while len(chosen_classes) < n:
            chosen_classes.append(rng.choice(class_dirs))

    out: list[tuple[str, Path]] = []
    for cls_dir in chosen_classes:
        # Walk one level: paddy_doctor has train/<class>/<img>; plantvillage
        # has color/<class>/<img>; plantdoc has <class>/<img>.
        files: list[Path] = []
        for ext in _IMG_EXTS:
            files.extend(cls_dir.rglob(f"*{ext}"))
            if len(files) > 32:
                break
        files = [f for f in files if f.is_file()]
        if not files:
            continue
        chosen = rng.choice(files)
        out.append((cls_dir.name, chosen))
    return out


def _resolve_paddy_root() -> Path:
    """Paddy Doctor's raw/ has either a flat class-per-folder layout OR a
    ``train_images/<class>/<img>`` layout depending on how it was pulled.
    Pick whichever is present."""
    candidates = [
        PADDY_ROOT / "train_images",
        PADDY_ROOT,
    ]
    for c in candidates:
        if c.is_dir() and any(p.is_dir() for p in c.iterdir()):
            return c
    return PADDY_ROOT


def _render_qc_sheet(
    rows: list[QCRow],
    bg_paths: list[Path],
    out_path: Path,
    title: str,
    rng: random.Random,
) -> None:
    """3-panel-per-row contact sheet for a dataset."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    n = len(rows)
    if n == 0:
        _LOGGER.warning("No QC rows for %s, skipping sheet.", title)
        return
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    fig.suptitle(title, fontsize=14)

    for i, row in enumerate(rows):
        pil = PILImage.open(row.image_path).convert("RGB")
        # Panel 0: original
        axes[i, 0].imshow(pil)
        axes[i, 0].set_title(
            f"[{i+1}] orig — {row.class_label[:32]}", fontsize=9,
        )
        axes[i, 0].axis("off")

        # Panel 1: mask overlay
        arr = np.asarray(pil)
        mask = np.asarray(row.result.mask)
        if mask.shape != arr.shape[:2]:
            mask_pil = PILImage.fromarray(mask).resize(pil.size, PILImage.Resampling.NEAREST)
            mask = np.asarray(mask_pil)
        overlay = arr.copy()
        # Tint foreground green, leave background untouched
        green = np.zeros_like(arr); green[..., 1] = 255
        alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        axes[i, 1].imshow(overlay)
        flag = "FLAGGED" if row.result.flagged_as_failure else "ok"
        axes[i, 1].set_title(
            f"mask ({row.result.method}, fg={row.result.foreground_fraction:.0%}, {flag})",
            fontsize=9,
        )
        axes[i, 1].axis("off")

        # Panel 2: composite onto random bg
        bg_path = bg_paths[rng.randrange(len(bg_paths))] if bg_paths else None
        if bg_path is not None and not row.result.flagged_as_failure:
            comp = composite_leaf_on_bg(
                pil, row.result.mask, bg_path, out_size=pil.size, rng=rng,
            )
            axes[i, 2].imshow(comp)
            axes[i, 2].set_title(f"composited on {bg_path.parent.name[:24]}", fontsize=9)
        else:
            axes[i, 2].text(
                0.5, 0.5,
                "(skipped — flagged mask)" if row.result.flagged_as_failure
                else "(no bg pool)",
                ha="center", va="center", transform=axes[i, 2].transAxes,
            )
        axes[i, 2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    _LOGGER.info("Saved QC sheet → %s", out_path.relative_to(PROJECT_ROOT))


def _render_bg_pool_sheet(
    pool: list, out_path: Path, n: int, rng: random.Random,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    if not pool:
        _LOGGER.warning("Empty pool, no bg sheet rendered.")
        return
    sample = rng.sample(pool, min(n, len(pool)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(
        f"Background pool sample ({n} of {len(pool)} total) — soil + Pandey backgrounds",
        fontsize=13,
    )
    axes = axes.reshape(rows, cols)
    for i, entry in enumerate(sample):
        ax = axes[i // cols, i % cols]
        ax.imshow(PILImage.open(entry.path).convert("RGB"))
        ax.set_title(f"{entry.source}\n{entry.rel_path[:36]}", fontsize=8)
        ax.axis("off")
    for j in range(len(sample), rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    _LOGGER.info("Saved bg-pool sheet → %s", out_path.relative_to(PROJECT_ROOT))


def _verdict(rows: list[QCRow]) -> tuple[str, float]:
    """Tiny heuristic for an honest verdict label."""
    if not rows:
        return ("no data", 0.0)
    n_flagged = sum(1 for r in rows if r.result.flagged_as_failure)
    flag_pct = n_flagged / len(rows)
    fg = [r.result.foreground_fraction for r in rows
          if not r.result.flagged_as_failure]
    median_fg = sorted(fg)[len(fg) // 2] if fg else 0.0
    if flag_pct == 0 and 0.15 <= median_fg <= 0.75:
        verdict = "clean"
    elif flag_pct <= 0.20:
        verdict = "acceptable"
    else:
        verdict = "poor"
    return (verdict, flag_pct)


def main() -> int:
    rng = random.Random(SAMPLE_SEED)
    print("=" * 70)
    print("Phase 5-R Part 1 QC sampler — segment + composite on a sample")
    print("=" * 70)

    # ---- background pool (cap per source for QC) -------------- #
    pool = build_background_pool(max_per_source=200)
    print()
    print("Background pool sizes per source:")
    for src, n in pool_size_by_source(pool).items():
        print(f"  {src:<24} {n}")
    print(f"  TOTAL                   {len(pool)}")
    if not pool:
        print("ERROR: background pool is empty — check the configured roots.")
        return 1
    bg_paths = [e.path for e in pool]

    # ---- per-dataset sampling --------------------------------- #
    datasets = [
        ("plantvillage", PLANTVILLAGE_ROOT, "lab",   "plantvillage_qc.png"),
        ("paddy",        _resolve_paddy_root(), "lab",   "paddy_qc.png"),
        ("plantdoc",     PLANTDOC_ROOT, "field", "plantdoc_qc.png"),
    ]

    per_ds_rows: dict[str, list[QCRow]] = {}
    for ds_name, ds_root, style, sheet_name in datasets:
        print()
        print(f"--- {ds_name} ({style}) — sampling {N_PER_DATASET} images ---")
        samples = _pick_samples(ds_root, N_PER_DATASET, rng)
        if not samples:
            print(f"  (no samples found under {ds_root})")
            continue
        rows: list[QCRow] = []
        for cls, p in samples:
            try:
                res = segment(p, style)
            except Exception as exc:
                _LOGGER.error("segment failed for %s: %s", p, exc)
                continue
            row = QCRow(
                dataset=ds_name, style=style, class_label=cls,
                image_path=p, result=res, bg_path=None,
            )
            rows.append(row)
            print(
                f"  [{cls[:24]:<24}] fg={res.foreground_fraction:>5.0%} "
                f"flagged={res.flagged_as_failure}  via {res.method}"
            )
        per_ds_rows[ds_name] = rows
        _render_qc_sheet(
            rows=rows, bg_paths=bg_paths,
            out_path=QC_OUT / sheet_name,
            title=f"Phase 5-R QC — {ds_name} ({style}, n={len(rows)})",
            rng=rng,
        )

    # ---- background-pool sheet -------------------------------- #
    _render_bg_pool_sheet(pool, QC_OUT / "background_pool.png", N_BG_GRID, rng)

    # ---- verdict + JSON manifest ------------------------------ #
    print()
    print("=" * 70)
    print("Per-dataset verdicts")
    print("=" * 70)
    manifest: dict[str, Any] = {
        "seed": SAMPLE_SEED,
        "background_pool": {
            "total": len(pool),
            "per_source": pool_size_by_source(pool),
        },
        "datasets": {},
    }
    for ds_name in ("plantvillage", "paddy", "plantdoc"):
        rows = per_ds_rows.get(ds_name, [])
        verdict, flag_pct = _verdict(rows)
        method = rows[0].result.method if rows else "n/a"
        print(
            f"  {ds_name:<14} verdict={verdict:<12} "
            f"flagged={flag_pct:>5.0%}  method={method}  n={len(rows)}"
        )
        manifest["datasets"][ds_name] = {
            "verdict": verdict, "flag_pct": flag_pct,
            "n": len(rows), "method": method,
        }

    print()
    print("Notes:")
    print("  - 'clean'      : zero flagged, median fg fraction inside [15%, 75%].")
    print("  - 'acceptable' : <=20% flagged; usable but worth reviewing the sheet.")
    print("  - 'poor'       : >20% flagged; do NOT proceed to Part 2 without")
    print("                   a SAM fallback for that dataset.")

    (QC_OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print()
    print(f"QC outputs in {QC_OUT.relative_to(PROJECT_ROOT)}:")
    for p in sorted(QC_OUT.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
