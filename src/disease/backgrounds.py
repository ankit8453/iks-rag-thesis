"""Random-background pool + leaf-onto-background compositor (Phase 5-R).

Two halves:

1. :func:`build_background_pool` — return a list of background image
   paths with per-image provenance.

   On the **laptop** the local soil raw/ trees are present, so we walk
   ``data/soil/phantomfs/raw/`` and ``data/soil/sirajganj_moisture/raw/``
   directly. On **Colab** those trees do NOT exist (they're gitignored),
   so the function falls back to the published HF datasets
   ``ankit-iiitdmj/iks-soil-phantomfs`` and
   ``ankit-iiitdmj/iks-soil-sirajganj-moisture`` — pulls a capped
   number of rows, writes them as ``.jpg`` under a local scratch dir,
   and returns those paths. The HF fallback is idempotent (a second
   call hits the on-disk scratch dir, no network).

2. :func:`composite_leaf_on_bg` — paste a segmented leaf onto a random
   background with small random scale / rotation / position jitter and a
   feathered alpha edge.

Background sources used at training time (per Phase 5-R Part 1 verdict):

- Phantom-fs gives 7 real Indian-deposit soil textures.
- Sirajganj 2025 adds field moisture variants (dry / moderate / wet).
- Dr. Pandey's ``Background_without_leaves`` folder is the ONLY piece
  of his dataset we use (the rest is a confirmed PlantVillage re-pack
  per ``docs/pandey_dataset_inspection.md``). It lives on the laptop
  only; on Colab the trainer just gets a slightly smaller pool from
  the two soil sources, which is still fine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

# Image extensions we accept as backgrounds.
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

# --------------------------------------------------------------------- #
# Local-first sources + HF fallback table
# --------------------------------------------------------------------- #

DEFAULT_BACKGROUND_ROOTS: tuple[tuple[str, Path], ...] = (
    ("phantomfs", PROJECT_ROOT / "data" / "soil" / "phantomfs" / "raw"),
    ("sirajganj", PROJECT_ROOT / "data" / "soil" / "sirajganj_moisture" / "raw"),
    (
        "pandey_background",
        Path(
            r"C:\Users\HP\Downloads\Plant_leaf_diseases_dataset"
            r"\Plant_leave_diseases_dataset_with_augmentation"
            r"\Background_without_leaves"
        ),
    ),
)

# Per-source HF fallback: when the local root is missing, pull this
# many rows from the HF dataset and cache them under
# ``PROJECT_ROOT/data/_bg_cache/<source>/``. ``None`` here ⇒ no HF
# fallback (laptop-only source — applies to Pandey).
HF_FALLBACK_TABLE: dict[str, dict[str, Any] | None] = {
    "phantomfs": {
        "dataset_id": "ankit-iiitdmj/iks-soil-phantomfs",
        "split": "train",
        "default_max": 500,
    },
    "sirajganj": {
        "dataset_id": "ankit-iiitdmj/iks-soil-sirajganj-moisture",
        "split": "train",
        "default_max": 500,
    },
    "pandey_background": None,
}

# Where HF-fallback rows get cached as JPEGs on disk.
HF_BG_CACHE_ROOT: Path = PROJECT_ROOT / "data" / "_bg_cache"


@dataclass
class BackgroundEntry:
    """One image in the random-background pool.

    ``source`` is one of the keys in :data:`DEFAULT_BACKGROUND_ROOTS`
    (``"phantomfs"`` / ``"sirajganj"`` / ``"pandey_background"``); the
    QC sheet and per-source ablations key off it.
    """

    path: Path
    source: str
    rel_path: str


# --------------------------------------------------------------------- #
# Pool construction
# --------------------------------------------------------------------- #


def _scan_dir(root: Path, max_n: int | None) -> list[Path]:
    found: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            found.append(p)
        if max_n is not None and len(found) >= max_n:
            break
    return found


def _populate_hf_fallback(
    source: str,
    hf_info: dict[str, Any],
    max_n: int,
) -> Path:
    """Download up to ``max_n`` rows from an HF dataset and cache them
    as JPEGs under :data:`HF_BG_CACHE_ROOT` / ``<source>/``. Returns the
    cache directory. Idempotent — already-cached rows are reused."""
    cache_dir = HF_BG_CACHE_ROOT / source
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(cache_dir.glob("*.jpg"))
    if len(existing) >= max_n:
        _LOGGER.info(
            "HF-fallback cache for %s already has %d/%d images at %s; reusing.",
            source, len(existing), max_n,
            cache_dir.relative_to(PROJECT_ROOT),
        )
        return cache_dir

    _LOGGER.info(
        "Populating HF-fallback cache for %s: pulling up to %d rows from %s split=%s ...",
        source, max_n, hf_info["dataset_id"], hf_info["split"],
    )
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(hf_info["dataset_id"], split=hf_info["split"])
    written = len(existing)
    for i, row in enumerate(ds):
        if written >= max_n:
            break
        out = cache_dir / f"hf_{i:06d}.jpg"
        if out.is_file() and out.stat().st_size > 0:
            continue
        row["image"].convert("RGB").save(out, format="JPEG", quality=88)
        written += 1
    _LOGGER.info(
        "HF-fallback cache for %s ready: %d images at %s",
        source, written, cache_dir.relative_to(PROJECT_ROOT),
    )
    return cache_dir


def build_background_pool(
    roots: tuple[tuple[str, Path], ...] | None = None,
    *,
    max_per_source: int | None = None,
    use_hf_fallback: bool = True,
    hf_max_per_source: int | None = None,
) -> list[BackgroundEntry]:
    """Walk the configured roots and return a flat background-image list.

    Parameters
    ----------
    roots
        Override the default ``(source_name, root_path)`` tuples.
    max_per_source
        Cap per source for the in-memory pool. Defaults to no cap (all
        images visible to the trainer).
    use_hf_fallback
        When ``True`` (default), if a configured local root is missing
        AND an entry exists in :data:`HF_FALLBACK_TABLE`, populate the
        local scratch cache from HF and use that.
    hf_max_per_source
        How many rows to pull from each HF fallback. Defaults to the
        ``default_max`` in :data:`HF_FALLBACK_TABLE`.
    """
    roots = roots if roots is not None else DEFAULT_BACKGROUND_ROOTS
    pool: list[BackgroundEntry] = []
    for source_name, root in roots:
        active_root = root
        if not root.is_dir() and use_hf_fallback:
            hf_info = HF_FALLBACK_TABLE.get(source_name)
            if hf_info is not None:
                max_n = (
                    hf_max_per_source
                    if hf_max_per_source is not None
                    else int(hf_info["default_max"])
                )
                try:
                    active_root = _populate_hf_fallback(
                        source_name, hf_info, max_n=max_n,
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "HF fallback for %s failed: %s; skipping source.",
                        source_name, exc,
                    )
                    continue
            else:
                _LOGGER.warning(
                    "Background source %r not found at %s and no HF "
                    "fallback configured; skipping.",
                    source_name, root,
                )
                continue
        elif not root.is_dir():
            _LOGGER.warning(
                "Background source %r not found at %s; skipping.",
                source_name, root,
            )
            continue

        found = _scan_dir(active_root, max_per_source)
        for p in found:
            pool.append(BackgroundEntry(
                path=p, source=source_name,
                rel_path=str(p.relative_to(active_root)),
            ))
        _LOGGER.info(
            "Background pool: %s contributed %d image(s) from %s.",
            source_name, len(found),
            active_root if active_root != root else "local",
        )
    return pool


def pool_size_by_source(pool: list[BackgroundEntry]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in pool:
        out[entry.source] = out.get(entry.source, 0) + 1
    return out


# --------------------------------------------------------------------- #
# Composite — unchanged from Part 1
# --------------------------------------------------------------------- #


def composite_leaf_on_bg(
    image: Any,
    mask: Any,
    background: Any,
    *,
    feather_px: int = 5,
    scale_range: tuple[float, float] = (0.7, 1.1),
    rotate_range_deg: tuple[float, float] = (-15.0, 15.0),
    position_jitter: float = 0.10,
    out_size: tuple[int, int] | None = None,
    rng: random.Random | None = None,
) -> Any:
    """Paste a segmented leaf onto a background with small jitter + feathering.

    See module docstring for the rationale. Implementation unchanged
    since Phase 5-R Part 1.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415
    from PIL import ImageFilter  # noqa: PLC0415

    rng = rng if rng is not None else random.Random()

    fg_pil = _to_pil_rgb(image)
    bg_pil = _to_pil_rgb(background)
    mask_pil = _to_pil_L(mask)

    if out_size is None:
        out_size = fg_pil.size

    bbox = mask_pil.getbbox()
    if bbox is None:
        return bg_pil.resize(out_size, PILImage.Resampling.BILINEAR)
    fg_crop = fg_pil.crop(bbox)
    mask_crop = mask_pil.crop(bbox)

    base_dim = min(out_size)
    target_dim = int(base_dim * rng.uniform(*scale_range))
    crop_w, crop_h = fg_crop.size
    crop_scale = target_dim / max(crop_w, crop_h)
    new_w = max(8, int(round(crop_w * crop_scale)))
    new_h = max(8, int(round(crop_h * crop_scale)))
    fg_crop = fg_crop.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
    mask_crop = mask_crop.resize((new_w, new_h), PILImage.Resampling.BILINEAR)

    angle = rng.uniform(*rotate_range_deg)
    fg_crop = fg_crop.rotate(angle, resample=PILImage.Resampling.BILINEAR, expand=True)
    mask_crop = mask_crop.rotate(angle, resample=PILImage.Resampling.BILINEAR, expand=True)

    if feather_px > 0:
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=feather_px))

    bg = bg_pil.resize(out_size, PILImage.Resampling.BILINEAR).convert("RGB")
    canvas = bg.copy()

    leaf_w, leaf_h = fg_crop.size
    cx = out_size[0] // 2
    cy = out_size[1] // 2
    jitter_x = int(rng.uniform(-position_jitter, position_jitter) * out_size[0])
    jitter_y = int(rng.uniform(-position_jitter, position_jitter) * out_size[1])
    paste_x = cx - leaf_w // 2 + jitter_x
    paste_y = cy - leaf_h // 2 + jitter_y

    canvas.paste(fg_crop, (paste_x, paste_y), mask=mask_crop)
    return canvas


def _to_pil_rgb(image: Any) -> Any:
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(image, (str, Path)):
        return PILImage.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        return PILImage.fromarray(image.astype("uint8")).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    raise TypeError(f"Unsupported background image type: {type(image).__name__}.")


def _to_pil_L(mask: Any) -> Any:
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(mask, PILImage.Image):
        return mask.convert("L")
    arr = np.asarray(mask, dtype="uint8")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return PILImage.fromarray(arr, mode="L")


__all__ = [
    "BackgroundEntry",
    "DEFAULT_BACKGROUND_ROOTS",
    "HF_BG_CACHE_ROOT",
    "HF_FALLBACK_TABLE",
    "build_background_pool",
    "composite_leaf_on_bg",
    "pool_size_by_source",
]
