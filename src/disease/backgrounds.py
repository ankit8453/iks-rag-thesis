"""Random-background pool + leaf-onto-background compositor (Phase 5-R).

Two halves:

1. :func:`build_background_pool` — walk the configured roots (Phantom-fs
   soil + Sirajganj soil + Dr. Pandey's ``Background_without_leaves``)
   and return a list of background image paths with per-image provenance
   (source name, relative path). This is the pool the training loader
   will sample from every epoch.
2. :func:`composite_leaf_on_bg` — paste a segmented leaf onto a random
   background with small random scale / rotation / position jitter and a
   feathered alpha edge so the seam doesn't betray itself as a vertical
   gradient cue.

Why these specific sources:

- Phantom-fs gives 7 real Indian-deposit soil textures
  (Alluvial / Arid / Black / Laterite / Mountain / Red / Yellow).
- Sirajganj 2025 adds field moisture variants (dry / moderate / wet).
- Dr. Pandey's Background_without_leaves contributes ~1.1k non-leaf
  urban photos — useful as an "anything BUT leaves" signal so the model
  never learns "this kind of background means leaf class X".

Per the Phase 9 finding, the model used image corners + backgrounds as a
shortcut. The retrain (Phase 5-R Part 2) re-randomises that channel
every epoch so the only invariant left in training is the leaf itself.
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

# Image extensions we accept as backgrounds. Skipping .gif and friends
# because they often need RGB conversion + frame-picking.
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

# Default roots — relative to PROJECT_ROOT for our two soil datasets,
# absolute for Dr. Pandey's drop (since it lives outside the repo).
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


@dataclass
class BackgroundEntry:
    """One image in the random-background pool.

    Attributes
    ----------
    path : Path
        Absolute path to the image on disk.
    source : str
        Top-level source identifier — ``"phantomfs"`` /
        ``"sirajganj"`` / ``"pandey_background"`` — so the QC sheet can
        show provenance and Phase 11 can per-source-ablate.
    rel_path : str
        Path relative to the source root, for human-readable logging.
    """

    path: Path
    source: str
    rel_path: str


# --------------------------------------------------------------------- #
# Pool construction
# --------------------------------------------------------------------- #


def build_background_pool(
    roots: tuple[tuple[str, Path], ...] | None = None,
    *,
    max_per_source: int | None = None,
) -> list[BackgroundEntry]:
    """Walk the configured roots and return a flat background-image list.

    Parameters
    ----------
    roots
        Override the default ``(source_name, root_path)`` tuples.
        Defaults to :data:`DEFAULT_BACKGROUND_ROOTS`.
    max_per_source
        If set, cap each source at the first N images (deterministic
        glob order). Useful for the Part 1 QC pass — we don't want to
        enumerate all 5k+ Phantom-fs files just to render an 8-image
        contact sheet.
    """
    roots = roots if roots is not None else DEFAULT_BACKGROUND_ROOTS
    pool: list[BackgroundEntry] = []
    for source_name, root in roots:
        if not root.is_dir():
            _LOGGER.warning(
                "Background source %r not found at %s; skipping.",
                source_name, root,
            )
            continue
        found: list[Path] = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                found.append(p)
            if max_per_source is not None and len(found) >= max_per_source:
                break
        for p in found:
            pool.append(BackgroundEntry(
                path=p, source=source_name,
                rel_path=str(p.relative_to(root)),
            ))
        _LOGGER.info("Background pool: %s contributed %d image(s).",
                     source_name, len(found))
    return pool


def pool_size_by_source(pool: list[BackgroundEntry]) -> dict[str, int]:
    """Per-source histogram of the pool — used by the QC report."""
    out: dict[str, int] = {}
    for entry in pool:
        out[entry.source] = out.get(entry.source, 0) + 1
    return out


# --------------------------------------------------------------------- #
# Composite
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

    Parameters
    ----------
    image
        Foreground RGB image (PIL.Image, numpy array, or path). The leaf
        is everywhere ``mask > 0``; the rest is ignored.
    mask
        Same-shape ``(H, W)`` uint8 mask from :func:`src.disease.segment.segment`.
    background
        Background image (PIL / numpy / path). Resized to match the
        foreground frame.
    feather_px
        Gaussian blur radius used to soften the mask edge before alpha
        blending. Hard edges leak as a vertical-gradient cue.
    scale_range, rotate_range_deg, position_jitter
        Per-sample random jitter ranges. ``position_jitter=0.10`` means
        the leaf centre may move up to ±10 % of the frame from centre.
    out_size
        ``(W, H)``. Defaults to the foreground image's size.
    rng
        Optional :class:`random.Random` so the QC pass can seed for
        reproducibility.

    Returns
    -------
    PIL.Image
        RGB composited image at ``out_size``.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415
    from PIL import ImageFilter  # noqa: PLC0415

    rng = rng if rng is not None else random.Random()

    fg_pil = _to_pil_rgb(image)
    bg_pil = _to_pil_rgb(background)
    mask_pil = _to_pil_L(mask)

    if out_size is None:
        out_size = fg_pil.size  # (W, H)

    # ---- foreground transform ---------------------------------- #
    # Crop fg + mask to the leaf bbox so jitter rotates around the leaf,
    # not the original frame's centre.
    bbox = mask_pil.getbbox()
    if bbox is None:
        # Degenerate mask: return the resized background as a no-op.
        return bg_pil.resize(out_size, PILImage.Resampling.BILINEAR)
    fg_crop = fg_pil.crop(bbox)
    mask_crop = mask_pil.crop(bbox)

    # Random scale: how big the leaf appears against the frame. Scale is
    # relative to the smaller of the output dimensions so the leaf
    # roughly fills the frame.
    base_dim = min(out_size)
    target_dim = int(base_dim * rng.uniform(*scale_range))
    crop_w, crop_h = fg_crop.size
    crop_scale = target_dim / max(crop_w, crop_h)
    new_w = max(8, int(round(crop_w * crop_scale)))
    new_h = max(8, int(round(crop_h * crop_scale)))
    fg_crop = fg_crop.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
    mask_crop = mask_crop.resize((new_w, new_h), PILImage.Resampling.BILINEAR)

    # Random rotation around the leaf centre. expand=True so we don't
    # clip the leaf, then we'll re-crop / centre below.
    angle = rng.uniform(*rotate_range_deg)
    fg_crop = fg_crop.rotate(angle, resample=PILImage.Resampling.BILINEAR, expand=True)
    mask_crop = mask_crop.rotate(angle, resample=PILImage.Resampling.BILINEAR, expand=True)

    # Feather the mask edge so the alpha blend is soft.
    if feather_px > 0:
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # ---- background -------------------------------------------- #
    bg = bg_pil.resize(out_size, PILImage.Resampling.BILINEAR).convert("RGB")
    canvas = bg.copy()

    # ---- random position --------------------------------------- #
    leaf_w, leaf_h = fg_crop.size
    cx = out_size[0] // 2
    cy = out_size[1] // 2
    jitter_x = int(rng.uniform(-position_jitter, position_jitter) * out_size[0])
    jitter_y = int(rng.uniform(-position_jitter, position_jitter) * out_size[1])
    paste_x = cx - leaf_w // 2 + jitter_x
    paste_y = cy - leaf_h // 2 + jitter_y

    canvas.paste(fg_crop, (paste_x, paste_y), mask=mask_crop)
    return canvas


# --------------------------------------------------------------------- #
# Internal coercion helpers
# --------------------------------------------------------------------- #


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
    "build_background_pool",
    "composite_leaf_on_bg",
    "pool_size_by_source",
]
