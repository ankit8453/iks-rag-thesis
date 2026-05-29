"""Phase 6 V3-tiling — patch extraction helpers for texture expansion.

Texture is a local, scale-invariant visual property: a small crop from a
"sandy_loam" soil photo is still sandy_loam. By tiling each source
texture image into a ``grid × grid`` rectangular grid of patches and
training each patch as a separate sample, we expand the effective
training set ~16x (at ``grid=4``) without collecting any new
photographs. The texture head sees ~3,500 patches instead of 223 source
images.

The **critical correctness constraint** is that patches inherit the
source image's train/val/test assignment — never split patches randomly.
A single random-shuffle leak would let patches from the same source
image land in both train and test, producing pixel-level overlap and
fake-high test accuracy. Functions here surface a ``source_id`` for
every patch so callers can assert disjointness across splits, and
``tests/soil/test_tiling.py`` includes a regression test for the
leakage case.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

_LOGGER = get_logger(__name__)


# Patch min-resolution threshold (pre-final-resize). Below this the
# downstream 224x224 resize is mostly upscaling pixels we don't have,
# which is no help to the model.
PATCH_MIN_PIXELS: int = 120

# Final patch size — matches the model's 224x224 input. The training
# pipeline can still augment-crop on top of this; this is the storage
# size in the tiled dataset.
PATCH_OUTPUT_SIZE: int = 224


def _lanczos():
    """Return PIL's LANCZOS resampling constant (compatible across PIL versions)."""
    from PIL import Image  # noqa: PLC0415

    # PIL 10 split sampling constants under ``Image.Resampling``; older
    # versions exposed them on the top-level Image module.
    return getattr(Image, "Resampling", Image).LANCZOS


def tile_image(img: "PILImage", grid: int) -> list["PILImage"]:
    """Split ``img`` into a ``grid × grid`` non-overlapping patch grid.

    Each patch is resized to ``PATCH_OUTPUT_SIZE × PATCH_OUTPUT_SIZE``
    via LANCZOS. The function returns ``grid * grid`` PIL images in
    row-major order (top-left first).

    Parameters
    ----------
    img
        Source image. Must be in a PIL-supported mode; non-RGB inputs
        are converted to RGB.
    grid
        Grid size per side. ``grid=4`` produces 16 patches. Must be
        ``>= 1``.

    Returns
    -------
    list[PIL.Image]
        ``grid * grid`` patches.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1; got {grid}.")

    rgb = img.convert("RGB") if img.mode != "RGB" else img
    w, h = rgb.size
    tile_w = w // grid
    tile_h = h // grid
    if tile_w == 0 or tile_h == 0:
        raise ValueError(
            f"Image too small ({w}x{h}) for grid={grid}: tile size would be 0."
        )

    resampling = _lanczos()
    patches: list[PILImage] = []
    for row in range(grid):
        for col in range(grid):
            left = col * tile_w
            upper = row * tile_h
            right = w if col == grid - 1 else left + tile_w
            lower = h if row == grid - 1 else upper + tile_h
            patch = rgb.crop((left, upper, right, lower))
            patch = patch.resize(
                (PATCH_OUTPUT_SIZE, PATCH_OUTPUT_SIZE),
                resampling,
            )
            patches.append(patch)
    return patches


def check_patch_resolution(
    images: Iterable["PILImage"],
    grid: int,
) -> dict:
    """Resolution sanity-check before tiling.

    Computes the median ``(width, height)`` across the supplied images
    and the resulting pre-resize patch size at the given grid. If the
    smaller patch dimension would be ``< PATCH_MIN_PIXELS``, a warning
    flag is set so the caller can re-think the grid.

    Returns
    -------
    dict
        ``{
            "n_images": int,
            "median_width": int, "median_height": int,
            "median_min_dim": int,
            "patch_width_pre_resize": int,
            "patch_height_pre_resize": int,
            "patch_min_pre_resize": int,
            "min_threshold": int,
            "warning": bool,
            "recommended_grid": int,
        }``
    """
    images = list(images)
    if not images:
        raise ValueError("Cannot check patch resolution on an empty image list.")

    widths = sorted(img.size[0] for img in images)
    heights = sorted(img.size[1] for img in images)
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]
    median_min = min(median_w, median_h)
    patch_w = median_w // grid
    patch_h = median_h // grid
    patch_min = min(patch_w, patch_h)

    warning = bool(patch_min < PATCH_MIN_PIXELS)
    # Recommend grid=3 (or grid-1, whichever larger) when too small.
    recommended_grid = max(1, min(grid - 1, 3)) if warning else grid

    out = {
        "n_images": len(images),
        "median_width": int(median_w),
        "median_height": int(median_h),
        "median_min_dim": int(median_min),
        "patch_width_pre_resize": int(patch_w),
        "patch_height_pre_resize": int(patch_h),
        "patch_min_pre_resize": int(patch_min),
        "min_threshold": int(PATCH_MIN_PIXELS),
        "warning": warning,
        "recommended_grid": int(recommended_grid),
    }
    if warning:
        _LOGGER.warning(
            "Resolution guard: patches at grid=%d would be %dx%d (min %dpx) — "
            "below the %dpx threshold. Recommended grid=%d.",
            grid, patch_w, patch_h, patch_min,
            PATCH_MIN_PIXELS, recommended_grid,
        )
    else:
        _LOGGER.info(
            "Resolution guard ok: median %dx%d -> patches %dx%d (min %dpx) at grid=%d.",
            median_w, median_h, patch_w, patch_h, patch_min, grid,
        )
    return out


def build_tiled_split(
    source_items: list[tuple[str, "PILImage", int]],
    grid: int,
) -> list[tuple["PILImage", int, str]]:
    """Tile every source image, propagating its label and ``source_id``.

    Parameters
    ----------
    source_items
        ``[(source_id, image, label), ...]`` — each source image is
        already tagged with a unique identifier (typically the
        original split + index, e.g. ``"train_0042"``). The caller is
        responsible for source_id uniqueness within and across splits.
    grid
        Grid size — see :func:`tile_image`.

    Returns
    -------
    list[(patch, label, source_id)]
        Length ``len(source_items) * grid * grid``. Patches from the
        same source image share the same ``source_id``, so callers can
        assert disjointness across splits with simple set operations.
    """
    out: list[tuple[PILImage, int, str]] = []
    for source_id, img, label in source_items:
        patches = tile_image(img, grid)
        for patch in patches:
            out.append((patch, int(label), str(source_id)))
    _LOGGER.info(
        "build_tiled_split: %d sources -> %d patches at grid=%d.",
        len(source_items), len(out), grid,
    )
    return out


__all__ = [
    "PATCH_MIN_PIXELS",
    "PATCH_OUTPUT_SIZE",
    "build_tiled_split",
    "check_patch_resolution",
    "tile_image",
]
