"""Leaf-from-background segmentation for the Phase 5-R retrain.

Phase 9's Grad-CAM analysis surfaced that the disease classifier attends
heavily to image corners / backgrounds / watermarks instead of the leaf
tissue. The retrain plan (Phase 5-R) breaks that shortcut by training on
leaves COMPOSITED onto random backgrounds each epoch — the background
stops correlating with the label, so the only signal the model can use
is the leaf itself.

This module is the **segmentation half** of that plumbing — given an
input image, return a binary alpha mask isolating the leaf so the
caller can composite. Two paths, routed by ``style``:

- ``"lab"``  — uniform / studio backgrounds (PlantVillage, Paddy Doctor).
  Classical OpenCV pipeline: HSV green-channel threshold + Otsu binarise,
  then GrabCut refinement seeded from the largest connected component.
  Fast (CPU-only, ~30 ms / image), no model load.
- ``"field"`` — real-world cluttered backgrounds (PlantDoc). U2Net via
  ``rembg``. Slower (~1-2 s / image on CPU) but copes with the noisy
  foliage / soil / fence backgrounds PlantDoc has.

Hard quality guard: if a returned mask covers < 5 % or > 95 % of the
image, :func:`segment` flags the result as ``flagged_as_failure=True``
so the QC pass can quarantine it before any retrain is fed bad masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

SegmentStyle = Literal["lab", "field"]

# Quality guard thresholds. A real leaf usually fills 15-75 % of the
# frame; outside this band the mask is almost certainly wrong (empty,
# whole-image, or a glob of background).
MIN_FOREGROUND_FRAC: float = 0.05
MAX_FOREGROUND_FRAC: float = 0.95

# Mask-edge feathering (used by :mod:`src.disease.backgrounds`).
DEFAULT_FEATHER_PX: int = 5


@dataclass
class SegmentResult:
    """One image's segmentation output.

    Attributes
    ----------
    mask : np.ndarray
        ``(H, W)`` uint8 mask, 0 = background, 255 = leaf foreground.
    foreground_fraction : float
        Share of pixels that are non-zero in ``mask`` — used for the
        quality guard.
    flagged_as_failure : bool
        True when the mask fell outside the
        ``[MIN_FOREGROUND_FRAC, MAX_FOREGROUND_FRAC]`` band; QC
        contact-sheets should highlight these rows.
    method : str
        Which segmenter produced this mask: ``"classical"`` or
        ``"rembg_u2net"`` — recorded for provenance.
    """

    mask: Any  # numpy.ndarray uint8 (H, W)
    foreground_fraction: float
    flagged_as_failure: bool
    method: str


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _to_bgr(image: Any) -> Any:
    """Coerce PIL / numpy input to an OpenCV ``(H, W, 3) uint8 BGR`` array."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, PILImage.Image):
        arr = np.asarray(image.convert("RGB"))
    elif isinstance(image, (str, Path)):
        arr = np.asarray(PILImage.open(str(image)).convert("RGB"))
    else:
        raise TypeError(
            f"Unsupported image type for segmentation: {type(image).__name__}. "
            "Pass a PIL.Image, numpy.ndarray, or filesystem path."
        )
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    # numpy is RGB-order; OpenCV ops expect BGR.
    return arr[:, :, ::-1].copy()


def _wrap_result(mask: Any, method: str) -> SegmentResult:
    """Compute foreground fraction + apply the quality guard."""
    import numpy as np  # noqa: PLC0415

    if mask is None or mask.size == 0:
        return SegmentResult(mask=mask, foreground_fraction=0.0,
                             flagged_as_failure=True, method=method)
    fg = float((mask > 0).mean())
    flagged = fg < MIN_FOREGROUND_FRAC or fg > MAX_FOREGROUND_FRAC
    return SegmentResult(
        mask=mask.astype(np.uint8),
        foreground_fraction=fg,
        flagged_as_failure=flagged,
        method=method,
    )


# --------------------------------------------------------------------- #
# Classical pipeline (lab images)
# --------------------------------------------------------------------- #


def classical_segment(image: Any) -> SegmentResult:
    """Segment a lab-style leaf via HSV + Otsu + GrabCut.

    Designed for the PlantVillage / Paddy Doctor lab regime where the
    background is broadly uniform (grey concrete, wood, white card,
    studio paddy field). Pipeline:

    1. Pull the green channel emphasis from HSV: leaves are usually
       saturated green-ish (hue 25–95 in OpenCV's 0–179 space).
    2. Otsu-binarise the saturation channel as a fallback signal —
       captures non-green diseased patches too.
    3. Combine the two via logical OR, then morphological close + open
       to clean speckle.
    4. Keep the largest connected component (the leaf).
    5. GrabCut refinement seeded by that component's bounding box,
       3 iterations — cheap on a 256–512 px image.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    bgr = _to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv)

    # 1. green-hue mask
    hue_mask = cv2.inRange(h, 25, 95)
    # 2. Otsu on saturation — diseased pixels (brown / yellow) often
    # still saturate above the background.
    _t, sat_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3. combine + clean
    rough = cv2.bitwise_or(hue_mask, sat_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k, iterations=2)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, k, iterations=1)

    # 4. largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rough)
    if n_labels <= 1:
        # Nothing survived — let the quality guard flag it.
        return _wrap_result(rough, method="classical")
    # Index 0 is the background; pick the largest of the rest.
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_largest = int(areas.argmax()) + 1
    largest = (labels == idx_largest).astype(np.uint8) * 255

    # 5. GrabCut refinement seeded by the bounding box of the LCC.
    x, y, w, hgt, _area = stats[idx_largest]
    if w < 8 or hgt < 8:
        return _wrap_result(largest, method="classical")
    mask = np.full(bgr.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    # Mark the LCC interior as probably-foreground, its halo as
    # probably-background — standard GrabCut seeding.
    mask[largest > 0] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            bgr, mask, (x, y, w, hgt), bgd_model, fgd_model,
            iterCount=3, mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        _LOGGER.debug("grabCut failed; falling back to LCC: %s", exc)
        return _wrap_result(largest, method="classical")
    out = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0,
    ).astype(np.uint8)
    return _wrap_result(out, method="classical")


# --------------------------------------------------------------------- #
# rembg pipeline (field images)
# --------------------------------------------------------------------- #


_REMBG_SESSION = None


def _get_rembg_session() -> Any:
    """Lazy-load a U2Net session — first call pulls ~170 MB of weights
    via rembg's cache. Cached on the module afterwards so per-image cost
    is just the forward pass."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session  # noqa: PLC0415

        _LOGGER.info("Loading U2Net session (rembg) ...")
        _REMBG_SESSION = new_session("u2net")
    return _REMBG_SESSION


def rembg_segment(image: Any) -> SegmentResult:
    """Segment a field-style leaf via the rembg U2Net wrapper.

    rembg returns an RGBA image whose alpha channel IS the mask we want.
    We binarise at 128 so the mask is a clean ``{0, 255}`` map for the
    composite step.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415
    from rembg import remove  # noqa: PLC0415

    # rembg accepts PIL or bytes; give it PIL RGB.
    if isinstance(image, (str, Path)):
        pil = PILImage.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil = PILImage.fromarray(image.astype("uint8"))
    else:
        pil = image.convert("RGB") if image.mode != "RGB" else image

    rgba = remove(pil, session=_get_rembg_session())
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return _wrap_result(np.zeros(pil.size[::-1], dtype=np.uint8),
                            method="rembg_u2net")
    alpha = arr[:, :, 3]
    mask = (alpha >= 128).astype(np.uint8) * 255
    return _wrap_result(mask, method="rembg_u2net")


# --------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------- #


def segment(image: Any, style: SegmentStyle) -> SegmentResult:
    """Route to the right segmenter based on dataset style.

    Parameters
    ----------
    image
        PIL.Image, numpy array, or filesystem path.
    style
        ``"lab"`` (PlantVillage / Paddy Doctor) or ``"field"`` (PlantDoc).
    """
    if style == "lab":
        return classical_segment(image)
    if style == "field":
        return rembg_segment(image)
    raise ValueError(
        f"style={style!r} must be 'lab' or 'field'. Add a new branch here "
        "if a future dataset needs a different segmenter."
    )


# --------------------------------------------------------------------- #
# Bounding-box helper
# --------------------------------------------------------------------- #


def find_leaf_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    """Tight bounding box around the non-zero region of ``mask``.

    Returns ``(x, y, w, h)`` or ``None`` if the mask is empty. Used
    downstream by the composite step (Phase 5-R Part 2 / 3) to centre
    the leaf inside the random background before pasting.
    """
    import numpy as np  # noqa: PLC0415

    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0 or not arr.any():
        return None
    ys, xs = np.where(arr > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


__all__ = [
    "DEFAULT_FEATHER_PX",
    "MAX_FOREGROUND_FRAC",
    "MIN_FOREGROUND_FRAC",
    "SegmentResult",
    "SegmentStyle",
    "classical_segment",
    "find_leaf_bbox",
    "rembg_segment",
    "segment",
]
