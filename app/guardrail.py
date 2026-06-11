"""No-leaf guardrail for the Phase 10 UI.

Two implementations, picked by ``HAS_NO_LEAF_CLASS`` in ``app.config``:

* **Native (Phase 5-R, 28 classes)** — the model itself has a ``no_leaf``
  reject class; we trust its prediction. If the top-1 label is
  ``no_leaf``, reject.

* **Fallback (OLD Phase 5, 27 classes)** — best-effort: run
  ``src.disease.segment.segment(image, style="field")`` and reject when
  the leaf foreground fraction is outside ``[LEAF_FOREGROUND_MIN,
  LEAF_FOREGROUND_MAX]``. This catches the common failure modes
  (uploads of soil, sky, full landscape, full close-up of a tomato)
  without false-rejecting normal leaf photos.

Both paths return ``(is_leaf: bool, reason: str)``. ``reason`` is human-
readable; the Streamlit handler shows it to the user when ``is_leaf`` is
False.

This module is import-safe: it touches no GPU, model weight, or HF Hub
at import. The fallback path imports ``src.disease.segment`` lazily so
test stubs can monkey-patch it without paying for ``rembg`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage  # noqa: F401

    from src.disease.infer import DiseaseInferenceEngine  # noqa: F401

_LOGGER = get_logger(__name__)


#: Class name the Phase 5-R model uses for the reject class. Matches the
#: 28th entry the trainer appended; keep in sync.
NO_LEAF_CLASS_NAME: str = "no_leaf"


def is_leaf(
    image: Any,
    *,
    has_no_leaf_class: bool,
    disease_engine: "DiseaseInferenceEngine | None" = None,
    foreground_min: float = 0.08,
    foreground_max: float = 0.92,
    segment_style: str = "field",
) -> tuple[bool, str]:
    """Return ``(is_leaf, reason)`` for an uploaded image.

    Parameters
    ----------
    image
        PIL.Image / numpy / torch tensor / path — anything the underlying
        engine or segmenter accepts.
    has_no_leaf_class
        ``True`` when ``disease_engine`` has a native ``no_leaf`` class
        (Phase 5-R). When ``True``, ``disease_engine`` MUST be supplied
        and is consulted directly. When ``False``, ``disease_engine`` is
        ignored and the segmentation fallback runs.
    disease_engine
        Required when ``has_no_leaf_class`` is True. Pass-through to its
        ``.predict(image)`` method; we read ``prediction.class_name``.
    foreground_min, foreground_max
        Accepted leaf-foreground band for the fallback path. Outside this
        band the upload is rejected. Defaults match
        ``app.config.LEAF_FOREGROUND_{MIN,MAX}``.
    segment_style
        ``"field"`` (rembg/U2Net — for cluttered uploads) or ``"lab"``.
        Forwarded to ``src.disease.segment.segment``.

    Returns
    -------
    (is_leaf, reason)
        ``is_leaf`` is True when the upload passes. ``reason`` is a short
        English explanation; it's empty on the happy path.
    """
    if has_no_leaf_class:
        if disease_engine is None:
            raise ValueError(
                "has_no_leaf_class=True requires a disease_engine to consult."
            )
        result = disease_engine.predict(image)
        class_name = getattr(result.prediction, "class_name", "") or ""
        if class_name.strip().lower() == NO_LEAF_CLASS_NAME:
            _LOGGER.info("Guardrail (native): rejected as %s", class_name)
            return False, (
                f"The disease model classified this upload as "
                f"'{NO_LEAF_CLASS_NAME}'."
            )
        return True, ""

    # ------------------------------------------------------------- #
    # Fallback path — segmentation-based foreground check.
    # ------------------------------------------------------------- #
    from src.disease.segment import segment  # local import

    seg = segment(image, style=segment_style)
    frac = float(seg.foreground_fraction)
    if frac < foreground_min:
        _LOGGER.info(
            "Guardrail (fallback): rejected, foreground=%.2f < %.2f",
            frac, foreground_min,
        )
        return False, (
            f"No clear leaf foreground detected "
            f"(foreground was {frac:.0%} of the image)."
        )
    if frac > foreground_max:
        _LOGGER.info(
            "Guardrail (fallback): rejected, foreground=%.2f > %.2f",
            frac, foreground_max,
        )
        return False, (
            f"The upload looks like a single solid surface, not a leaf "
            f"on a background (foreground was {frac:.0%} of the image)."
        )
    return True, ""


__all__ = ["NO_LEAF_CLASS_NAME", "is_leaf"]
