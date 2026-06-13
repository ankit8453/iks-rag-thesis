"""Pretrained YOLO leaf detector → crop, for the C-PD inference pipeline.

The C-PD disease model (EXPERIMENT_LOG §6c) was trained on leaf CROPS, so
at inference we must crop the leaf first, then classify. This wraps the
pretrained YOLOv8 leaf detector
``foduucom/plant-leaf-detection-and-classification`` (no training needed)
and returns the highest-confidence leaf crop, or the full image if
nothing is detected.

Shared by Phase 9 (explainability) and Phase 10 (UI) so there is ONE
crop-first code path. The detector model loads lazily on first
``.crop()`` so importing this module is cheap and offline-safe.
"""

from __future__ import annotations

from typing import Any

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

#: Pretrained YOLOv8 leaf detector (HF Hub). mAP@0.5 ≈ 0.946. Inference-only.
DEFAULT_YOLO_REPO: str = "foduucom/plant-leaf-detection-and-classification"

#: Detection confidence floor. Below this we treat the image as "no leaf
#: found" and fall back to the full image rather than crop noise.
DEFAULT_CONF: float = 0.25


class LeafCropper:
    """Detect the leaf with a pretrained YOLO and crop to it.

    Parameters
    ----------
    repo
        HF Hub repo holding the YOLO ``.pt`` weights.
    conf
        Minimum detection confidence to accept a box.
    pad_frac
        Fraction of the box width/height to pad on each side before
        cropping (a little margin so a tight box doesn't clip the leaf
        edge). Clamped to image bounds.

    Notes
    -----
    The model loads on the FIRST :meth:`crop` call (lazy). If the
    detector or ``ultralytics`` is unavailable, :meth:`crop` degrades
    gracefully to returning the full image with ``found=False`` so the
    pipeline never hard-crashes on a detection failure.
    """

    def __init__(
        self,
        repo: str = DEFAULT_YOLO_REPO,
        conf: float = DEFAULT_CONF,
        pad_frac: float = 0.05,
    ) -> None:
        self.repo = repo
        self.conf = float(conf)
        self.pad_frac = float(pad_frac)
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        from huggingface_hub import hf_hub_download, list_repo_files
        from ultralytics import YOLO

        weights = [f for f in list_repo_files(self.repo) if f.endswith(".pt")]
        if not weights:
            raise RuntimeError(f"No .pt weights found in {self.repo!r}.")
        name = "best.pt" if "best.pt" in weights else weights[0]
        path = hf_hub_download(self.repo, name)
        self._model = YOLO(path)
        _LOGGER.info("LeafCropper: loaded YOLO weights %s from %s", name, self.repo)
        return self._model

    def detect_box(self, image: Any) -> tuple[int, int, int, int] | None:
        """Return the highest-confidence leaf box ``(x1,y1,x2,y2)`` or None."""
        try:
            model = self._ensure_model()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("LeafCropper: detector unavailable (%s); no crop.", exc)
            return None
        res = model.predict(image, verbose=False, conf=self.conf)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        confs = res.boxes.conf.tolist()
        best = max(range(len(confs)), key=lambda i: confs[i])
        x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[best].tolist())
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def crop(self, image: Any) -> tuple[Any, bool]:
        """Return ``(cropped_pil, found)``.

        ``found=False`` (and the full image) when no leaf is detected or
        the detector is unavailable — so callers always get a usable image.
        """
        pil = image.convert("RGB") if image.mode != "RGB" else image
        box = self.detect_box(pil)
        if box is None:
            return pil, False
        w, h = pil.size
        x1, y1, x2, y2 = box
        px = int(round((x2 - x1) * self.pad_frac))
        py = int(round((y2 - y1) * self.pad_frac))
        left = max(0, x1 - px)
        top = max(0, y1 - py)
        right = min(w, x2 + px)
        bottom = min(h, y2 + py)
        if right <= left or bottom <= top:
            return pil, False
        return pil.crop((left, top, right, bottom)), True


__all__ = ["DEFAULT_CONF", "DEFAULT_YOLO_REPO", "LeafCropper"]
