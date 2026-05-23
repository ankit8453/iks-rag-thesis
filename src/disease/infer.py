"""Single-image inference for the disease classifier (Phase 5 §F).

Used in Phase 8 multimodal integration + the Streamlit demo. Loads a
trained :class:`~src.disease.model.DiseaseClassifier` from either an
HF Hub model repo or a local checkpoint, runs forward + Grad-CAM in
one go.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.model import DiseaseClassifier, DiseasePrediction
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)


@dataclass
class InferenceResult:
    """Enriched single-image prediction with top-k + optional Grad-CAM.

    Wraps :class:`DiseasePrediction` (kept for backwards compatibility
    with Phase 8 integration) and adds:

    - ``top_k``: list of (class_name, probability) tuples for the top-5.
    - ``gradcam_overlay``: optional H×W×3 numpy uint8 array overlaying
      the CAM heatmap on the input image.
    """

    prediction: DiseasePrediction
    top_k: list[tuple[str, float]]
    gradcam_overlay: Any | None = None  # numpy.ndarray when present


def _load_checkpoint_from_source(model_source: str, work_dir: Path) -> tuple[dict[str, Any], int]:
    """Resolve ``model_source`` to a state-dict + inferred num_classes.

    ``model_source`` may be either an HF Hub model repo ID (e.g.
    ``ankit-iiitdmj/iks-disease-plantdoc``) or a path to a local
    ``checkpoint_*.pt`` file.
    """
    import torch  # noqa: PLC0415

    path = Path(model_source)
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=False)
        _LOGGER.info("Loaded local checkpoint: %s", path)
    else:
        # Assume HF Hub repo ID. Pull checkpoint_best.pt if present,
        # else fall back to checkpoint_latest.pt.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from huggingface_hub.errors import EntryNotFoundError  # noqa: PLC0415

        try:
            local = hf_hub_download(
                repo_id=model_source,
                filename="checkpoint_best.pt",
                repo_type="model",
                cache_dir=str(work_dir / ".hf_cache"),
            )
        except EntryNotFoundError:
            local = hf_hub_download(
                repo_id=model_source,
                filename="checkpoint_latest.pt",
                repo_type="model",
                cache_dir=str(work_dir / ".hf_cache"),
            )
        state = torch.load(local, map_location="cpu", weights_only=False)
        _LOGGER.info("Loaded HF Hub checkpoint: %s", model_source)

    model_state = state.get("model_state", state)
    # Sequential layout: backbone (idx 0) + head (idx 1). The head is
    # Sequential(Dropout, Linear); the Linear weight gives num_classes.
    linear_weight_key = "1.1.weight"
    if linear_weight_key not in model_state:
        # Older / mismatched layout — search for any Linear weight key.
        candidates = [k for k in model_state if k.endswith(".weight") and "fc" in k.lower()]
        if not candidates:
            raise KeyError(
                "Could not infer num_classes from checkpoint — no head "
                "Linear weight key found."
            )
        linear_weight_key = candidates[-1]
    num_classes = int(model_state[linear_weight_key].shape[0])
    return model_state, num_classes


class DiseaseInferenceEngine:
    """Load + run inference + Grad-CAM on a trained disease classifier.

    Parameters
    ----------
    model_source : str
        HF Hub model repo ID (e.g. ``ankit-iiitdmj/iks-disease-plantdoc``)
        or a local checkpoint file path.
    class_names : list[str] | None
        Class labels in index order. If None, integer indices are used
        as labels (``class_0``, ``class_1``, ...).
    device : str
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"`` so single-image
        inference works in the demo without GPU.
    image_size : int
        Resize-and-centre-crop target. Default 380 (EfficientNet-B4
        native).
    """

    def __init__(
        self,
        model_source: str,
        class_names: list[str] | None = None,
        device: str = "cpu",
        image_size: int = 380,
        work_dir: Path | None = None,
    ) -> None:
        import torch  # noqa: PLC0415

        self.device = device
        self.image_size = int(image_size)
        self._work_dir = Path(work_dir) if work_dir is not None else Path(".") / "_inference"
        self._work_dir.mkdir(parents=True, exist_ok=True)

        state, num_classes = _load_checkpoint_from_source(model_source, self._work_dir)
        self.num_classes = num_classes
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but the "
                f"checkpoint head has num_classes={num_classes}."
            )
        self.class_names = list(class_names)

        # Build the model with pretrained=False (we overwrite weights).
        self.model = DiseaseClassifier(
            num_classes=num_classes, pretrained=False, dropout_rate=0.3
        )
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        _LOGGER.info(
            "DiseaseInferenceEngine ready: source=%s device=%s num_classes=%d image_size=%d",
            model_source, device, num_classes, image_size,
        )

    # ------------------------------------------------------------- #

    def _to_tensor(self, image: Any) -> "torch.Tensor":
        """Coerce PIL.Image | np.ndarray | torch.Tensor → ``(1, 3, H, W)`` tensor."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            if isinstance(image, np.ndarray):
                pil_img = PILImage.fromarray(image).convert("RGB")
            elif isinstance(image, PILImage.Image):
                pil_img = image.convert("RGB")
            else:
                raise TypeError(
                    f"Unsupported image type: {type(image).__name__}. "
                    f"Pass PIL.Image, numpy.ndarray, or torch.Tensor."
                )
            arr = np.asarray(
                pil_img.resize(
                    (self.image_size, self.image_size), PILImage.Resampling.BILINEAR
                )
            )
            # Normalise with ImageNet stats; production training used
            # the per-dataset stats but for single-image inference the
            # ImageNet defaults are close enough.
            arr = arr.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device).float()

    # ------------------------------------------------------------- #

    def predict(self, image: Any, with_gradcam: bool = False) -> InferenceResult:
        """Run forward + optional Grad-CAM on a single image."""
        import torch  # noqa: PLC0415
        from torch.nn import functional as F  # noqa: PLC0415

        tensor = self._to_tensor(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        top1_idx = int(probs.argmax().item())
        top1_prob = float(probs[top1_idx].item())
        k = min(5, self.num_classes)
        top_probs, top_idx = probs.topk(k)
        top_k_list = [
            (self.class_names[int(i)], float(p))
            for i, p in zip(top_idx.tolist(), top_probs.tolist(), strict=True)
        ]
        pred = DiseasePrediction(
            class_index=top1_idx,
            class_name=self.class_names[top1_idx],
            confidence=top1_prob,
            logits=[float(x) for x in logits.squeeze(0).tolist()],
        )

        overlay = None
        if with_gradcam:
            from src.disease.gradcam import compute_gradcam, overlay_gradcam_on_image  # noqa: PLC0415

            cam = compute_gradcam(self.model, tensor, target_class=top1_idx)
            overlay = overlay_gradcam_on_image(image, cam, alpha=0.5)

        return InferenceResult(prediction=pred, top_k=top_k_list, gradcam_overlay=overlay)


__all__ = [
    "DiseaseInferenceEngine",
    "InferenceResult",
]
