"""Single-image inference for the Phase 6 multi-task soil classifier.

Single-image counterpart of the batched Phase 6 training/eval pipeline.
Loads a trained ``SoilMultiTaskClassifier`` from either an HF Hub model
repo (e.g. ``ankit-iiitdmj/iks-soil-multitask-v2``) or a local
``.pt`` checkpoint file, runs the three visual heads, and returns the
top-1 label + confidence for each head packed into a
:class:`SoilPrediction`.

Phase 8 (multimodal integration) also needs the penultimate 1280-dim
GAP-pooled backbone feature, so :meth:`SoilInferenceEngine.predict`
optionally returns it alongside the prediction. The Phase 6 training
script supervises three heads only — soil_type (7-class Phantom-fs
Indian deposits), moisture_appearance (3-class Sirajganj 2025
dry/moderate/wet), and texture (3-class IRSID coarse/fine/mixed). Per
master reference §11 / guardrail #2 the soil module is **visual only**
— no NPK, pH, fertility, organic matter %, or other chemical /
quantitative property is ever produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.soil.config import SoilConfig
from src.soil.model import HEAD_NUM_CLASSES, SoilMultiTaskClassifier, SoilPrediction
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

_LOGGER = get_logger(__name__)

# Defaults match the upstream Phantom-fs / Sirajganj / IRSID training
# label orderings used by the Phase 6 v3-tiling checkpoint on HF. If a
# future checkpoint is trained with a different ordering, pass
# ``head_class_names=`` to override.
DEFAULT_SOIL_TYPE_CLASSES: list[str] = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]
DEFAULT_MOISTURE_CLASSES: list[str] = ["dry", "moderate", "wet"]
DEFAULT_TEXTURE_CLASSES: list[str] = ["coarse", "fine", "mixed"]

DEFAULT_HF_REPO: str = "ankit-iiitdmj/iks-soil-multitask-v2"


@dataclass
class SoilInferenceResult:
    """Per-image soil prediction + (optional) backbone embedding.

    ``embedding`` is the 1280-dim GAP-pooled EfficientNet-B0 feature
    vector that feeds the three heads. Phase 8's
    :class:`~src.integration.strategy_multimodal_embedding.MultimodalEmbeddingStrategy`
    consumes it as one half of the visual feature fusion (the other
    half being the Phase 5 disease backbone feature).
    """

    prediction: SoilPrediction
    embedding: Any | None = None  # numpy.ndarray when present


def _load_state_dict(model_source: str, work_dir: Path) -> dict[str, Any]:
    """Resolve ``model_source`` (HF repo id OR local path) to a state-dict."""
    import torch  # noqa: PLC0415

    path = Path(model_source)
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=False)
        _LOGGER.info("Loaded local soil checkpoint: %s", path)
    else:
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
        _LOGGER.info("Loaded HF Hub soil checkpoint: %s", model_source)

    # Training scripts wrap the state dict under "model_state" or
    # "state_dict"; raw torch saves expose the dict directly.
    if isinstance(state, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


class SoilInferenceEngine:
    """Load + run a trained ``SoilMultiTaskClassifier`` for one image.

    Parameters
    ----------
    model_source
        HF Hub model repo id (e.g. ``ankit-iiitdmj/iks-soil-multitask-v2``)
        OR a local ``.pt`` checkpoint file path.
    config
        :class:`SoilConfig`. Used for ``image_size`` (224 locked) and
        backbone identity. If ``None``, the defaults are used.
    head_class_names
        Optional override for the three heads' label orderings.
    device
        ``"cpu"`` or ``"cuda"``. CPU is fine for single-image inference.
    work_dir
        Where to cache HF downloads. Defaults to ``./_inference/``.
    """

    def __init__(
        self,
        model_source: str = DEFAULT_HF_REPO,
        config: SoilConfig | None = None,
        head_class_names: dict[str, list[str]] | None = None,
        device: str = "cpu",
        work_dir: Path | None = None,
    ) -> None:
        self.config = config if config is not None else SoilConfig()
        self.device = device
        self._work_dir = Path(work_dir) if work_dir is not None else Path(".") / "_inference"
        self._work_dir.mkdir(parents=True, exist_ok=True)

        names = dict(head_class_names or {})
        self.soil_type_classes = names.get("soil_type", DEFAULT_SOIL_TYPE_CLASSES)
        self.moisture_classes = names.get("moisture", DEFAULT_MOISTURE_CLASSES)
        self.texture_classes = names.get("texture", DEFAULT_TEXTURE_CLASSES)

        # Sanity-check class-count alignment with the model heads.
        for head, names_list in (
            ("soil_type", self.soil_type_classes),
            ("moisture", self.moisture_classes),
            ("texture", self.texture_classes),
        ):
            expected = HEAD_NUM_CLASSES[head]
            if len(names_list) != expected:
                raise ValueError(
                    f"head_class_names[{head!r}] has {len(names_list)} entries "
                    f"but the SoilMultiTaskClassifier {head} head emits {expected}."
                )

        state = _load_state_dict(model_source, self._work_dir)
        self.model = SoilMultiTaskClassifier(
            backbone_name=self.config.backbone, pretrained=False, dropout=0.3,
        )
        # ``strict=False`` so older checkpoints with stale extra keys still load.
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        _LOGGER.info(
            "SoilInferenceEngine ready: source=%s device=%s image_size=%d",
            model_source, device, self.config.image_size,
        )

    # ------------------------------------------------------------- #

    def _to_tensor(self, image: Any) -> "torch.Tensor":
        """PIL / numpy / tensor → ``(1, 3, H, W)`` ImageNet-normalised tensor."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device).float()

        if isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image type: {type(image).__name__}. "
                f"Pass PIL.Image, numpy.ndarray, or torch.Tensor."
            )
        size = self.config.image_size
        arr = np.asarray(
            pil_img.resize((size, size), PILImage.Resampling.BILINEAR)
        ).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        )

    # ------------------------------------------------------------- #

    def predict(self, image: Any, with_embedding: bool = False) -> SoilInferenceResult:
        """Run forward + return per-head argmax + (optional) backbone embedding."""
        import torch  # noqa: PLC0415
        from torch.nn import functional as F  # noqa: PLC0415

        tensor = self._to_tensor(image)
        with torch.no_grad():
            features = self.model.backbone(tensor)            # (1, 1280)
            outputs = {
                "soil_type": self.model.soil_type_head(features),
                "moisture": self.model.moisture_head(features),
                "texture": self.model.texture_head(features),
            }

        def _argmax_label(head: str, classes: list[str]) -> tuple[str, float]:
            probs = F.softmax(outputs[head], dim=1).squeeze(0)
            idx = int(probs.argmax().item())
            return classes[idx], float(probs[idx].item())

        soil_label, soil_conf = _argmax_label("soil_type", self.soil_type_classes)
        moist_label, moist_conf = _argmax_label("moisture", self.moisture_classes)
        tex_label, tex_conf = _argmax_label("texture", self.texture_classes)

        pred = SoilPrediction(
            soil_type=soil_label,
            moisture_appearance=moist_label,
            texture=tex_label,
            per_head_confidence={
                "soil_type": soil_conf,
                "moisture_appearance": moist_conf,
                "texture": tex_conf,
            },
        )
        embedding = None
        if with_embedding:
            embedding = features.squeeze(0).cpu().numpy()
        return SoilInferenceResult(prediction=pred, embedding=embedding)


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    config: SoilConfig,
) -> SoilPrediction:
    """Convenience wrapper: load engine, predict one image, return prediction.

    Equivalent to::

        engine = SoilInferenceEngine(str(checkpoint_path), config=config)
        engine.predict(Image.open(image_path)).prediction

    Phase 8 uses :class:`SoilInferenceEngine` directly so the backbone
    embedding can be captured alongside the prediction.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    engine = SoilInferenceEngine(str(checkpoint_path), config=config)
    image = PILImage.open(image_path)
    return engine.predict(image).prediction


__all__ = [
    "DEFAULT_HF_REPO",
    "DEFAULT_MOISTURE_CLASSES",
    "DEFAULT_SOIL_TYPE_CLASSES",
    "DEFAULT_TEXTURE_CLASSES",
    "SoilInferenceEngine",
    "SoilInferenceResult",
    "predict_image",
]
