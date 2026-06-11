"""Grad-CAM helpers for the disease + multi-task soil models (Phase 9).

Locks one shape for the per-prediction visual explanation:

- Disease model (EfficientNet-B4 @ 380×380, single head) — Grad-CAM
  against the predicted class.
- Soil multi-task model (EfficientNet-B0 @ 224×224, three heads:
  ``soil_type`` / ``moisture`` / ``texture``) — Grad-CAM against EACH
  head's predicted class. The multi-output forward
  ``{soil_type, moisture, texture}`` breaks ``pytorch_grad_cam``'s
  ``ClassifierOutputTarget`` (which assumes ``forward`` returns a
  single tensor), so :class:`SoilHeadWrapper` wraps the model + a head
  name and returns ONLY that head's logits.

Both vision models are FULL-PRECISION on disk (only the Llama generator
runs in 4-bit), so gradients flow naturally — no bitsandbytes detangling
needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

    from src.disease.infer import DiseaseInferenceEngine  # noqa: F401
    from src.soil.infer import SoilInferenceEngine  # noqa: F401
    from src.soil.model import SoilMultiTaskClassifier  # noqa: F401

_LOGGER = get_logger(__name__)

# ImageNet stats (same as inference preprocessing). EfficientNet
# B0 and B4 both use these in the Phase 5/6 training code.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Soil head names. Must match the keys ``SoilMultiTaskClassifier.forward``
# returns and the attribute names on the model (``soil_type_head`` etc.).
SOIL_HEADS: tuple[str, ...] = ("soil_type", "moisture", "texture")


# --------------------------------------------------------------------- #
# Target-layer discovery
# --------------------------------------------------------------------- #


def find_target_layer(backbone: Any) -> Any:
    """Locate the Grad-CAM target layer in a timm EfficientNet backbone.

    Prefers the final 1×1 expansion conv (``conv_head``) — its output
    feeds the global-average-pool and gives the highest-resolution
    class-discriminative heatmap. Falls back to the last block when
    ``conv_head`` is absent (rare; mostly seen in custom-pruned
    backbones). Logs a WARNING on fallback so the run is auditable.

    Parameters
    ----------
    backbone
        A timm ``EfficientNet`` (the bare ``num_classes=0`` backbone is
        fine — what matters is that ``.conv_head`` or ``.blocks`` is
        exposed). A custom wrapper (e.g. the ``DiseaseClassifier``'s
        ``nn.Sequential(backbone, head)``) should be unwrapped to its
        backbone before calling.
    """
    if hasattr(backbone, "conv_head"):
        layer = backbone.conv_head
        _LOGGER.info(
            "Grad-CAM target layer: %s.conv_head (%s)",
            type(backbone).__name__, type(layer).__name__,
        )
        return layer

    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        if len(blocks) > 0:
            last_block = blocks[-1]
            _LOGGER.warning(
                "Grad-CAM target layer: %s has no .conv_head; falling back "
                "to .blocks[-1] (%s). Heatmap resolution may be slightly "
                "coarser than usual.",
                type(backbone).__name__, type(last_block).__name__,
            )
            return last_block

    raise AttributeError(
        f"find_target_layer: {type(backbone).__name__} exposes neither "
        ".conv_head nor a .blocks attribute. Pass an unwrapped timm "
        "EfficientNet backbone (num_classes=0) instead of a wrapper."
    )


# --------------------------------------------------------------------- #
# SoilHeadWrapper
# --------------------------------------------------------------------- #


def _import_nn() -> Any:
    """Lazy ``torch.nn`` import so test discovery doesn't need torch
    when only ``find_target_layer`` is being exercised."""
    from torch import nn  # noqa: PLC0415
    return nn


class SoilHeadWrapper:
    """Forward only one head's logits — required for Grad-CAM targeting.

    :class:`pytorch_grad_cam.ClassifierOutputTarget` expects a model
    whose ``forward`` returns a ``(B, C)`` logits tensor for *one*
    classifier. ``SoilMultiTaskClassifier`` returns a
    ``dict[head_name, tensor]`` instead, so trying to Grad-CAM it
    directly is undefined. This wrapper takes the multi-task model + a
    head name and exposes a single-tensor ``forward`` for just that
    head.

    Parameters
    ----------
    soil_model
        :class:`SoilMultiTaskClassifier` — the trained Phase 6 multi-task
        model.
    head
        One of ``"soil_type"`` / ``"moisture"`` / ``"texture"``.
    """

    def __init__(self, soil_model: Any, head: str) -> None:
        if head not in SOIL_HEADS:
            raise ValueError(
                f"head={head!r} not in {SOIL_HEADS}. Soil model has three "
                "heads; pass one of them."
            )
        nn = _import_nn()
        # Build the wrapper as a proper nn.Module so .eval() / .parameters()
        # / device-move behave as pytorch_grad_cam expects.
        self._module = _make_soil_head_module(nn, soil_model, head)
        self.head = head
        self.soil_model = soil_model

    @property
    def module(self) -> Any:
        return self._module

    def __call__(self, x: Any) -> Any:
        return self._module(x)


def _make_soil_head_module(nn_mod: Any, soil_model: Any, head: str) -> Any:
    """Factory so subclassing ``nn.Module`` can happen at call time
    (avoids a hard torch import at module load)."""
    head_attr = f"{head}_head"
    if not hasattr(soil_model, head_attr):
        raise AttributeError(
            f"soil_model is missing attribute {head_attr!r} — does it "
            "have all three of soil_type_head / moisture_head / texture_head?"
        )

    class _SoilHeadOnly(nn_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = soil_model.backbone
            self.head_module = getattr(soil_model, head_attr)
            self._head_name = head

        def forward(self, x: Any) -> Any:
            features = self.backbone(x)
            return self.head_module(features)

    return _SoilHeadOnly()


# --------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------- #


@dataclass
class GradCAMResult:
    """One Grad-CAM run's output.

    Attributes
    ----------
    overlay_rgb : np.ndarray
        Heatmap overlaid on the original image (uint8 H×W×3).
    heatmap : np.ndarray
        Raw normalised CAM (float32 H×W in ``[0, 1]``).
    pred_label : str
        Human-readable predicted label (NOT a ``class_<i>`` placeholder).
    pred_conf : float
        Softmax probability of ``pred_label``.
    pred_index : int
        Raw argmax index, kept for auditing the label-mapping table.
    """

    overlay_rgb: Any  # numpy.ndarray
    heatmap: Any      # numpy.ndarray
    pred_label: str
    pred_conf: float
    pred_index: int


# --------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------- #


def _preprocess_for_gradcam(
    image_path: Path | str | Any, image_size: int,
) -> tuple[Any, Any, Any]:
    """Open an image and return (tensor, rgb_uint8, rgb_float).

    Accepts EITHER a filesystem path (str / Path) OR an already-loaded
    :class:`PIL.Image.Image` so the Phase 5-R Grad-CAM audit can pass
    HF-row PIL images straight in (no temp file round-trip).

    - ``tensor``    : ``(1, 3, H, W)`` ImageNet-normalised, requires grad.
    - ``rgb_uint8`` : H×W×3 uint8 RGB array — what the overlay sits on.
    - ``rgb_float`` : H×W×3 float32 in ``[0, 1]`` — pytorch_grad_cam's
      ``show_cam_on_image`` accepts this format.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(image_path, PILImage.Image):
        pil_img = image_path.convert("RGB")
    else:
        pil_img = PILImage.open(str(image_path)).convert("RGB")
    pil_img = pil_img.resize((image_size, image_size), PILImage.Resampling.BILINEAR)
    rgb_uint8 = np.asarray(pil_img, dtype=np.uint8)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    arr = (rgb_float - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, rgb_uint8, rgb_float


# --------------------------------------------------------------------- #
# Grad-CAM core
# --------------------------------------------------------------------- #


def _run_gradcam(
    forward_module: Any,
    target_layer: Any,
    tensor: Any,
    rgb_float: Any,
    target_class: int,
) -> tuple[Any, Any]:
    """Run pytorch_grad_cam and produce (heatmap, overlay_rgb).

    Caller is responsible for picking the target layer + target class.
    The model is forced to ``eval()`` here so dropout / batchnorm don't
    perturb the heatmap; gradients ARE enabled (no ``torch.no_grad``)
    because Grad-CAM needs them by definition.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pytorch_grad_cam import GradCAM  # noqa: PLC0415
    from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: PLC0415
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: PLC0415

    device = next(forward_module.parameters()).device
    tensor = tensor.to(device).float().requires_grad_(True)
    forward_module.eval()

    cam = GradCAM(model=forward_module, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(int(target_class))]
    # GradCAM expects ``input_tensor`` and ``targets``; output is (B, H, W) float.
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    heatmap = grayscale_cam[0]  # (H, W) float32 in [0, 1]

    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    # ``show_cam_on_image`` returned float64 in [0, 1] in older
    # pytorch_grad_cam, but ≥ 1.5 returns uint8 in [0, 255] directly.
    # The old "always multiply by 255" path saturates the uint8 output
    # to white and the heatmap becomes invisible. Detect and handle
    # both.
    overlay = np.asarray(overlay)
    if overlay.dtype == np.uint8:
        overlay_rgb = overlay
    else:
        overlay_rgb = (overlay.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return heatmap.astype(np.float32), overlay_rgb


# --------------------------------------------------------------------- #
# Public: disease_gradcam
# --------------------------------------------------------------------- #


def disease_gradcam(
    image_path: Path | str | Any,
    engine: "DiseaseInferenceEngine",
) -> GradCAMResult:
    """Run Grad-CAM on the disease classifier for one image.

    Preprocesses the image exactly as :class:`DiseaseInferenceEngine`
    does (380×380, ImageNet stats), runs the model on CPU/CUDA per
    ``engine.device``, picks the predicted class as the Grad-CAM
    target, and returns a :class:`GradCAMResult` with the overlay,
    the raw heatmap, the predicted label (mapped via
    ``engine.class_names``), confidence, and the raw argmax index for
    auditability.

    Parameters
    ----------
    image_path
        Path to a leaf image. Any PIL-readable format is fine.
    engine
        A loaded :class:`DiseaseInferenceEngine`.
    """
    import torch  # noqa: PLC0415

    tensor, _rgb_uint8, rgb_float = _preprocess_for_gradcam(
        image_path, image_size=engine.image_size,
    )

    # Predicted class via the engine's standard predict() so the
    # explanation targets the SAME label the rest of the pipeline saw.
    pil = _open_for_engine(image_path)
    pred = engine.predict(pil).prediction

    backbone = engine.model.get_feature_extractor()
    target_layer = find_target_layer(backbone)
    forward_module = engine.model._module if hasattr(engine.model, "_module") else engine.model  # noqa: SLF001

    heatmap, overlay_rgb = _run_gradcam(
        forward_module=forward_module,
        target_layer=target_layer,
        tensor=tensor,
        rgb_float=rgb_float,
        target_class=int(pred.class_index),
    )

    label = engine.class_names[int(pred.class_index)]
    return GradCAMResult(
        overlay_rgb=overlay_rgb,
        heatmap=heatmap,
        pred_label=label,
        pred_conf=float(pred.confidence),
        pred_index=int(pred.class_index),
    )


# --------------------------------------------------------------------- #
# Public: soil_gradcam
# --------------------------------------------------------------------- #


def soil_gradcam(
    image_path: Path | str,
    soil_engine: "SoilInferenceEngine",
    head: str,
) -> GradCAMResult:
    """Run Grad-CAM on ONE head of the multi-task soil classifier.

    Wraps ``soil_engine.model`` with :class:`SoilHeadWrapper` (forwarding
    only the chosen head's logits) so ``ClassifierOutputTarget`` has a
    single-tensor output to attribute. Preprocesses at the engine's
    configured image size (224 for Phase 6 B0).

    Parameters
    ----------
    image_path
        Path to a soil image.
    soil_engine
        A loaded :class:`SoilInferenceEngine`.
    head
        One of ``"soil_type"`` / ``"moisture"`` / ``"texture"``.

    Notes
    -----
    The per-head predicted label is read from the engine's
    :class:`SoilInferenceResult.prediction` (so the wrapper's argmax
    matches what the rest of the pipeline saw). The model is forced to
    ``eval()`` mode for the duration of the Grad-CAM pass.
    """
    if head not in SOIL_HEADS:
        raise ValueError(f"head={head!r} not in {SOIL_HEADS}.")

    tensor, _rgb_uint8, rgb_float = _preprocess_for_gradcam(
        image_path, image_size=soil_engine.config.image_size,
    )

    # Use the engine to get the per-head prediction (so the explanation
    # targets the same label the rest of the pipeline saw).
    pil = _open_for_engine(image_path)
    inf_result = soil_engine.predict(pil, with_embedding=False)
    pred_attr_map = {
        "soil_type": ("soil_type", soil_engine.soil_type_classes),
        "moisture": ("moisture_appearance", soil_engine.moisture_classes),
        "texture": ("texture", soil_engine.texture_classes),
    }
    pred_attr, classes = pred_attr_map[head]
    label = getattr(inf_result.prediction, pred_attr)
    pred_idx = classes.index(label)
    conf = float(inf_result.prediction.per_head_confidence.get(pred_attr, 0.0))

    # Wrap so forward returns ONE head's logits, then Grad-CAM.
    wrapper = SoilHeadWrapper(soil_engine.model, head=head)
    forward_module = wrapper.module
    forward_module.to(soil_engine.device).eval()
    target_layer = find_target_layer(soil_engine.model.backbone)

    heatmap, overlay_rgb = _run_gradcam(
        forward_module=forward_module,
        target_layer=target_layer,
        tensor=tensor,
        rgb_float=rgb_float,
        target_class=int(pred_idx),
    )
    return GradCAMResult(
        overlay_rgb=overlay_rgb,
        heatmap=heatmap,
        pred_label=label,
        pred_conf=conf,
        pred_index=int(pred_idx),
    )


# --------------------------------------------------------------------- #
# Back-compat: keep ``compute_gradcam`` signature for older callers
# --------------------------------------------------------------------- #


def compute_gradcam(  # noqa: D401
    classifier: Any,
    image_path: Path | str,
    output_path: Path | str,
    target_class: int | None = None,
) -> Path:
    """Back-compat shim: write a PNG of the disease Grad-CAM overlay.

    Kept so the original Phase 5 callers still type-check. New code
    should use :func:`disease_gradcam` directly — it returns the
    overlay as a numpy array (so callers can re-use it without a disk
    round-trip).
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    from src.disease.infer import DiseaseInferenceEngine  # noqa: PLC0415

    # ``classifier`` may be a raw DiseaseClassifier or already an engine.
    if isinstance(classifier, DiseaseInferenceEngine):
        engine = classifier
    else:
        raise TypeError(
            "compute_gradcam back-compat shim now expects a "
            "DiseaseInferenceEngine. Build one and pass it instead "
            "of the bare classifier."
        )
    if target_class is not None:
        _LOGGER.warning(
            "compute_gradcam: target_class=%d ignored — the new "
            "disease_gradcam always uses the model's argmax prediction.",
            target_class,
        )
    result = disease_gradcam(image_path, engine)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(result.overlay_rgb).save(out)
    return out


def _open_for_engine(image_path: Path | str | Any) -> Any:
    """Return a PIL.Image so the engines can run their own preprocessing.

    Accepts EITHER a filesystem path OR an already-loaded
    :class:`PIL.Image.Image` (Phase 5-R audit case)."""
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(image_path, PILImage.Image):
        return image_path.convert("RGB") if image_path.mode != "RGB" else image_path

    return PILImage.open(str(image_path)).convert("RGB")


__all__ = [
    "GradCAMResult",
    "SOIL_HEADS",
    "SoilHeadWrapper",
    "compute_gradcam",
    "disease_gradcam",
    "find_target_layer",
    "soil_gradcam",
]
