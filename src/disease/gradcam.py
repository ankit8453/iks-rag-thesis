"""Grad-CAM explanations for disease predictions (Phase 5 §F).

Wraps :mod:`pytorch_grad_cam` for the EfficientNet-B4 disease
classifier. The target layer is the last convolutional block in timm's
EfficientNet — in our `nn.Sequential(backbone, head)` layout that's
``model._backbone.conv_head`` (the final 1×1 expansion before global
average pooling). If a later timm version renames that attribute, the
helper falls back to ``model._backbone.blocks[-1]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401

    from src.disease.model import DiseaseClassifier  # noqa: F401

_LOGGER = get_logger(__name__)


def _resolve_target_layer(model: Any) -> Any:
    """Return the EfficientNet target layer for Grad-CAM.

    EfficientNet's ``conv_head`` is the final 1×1 conv whose output
    feeds the global-average pool; using it as the CAM target gives a
    high-res, class-discriminative heatmap.
    """
    backbone = model.get_feature_extractor() if hasattr(model, "get_feature_extractor") else model
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        return backbone.blocks[-1]
    raise AttributeError(
        "Could not resolve Grad-CAM target layer on the EfficientNet backbone."
    )


def compute_gradcam(
    model: Any,
    image_tensor: "torch.Tensor",
    target_class: int | None = None,
) -> "np.ndarray":
    """Compute the Grad-CAM heatmap for a single image tensor.

    Parameters
    ----------
    model : DiseaseClassifier
        Trained classifier (or any nn.Module-like with
        ``get_feature_extractor()``).
    image_tensor : torch.Tensor
        Either ``(C, H, W)`` or ``(1, C, H, W)``. Must already be
        normalised — the same pipeline the classifier was trained
        on.
    target_class : int, optional
        Which class to explain. If None, uses the argmax prediction.

    Returns
    -------
    numpy.ndarray
        Heatmap of shape ``(H, W)``, values in ``[0, 1]``.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pytorch_grad_cam import GradCAM  # noqa: PLC0415
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: PLC0415

    if image_tensor.dim() == 3:
        input_tensor = image_tensor.unsqueeze(0)
    else:
        input_tensor = image_tensor

    target_layer = _resolve_target_layer(model)
    # GradCAM accepts an nn.Module — our DiseaseClassifier wraps
    # nn.Sequential in self._module. Provide that for forward calls.
    callable_model = model._module if hasattr(model, "_module") else model

    targets: list[Any] | None
    if target_class is not None:
        targets = [ClassifierOutputTarget(int(target_class))]
    else:
        targets = None  # GradCAM uses argmax automatically

    cam = GradCAM(model=callable_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # Shape: (B, H, W) — squeeze to (H, W).
    return np.asarray(grayscale_cam[0], dtype=np.float32)


def overlay_gradcam_on_image(
    image: Any,
    cam: "np.ndarray",
    alpha: float = 0.5,
) -> "np.ndarray":
    """Blend a Grad-CAM heatmap with the source image. Returns ``(H, W, 3)`` uint8.

    Parameters
    ----------
    image : PIL.Image | numpy.ndarray | torch.Tensor
        Source image. Resized to match ``cam.shape``.
    cam : numpy.ndarray
        Heatmap in ``[0, 1]``, shape ``(H, W)``.
    alpha : float
        Heatmap opacity in ``[0, 1]``.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    H, W = cam.shape

    if isinstance(image, np.ndarray):
        rgb = PILImage.fromarray(image).convert("RGB").resize((W, H))
    elif isinstance(image, PILImage.Image):
        rgb = image.convert("RGB").resize((W, H))
    else:
        # Assume torch.Tensor in (C, H, W) or (1, C, H, W). De-normalise
        # back to a viewable image using ImageNet stats.
        try:
            import torch  # noqa: PLC0415

            if isinstance(image, torch.Tensor):
                tensor = image.squeeze(0) if image.dim() == 4 else image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denorm = (tensor.detach().cpu() * std + mean).clamp(0, 1)
                arr = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb = PILImage.fromarray(arr).resize((W, H))
            else:  # pragma: no cover — defensive
                raise TypeError(type(image).__name__)
        except ImportError as exc:  # pragma: no cover
            raise TypeError(f"Cannot handle image type {type(image).__name__}") from exc

    base_arr = np.asarray(rgb, dtype=np.float32) / 255.0

    # Hot colormap without matplotlib.
    heat_norm = np.clip(cam, 0.0, 1.0)
    heat_rgb = np.stack(
        [
            np.clip(heat_norm * 1.5, 0.0, 1.0),                     # R
            np.clip(heat_norm * 1.5 - 0.5, 0.0, 1.0),                # G
            np.clip(heat_norm * 1.5 - 1.0, 0.0, 1.0),                # B
        ],
        axis=-1,
    )

    blended = (1.0 - alpha) * base_arr + alpha * heat_rgb
    return (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)


__all__ = ["compute_gradcam", "overlay_gradcam_on_image"]
