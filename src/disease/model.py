"""EfficientNet-B4 plant disease classifier.

Per master reference §10 (Disease Module) the backbone is fixed to
EfficientNet-B4 via :mod:`timm`. Implementation lands in Phase 5 — this
file currently exposes only the class shape so callers can type-check
against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.disease.config import DiseaseConfig
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import torch  # noqa: F401 — type-only import
    from torch import nn  # noqa: F401

_LOGGER = get_logger(__name__)


@dataclass
class DiseasePrediction:
    """Output schema for a single disease prediction.

    Attributes
    ----------
    class_index : int
        Argmax over the 38 PlantVillage classes.
    class_name : str
        Human-readable class label.
    confidence : float
        Softmax probability of the predicted class, in ``[0, 1]``.
    logits : list[float]
        Raw model logits (length = ``num_classes``). Useful for downstream
        calibration / Grad-CAM.
    """

    class_index: int
    class_name: str
    confidence: float
    logits: list[float]


class DiseaseClassifier:
    """EfficientNet-B4 wrapper for PlantVillage disease classification.

    Parameters
    ----------
    config : DiseaseConfig
        Validated configuration. The backbone is always
        ``efficientnet_b4`` per the locked stack.

    Notes
    -----
    The intended implementation (Phase 5, Week 16) is::

        import timm
        self.model = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )

    Phase 5 also adds:
    - a per-epoch freeze schedule controlled by
      ``config.freeze_backbone_epochs``
    - mixed-precision training
    - integration with :mod:`src.disease.gradcam` for explainability
    """

    def __init__(self, config: DiseaseConfig) -> None:
        self.config = config
        self._model: object | None = None  # Phase 5: torch.nn.Module
        _LOGGER.debug("DiseaseClassifier initialised with config: %s", config)

    def build(self) -> None:
        """Instantiate the timm backbone and replace the classifier head.

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 5 — Week 16 (Disease Module).
        """
        raise NotImplementedError("Phase 5 — Week 16: instantiate timm EfficientNet-B4.")

    def predict(self, image: "object") -> DiseasePrediction:
        """Run inference on a single preprocessed image tensor.

        Parameters
        ----------
        image : torch.Tensor
            Tensor of shape ``(3, H, W)`` already normalised per
            ``AugmentationConfig``.

        Returns
        -------
        DiseasePrediction
            Argmax class, probability, and full logit vector.

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 5 — Week 17 (inference).
        """
        raise NotImplementedError("Phase 5 — Week 17: implement disease inference.")

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable state dict.

        Raises
        ------
        NotImplementedError
            Phase 5 — Week 16.
        """
        raise NotImplementedError("Phase 5 — Week 16: implement state_dict.")
