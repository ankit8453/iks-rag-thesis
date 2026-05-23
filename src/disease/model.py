"""EfficientNet-B4 plant disease classifier (Phase 5 §D).

Per the locked stack: 380×380 input, `timm.create_model("efficientnet_b4",
pretrained=True, num_classes=num_classes)`, custom head with a dropout
layer in front of the classifier linear.

The class subclasses :class:`torch.nn.Module` so it integrates directly
with PyTorch training loops, ``torch.cuda.amp.autocast``, and
``hf_hub_download``-backed checkpoint resume.

This module also keeps the small :class:`DiseasePrediction` dataclass
used by :mod:`src.disease.infer` for the Phase 8 integration handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

_LOGGER = get_logger(__name__)


@dataclass
class DiseasePrediction:
    """Single-image disease prediction (used by :mod:`src.disease.infer`).

    Attributes
    ----------
    class_index : int
        Argmax over the classifier's output. Maps to ``class_map.json``.
    class_name : str
        Human-readable class label.
    confidence : float
        Softmax probability of the predicted class, in ``[0, 1]``.
    logits : list[float]
        Raw model logits (length = ``num_classes``). Useful for
        downstream calibration / Grad-CAM.
    """

    class_index: int
    class_name: str
    confidence: float
    logits: list[float]


def _build_disease_classifier_module(
    num_classes: int,
    pretrained: bool,
    dropout_rate: float,
) -> "nn.Module":
    """Construct the underlying timm + custom-head ``nn.Module``.

    Kept as a free function so the lazy ``timm`` / ``torch`` imports
    don't run at module-import time.
    """
    import timm  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    backbone = timm.create_model(
        "efficientnet_b4",
        pretrained=pretrained,
        num_classes=0,           # detach timm's default classifier head
        global_pool="avg",       # GAP → flat feature vector
    )
    feat_dim: int = backbone.num_features

    head = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(feat_dim, num_classes),
    )

    return nn.Sequential(
        backbone,
        head,
    )


class DiseaseClassifier:
    """EfficientNet-B4 classifier for plant-disease leaves.

    Subclasses dynamically pick up :class:`torch.nn.Module` so the
    class statement itself doesn't import ``torch`` at module-import
    time (keeps the rest of the package import-safe in environments
    without torch installed, e.g. when only doing data validation).

    Parameters
    ----------
    num_classes : int
        Number of output classes. 38 for PlantVillage, 27 for PlantDoc,
        10 for Paddy Doctor — the same instance is fine-tuned across
        stages with the head re-built per dataset.
    pretrained : bool, default True
        Pull ImageNet weights via timm. Always True for the locked
        training plan; the False path exists for unit tests.
    dropout_rate : float, default 0.3
        Dropout probability before the final linear classifier.

    Notes
    -----
    The internal layout is::

        self._backbone        # timm efficientnet_b4 (num_classes=0)
        self._head            # nn.Sequential(Dropout, Linear)
        self._module          # nn.Sequential(backbone, head)

    Storing the wrapper in ``self._module`` rather than mixing it into
    ``self`` directly avoids name collisions with our own attribute
    set. ``__call__`` forwards to ``self._module``.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        from torch import nn  # noqa: PLC0415

        # Make `DiseaseClassifier` itself behave like an nn.Module via
        # delegation. We could subclass nn.Module directly but that
        # forces a torch import at module-import time.
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.dropout_rate = float(dropout_rate)

        self._module: nn.Module = _build_disease_classifier_module(
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            dropout_rate=self.dropout_rate,
        )
        # Convenience aliases — Sequential children in order.
        self._backbone: nn.Module = self._module[0]
        self._head: nn.Module = self._module[1]
        _LOGGER.info(
            "DiseaseClassifier built: efficientnet_b4 num_classes=%d pretrained=%s dropout=%.2f",
            self.num_classes,
            self.pretrained,
            self.dropout_rate,
        )

    # ----- nn.Module-style API ----------------------------------- #

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self._module(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return logits of shape ``(batch, num_classes)``."""
        return self._module(x)

    def parameters(self):
        return self._module.parameters()

    def named_parameters(self):
        return self._module.named_parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._module.load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        self._module = self._module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    # ----- freeze schedule --------------------------------------- #

    def freeze_backbone(self) -> int:
        """Freeze every backbone parameter; head stays trainable.

        Returns
        -------
        int
            Number of trainable parameters remaining (head only).
        """
        for p in self._backbone.parameters():
            p.requires_grad = False
        for p in self._head.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self._module.parameters() if p.requires_grad)
        _LOGGER.info("Backbone frozen — trainable params: %d (head only)", trainable)
        return trainable

    def unfreeze_backbone(self) -> int:
        """Unfreeze every parameter. Returns the new trainable count."""
        for p in self._module.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self._module.parameters() if p.requires_grad)
        _LOGGER.info("Backbone unfrozen — trainable params: %d", trainable)
        return trainable

    # ----- Phase 8 integration helpers --------------------------- #

    def get_feature_extractor(self) -> "nn.Module":
        """Return the backbone (timm model) for downstream feature use.

        Phase 8 integration: the multimodal context module will hook
        into the GAP-pooled feature vector here.
        """
        return self._backbone

    @property
    def head(self) -> "nn.Module":
        """The dropout+linear classification head."""
        return self._head

    # ----- introspection / debugging ----------------------------- #

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self._module.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self._module.parameters())


__all__ = ["DiseaseClassifier", "DiseasePrediction"]
