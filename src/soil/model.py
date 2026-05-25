"""Multi-task soil classifier (Phase 6 §D, B0 edition).

VISUAL ONLY. The soil module outputs ``soil_type``,
``moisture_appearance``, and ``texture``. It does **not** predict NPK,
pH, fertility, organic matter %, or any other chemical composition
(per master reference §11 / guardrail #2).

Architecture per master plan §22 (LOCKED): EfficientNet-B0 at 224x224,
1280-dim pooled features feed three independent classification heads.

- ``soil_type``  : 7 classes (Phantom-fs Indian deposits)
- ``moisture``   : 3 classes (Sirajganj 2025 wet / moist / dry)
- ``texture``    : 3 classes (IRSID + VIT, USDA-collapsed)

Total parameters: ~5.3M (~4.0M backbone + ~17K across the three heads).
Well-matched to ~2,600 combined training images; B4 would overfit.

Post-Phase-4 reconciliation (supervisor sign-off received): the
``surface`` and ``cover_state`` heads from the original Week-2 design
were dropped during the soil-parameter coverage audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
from torch import nn

from src.soil.config import SoilConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


# Locked per master plan §22. ``SoilMultiTaskClassifier`` defaults to
# this and ``src/soil/config.py`` enforces it at the type level.
DEFAULT_BACKBONE: str = "efficientnet_b0"

# Per-head class counts. These are the source of truth for the head
# linear-layer output dimensions; the dataset configs in
# ``configs/data/soil_*_classes.yaml`` must agree.
HEAD_NUM_CLASSES: dict[str, int] = {
    "soil_type": 7,
    "moisture": 3,
    "texture": 3,
}


@dataclass
class SoilPrediction:
    """Multi-task prediction output (legacy, kept for backward compat).

    The Phase 6 training code returns a dict of per-head logits rather
    than this dataclass — this is still here because
    :mod:`src.integration.context` and :mod:`src.soil.infer` reference
    the type. Future Phase 8 work will harmonise the two.

    Notes
    -----
    Only the three visual heads appear here. The presence of any of
    ``npk``, ``ph``, ``fertility``, ``organic_matter`` or
    ``chemical_composition`` in this dataclass should fail code review.
    """

    soil_type: str
    moisture_appearance: str
    texture: str
    per_head_confidence: dict[str, float] = field(default_factory=dict)


class SoilClassifier:
    """Phase-4 stub kept for backward compatibility — VISUAL ONLY.

    The real Phase 6 model is :class:`SoilMultiTaskClassifier` below.
    ``tests/soil/test_smoke.py`` still references this class to confirm
    the placeholder-doc invariants (especially that the soil module is
    visual-only — never NPK, pH, fertility, organic-matter %, or other
    chemical / quantitative properties; see guardrail #2 / §11).
    New code should use :class:`SoilMultiTaskClassifier`.
    """

    def __init__(self, config: SoilConfig) -> None:
        self.config = config
        _LOGGER.debug(
            "SoilClassifier (legacy stub) — heads=%s; use SoilMultiTaskClassifier "
            "for Phase 6 training.",
            config.multi_task_heads,
        )

    def build(self) -> None:
        if any(h in self.config.disallowed_outputs for h in self.config.multi_task_heads):
            raise ValueError(
                "Configuration error: a disallowed chemical attribute appears "
                "as a head. The soil module is visual-only."
            )
        raise NotImplementedError(
            "Legacy stub: use SoilMultiTaskClassifier from src.soil.model "
            "for Phase 6 multi-task training."
        )

    def predict(self, image: object) -> SoilPrediction:
        raise NotImplementedError(
            "Legacy stub: use SoilMultiTaskClassifier from src.soil.model "
            "and call its forward() directly."
        )


class SoilMultiTaskClassifier(nn.Module):
    """EfficientNet-B0 backbone + 3 dropout-then-linear heads.

    Parameters
    ----------
    backbone_name : str
        timm backbone identifier. **Default and locked value is
        ``"efficientnet_b0"``** per master plan §22 and
        ``src/soil/config.py``.
    pretrained : bool, default True
        Pull ImageNet weights via timm.
    dropout : float, default 0.3
        Dropout probability inside each head, applied to the GAP-pooled
        feature vector before the linear classifier.

    Notes
    -----
    Forward returns a dict::

        {"soil_type": (B, 7), "moisture": (B, 3), "texture": (B, 3)}

    keys match :data:`HEAD_NUM_CLASSES`. Loss masking with
    ``ignore_index=-1`` (see :func:`src.soil.train.compute_multitask_loss`)
    handles heterogeneous supervision — each batch sample has a real
    label for exactly one head and ``-1`` for the others.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.dropout = float(dropout)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=self.pretrained,
            num_classes=0,           # detach timm's classifier head
            global_pool="avg",       # GAP -> flat feature vector
        )
        feat_dim: int = int(self.backbone.num_features)

        self.soil_type_head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, HEAD_NUM_CLASSES["soil_type"]),
        )
        self.moisture_head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, HEAD_NUM_CLASSES["moisture"]),
        )
        self.texture_head = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, HEAD_NUM_CLASSES["texture"]),
        )

        _LOGGER.info(
            "SoilMultiTaskClassifier built: %s heads=[soil_type=%d, moisture=%d, "
            "texture=%d] dropout=%.2f",
            backbone_name,
            HEAD_NUM_CLASSES["soil_type"],
            HEAD_NUM_CLASSES["moisture"],
            HEAD_NUM_CLASSES["texture"],
            self.dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)  # (B, 1280) for B0
        return {
            "soil_type": self.soil_type_head(features),
            "moisture": self.moisture_head(features),
            "texture": self.texture_head(features),
        }

    # ----- freeze schedule --------------------------------------- #

    def freeze_backbone(self) -> int:
        """Freeze the timm backbone; the three heads stay trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for head in (self.soil_type_head, self.moisture_head, self.texture_head):
            for p in head.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _LOGGER.info("Backbone frozen — trainable params: %d (heads only)", trainable)
        return trainable

    def unfreeze_backbone(self) -> int:
        """Unfreeze every parameter."""
        for p in self.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _LOGGER.info("Backbone unfrozen — trainable params: %d", trainable)
        return trainable

    # ----- introspection ----------------------------------------- #

    def head_param_count(self) -> int:
        return sum(
            p.numel()
            for head in (self.soil_type_head, self.moisture_head, self.texture_head)
            for p in head.parameters()
        )

    def backbone_param_count(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())


__all__ = [
    "DEFAULT_BACKBONE",
    "HEAD_NUM_CLASSES",
    "SoilClassifier",
    "SoilMultiTaskClassifier",
    "SoilPrediction",
]
