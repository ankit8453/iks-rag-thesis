"""Multi-task soil classifier.

VISUAL ONLY. The soil module outputs ``soil_type``, ``texture``,
``surface``, ``moisture_appearance``, and ``cover_state``. It does
**not** predict NPK, pH, fertility, organic matter %, or any other
chemical composition (per master reference §11 / guardrail #2).

Backbone is EfficientNet-B0 because the soil dataset (~1,300 images) is
much smaller than PlantVillage and a deeper backbone (B4) would overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.soil.config import SoilConfig
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class SoilPrediction:
    """Multi-task prediction output.

    Notes
    -----
    Only visual attributes appear here. The presence of any of
    ``npk``, ``ph``, ``fertility``, ``organic_matter`` or
    ``chemical_composition`` in this dataclass should fail code review.
    """

    soil_type: str
    texture: str
    surface: str
    moisture_appearance: str
    cover_state: str
    per_head_confidence: dict[str, float] = field(default_factory=dict)


class SoilClassifier:
    """EfficientNet-B0 with one classification head per visual attribute.

    Parameters
    ----------
    config : SoilConfig
        Validated soil configuration. The disallowed-outputs list is
        re-checked in :meth:`build` as a belt-and-braces measure.

    Notes
    -----
    Phase 6 implementation (Week 19) plan:

    1. ``self.backbone = timm.create_model("efficientnet_b0",
       pretrained=True, num_classes=0, global_pool="avg")``
    2. Per head in ``config.multi_task_heads``, attach
       ``nn.Linear(self.backbone.num_features, head_num_classes[head])``.
    3. Forward returns ``dict[head_name, logits]``.

    The multi-task loss is a sum of per-head cross-entropy weighted by
    ``head_weight`` (defaulting to uniform). See ``src/soil/train.py``.

    Per guardrail #4 the evaluation pipeline must also support a
    region-held-out split — that is wired in :mod:`src.soil.dataset`, not
    here.
    """

    def __init__(self, config: SoilConfig) -> None:
        self.config = config
        self._model: object | None = None  # Phase 6: torch.nn.Module
        _LOGGER.debug("SoilClassifier initialised (heads=%s)", config.multi_task_heads)

    def build(self) -> None:
        """Instantiate backbone + per-head linear classifiers.

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 6 — Week 19.
        """
        # Re-assert the disallowed-output constraint; cheap insurance.
        if any(h in self.config.disallowed_outputs for h in self.config.multi_task_heads):
            raise ValueError(
                "Configuration error: a disallowed chemical attribute appears "
                "as a head. The soil module is visual-only."
            )
        raise NotImplementedError("Phase 6 — Week 19: instantiate timm EfficientNet-B0.")

    def predict(self, image: object) -> SoilPrediction:
        """Run multi-task inference on a single image tensor.

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 6 — Week 19.
        """
        raise NotImplementedError("Phase 6 — Week 19: implement multi-task inference.")
