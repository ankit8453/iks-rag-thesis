"""Multimodal context dataclass.

Bundles disease + soil predictions with crop metadata and user-supplied
causal context. This is the object the three integration strategies
consume.

Per master reference §13 / contribution C5: the system does NOT infer
causal pathway from images. ``causal_context`` is user-provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.integration.causation import CausalContext

if TYPE_CHECKING:
    from src.disease.model import DiseasePrediction
    from src.soil.model import SoilPrediction


@dataclass(frozen=True)
class MultimodalContext:
    """Unified per-query state for the integration step.

    Attributes
    ----------
    disease_pred : DiseasePrediction
        Output of :class:`~src.disease.model.DiseaseClassifier`.
    soil_pred : SoilPrediction
        Output of :class:`~src.soil.model.SoilClassifier`. Visual
        attributes only — see guardrail #2.
    crop_type : str
        Farmer-supplied crop name.
    causal_context : CausalContext
        Farmer-supplied causal hypothesis. NOT inferred from images.
    user_notes : str | None
        Free-form additional context (locality, season, intervention
        history, etc.).
    """

    disease_pred: DiseasePrediction
    soil_pred: SoilPrediction
    crop_type: str
    causal_context: CausalContext
    user_notes: str | None = None
