"""Multimodal context dataclass + Phase 8 builder.

Bundles disease + soil predictions with crop metadata, user-supplied
causal context, and (optionally) the penultimate visual feature
embeddings used by Strategy C (the embedding-projection ablation).

Per master reference §13 / contribution C5: the system does NOT infer
causal pathway from images — ``causal_context`` is user-supplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.integration.causation import CausalContext, CausalPathway

# Disease-engine fallback labels look like ``class_0`` / ``class_12`` —
# they are NEVER acceptable in downstream query construction. The
# assertion guard below catches any leak so we never silently retrieve
# against an index-only query again.
_INDEX_LABEL_RE = re.compile(r"^class_\d+$", re.IGNORECASE)

if TYPE_CHECKING:
    from src.disease.model import DiseasePrediction
    from src.soil.model import SoilPrediction


@dataclass(frozen=True)
class MultimodalContext:
    """Unified per-query state for the Phase 8 integration step.

    Attributes
    ----------
    disease_pred : DiseasePrediction
        Output of :class:`~src.disease.model.DiseaseClassifier`.
    soil_pred : SoilPrediction
        Output of :class:`~src.soil.model.SoilMultiTaskClassifier`. Visual
        attributes only — see guardrail #2.
    crop_type : str
        Farmer-supplied crop name (e.g. ``"rice"``, ``"mango"``).
    causal_context : CausalContext
        Farmer-supplied causal hypothesis. NOT inferred from images.
    user_notes : str | None
        Free-form additional context (locality, season, intervention
        history, etc.).
    disease_emb : np.ndarray | None
        Optional penultimate B4 feature vector (1792-dim) captured at
        inference. Required only by Strategy C
        (:class:`~src.integration.strategy_multimodal_embedding.MultimodalEmbeddingStrategy`);
        Strategies A and B do not look at it.
    soil_emb : np.ndarray | None
        Optional penultimate B0 feature vector (1280-dim), same role as
        ``disease_emb``.
    """

    disease_pred: "DiseasePrediction"
    soil_pred: "SoilPrediction"
    crop_type: str
    causal_context: CausalContext
    user_notes: str | None = None
    disease_emb: Any | None = field(default=None)  # np.ndarray when present
    soil_emb: Any | None = field(default=None)


# --------------------------------------------------------------------- #
# Phase 8 helper: leaf+soil image → MultimodalContext
# --------------------------------------------------------------------- #


def build_multimodal_context(
    leaf_image: Any,
    soil_image: Any,
    crop_type: str,
    causal_pathway: CausalPathway | str = CausalPathway.UNKNOWN,
    causal_notes: str | None = None,
    *,
    disease_engine: Any | None = None,
    soil_engine: Any | None = None,
    disease_model_source: str = "ankit-iiitdmj/iks-disease-plantdoc",
    soil_model_source: str = "ankit-iiitdmj/iks-soil-multitask-v2",
    device: str = "cpu",
    work_dir: Path | None = None,
    capture_embeddings: bool = True,
) -> MultimodalContext:
    """Orchestrate Phase 5 disease + Phase 6 soil inference into one context.

    Parameters
    ----------
    leaf_image, soil_image
        PIL / numpy / torch tensor images.
    crop_type
        Farmer-supplied crop name. Free-form string.
    causal_pathway
        :class:`CausalPathway` or its ``.value`` (``"soil_driven"`` /
        ``"pest_vector"`` / ``"contagion"`` / ``"unknown"``). Default
        ``UNKNOWN`` so the C5 hook is OPTIONAL — matches Phase 8 prompt
        decision #2 (default "unspecified").
    causal_notes
        Optional free-text farmer narrative on the suspected cause.
    disease_engine, soil_engine
        Pre-built inference engines. If ``None``, each is constructed on
        demand from ``*_model_source``. Notebooks usually build the
        engines once at the top and pass them in to avoid reloading
        weights per call.
    disease_model_source, soil_model_source
        HF Hub repo IDs or local checkpoint paths. Ignored if the
        corresponding engine is provided.
    device
        ``"cpu"`` / ``"cuda"`` for both engines if they're built here.
    work_dir
        Cache dir for HF downloads if the engines are built here.
    capture_embeddings
        If True (default), also captures the penultimate visual
        embeddings needed by Strategy C. Cheap — same forward pass.

    Returns
    -------
    MultimodalContext
        Populated context, including ``disease_emb`` and ``soil_emb``
        when ``capture_embeddings`` is True.
    """
    # ---- normalise causal pathway --------------------------------- #
    if isinstance(causal_pathway, str):
        try:
            pathway = CausalPathway(causal_pathway)
        except ValueError as exc:
            valid = [p.value for p in CausalPathway]
            raise ValueError(
                f"causal_pathway={causal_pathway!r} not recognised; "
                f"expected one of {valid}."
            ) from exc
    else:
        pathway = causal_pathway
    causal = CausalContext(pathway=pathway, notes=causal_notes)

    # ---- disease inference ---------------------------------------- #
    if disease_engine is None:
        from src.disease.infer import DiseaseInferenceEngine  # noqa: PLC0415
        disease_engine = DiseaseInferenceEngine(
            model_source=disease_model_source, device=device, work_dir=work_dir,
        )
    if capture_embeddings:
        disease_result, disease_emb = disease_engine.predict_with_embedding(leaf_image)
    else:
        disease_result = disease_engine.predict(leaf_image)
        disease_emb = None
    disease_pred = disease_result.prediction

    # ---- soil inference ------------------------------------------- #
    if soil_engine is None:
        from src.soil.infer import SoilInferenceEngine  # noqa: PLC0415
        soil_engine = SoilInferenceEngine(
            model_source=soil_model_source, device=device, work_dir=work_dir,
        )
    soil_result = soil_engine.predict(soil_image, with_embedding=capture_embeddings)
    soil_pred = soil_result.prediction
    soil_emb = soil_result.embedding

    # Guard against the Bug-1 regression: if the disease engine fell
    # back to ``class_<i>`` placeholders (because no class_map.json
    # could be resolved), the rendered Strategy-A / Strategy-B query
    # would carry "class 0" instead of a real disease name and the
    # whole multimodal comparison would be uninterpretable. Refuse to
    # return a poisoned context — fail loudly here, the notebook fixes
    # it by passing ``class_names=`` explicitly to the engine instead.
    if _INDEX_LABEL_RE.match(disease_pred.class_name or ""):
        raise ValueError(
            f"Disease engine emitted placeholder label "
            f"{disease_pred.class_name!r} instead of a real class name. "
            "This means the engine could not resolve a class_map.json. "
            "Pass class_names=[...] when constructing DiseaseInferenceEngine, "
            "or ensure the relevant data/splits/<dataset>/class_map.json "
            "is present in the repo so build_visual_context can auto-load it."
        )

    return MultimodalContext(
        disease_pred=disease_pred,
        soil_pred=soil_pred,
        crop_type=crop_type,
        causal_context=causal,
        disease_emb=disease_emb,
        soil_emb=soil_emb,
    )
