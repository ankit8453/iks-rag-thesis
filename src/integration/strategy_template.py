"""Template-based integration strategy (ablation A of contribution C2).

Renders a deterministic plain-text template using the disease + soil
predictions, crop type, and the user's causal context. Used as the
transparent, reproducible baseline against which the LLM-mediated
strategy (B) and the multimodal-embedding strategy (C) are compared.

Per master plan §16 / Phase 8 prompt decision #1, the C5 causal_context
appends an optional clause that conditions retrieval on the suspected
pathway (soil-driven / pest-vector / contagion). When the pathway is
``UNKNOWN`` the clause is omitted so the query is not biased.
"""

from __future__ import annotations

from src.integration.causation import CausalPathway
from src.integration.config import TemplateStrategyConfig
from src.integration.context import MultimodalContext


# Causal-pathway → retrieval-bias clause. Phrasing chosen to match the
# vocabulary that recurs in the IKS corpus (Vrikshayurveda's tree-care
# verses, Krishi Parashara's rainfall + sowing, etc.) rather than modern
# agronomy jargon. ``UNKNOWN`` deliberately yields no clause.
_PATHWAY_CLAUSES: dict[CausalPathway, str] = {
    CausalPathway.SOIL_DRIVEN: (
        "with emphasis on soil restoration, nourishment, and root health"
    ),
    CausalPathway.PEST_VECTOR: (
        "with emphasis on pest deterrence and protection of the plant"
    ),
    CausalPathway.CONTAGION: (
        "with emphasis on preventing the spread of the affliction between plants"
    ),
    CausalPathway.UNKNOWN: "",
}


class TemplateStrategy:
    """Deterministic template-based query builder (Strategy A).

    Parameters
    ----------
    config
        :class:`TemplateStrategyConfig`. Reserved for future tuning
        (``include_confidence`` etc.); the current implementation reads
        only the default fields.
    """

    def __init__(self, config: TemplateStrategyConfig) -> None:
        self.config = config

    def build_query(self, context: MultimodalContext) -> str:
        """Render a deterministic retrieval query from the context.

        Format::

            Organic treatment for <disease> affecting <crop> grown in
            <soil_type> soil that appears <moisture_appearance> with
            <texture> texture[, <causal-clause>].
        """
        disease = _humanise_label(context.disease_pred.class_name)
        crop = context.crop_type.strip() or "crops"
        soil_type = _humanise_label(context.soil_pred.soil_type)
        moisture = context.soil_pred.moisture_appearance
        texture = context.soil_pred.texture

        base = (
            f"Organic treatment for {disease} affecting {crop} grown in "
            f"{soil_type} soil that appears {moisture} with {texture} texture"
        )
        clause = _PATHWAY_CLAUSES.get(context.causal_context.pathway, "")
        if clause:
            return f"{base}, {clause}."
        return f"{base}."


# --------------------------------------------------------------------- #


def _humanise_label(raw: str) -> str:
    """Convert dataset-style labels (e.g. ``"Alluvial_Soil"``,
    ``"Tomato___Leaf_Mold"``) to a query-friendly phrase. Lower-cases,
    drops trailing ``"_Soil"`` / ``"_Leaf"``, replaces underscores
    with spaces, collapses multi-spaces.
    """
    s = raw.replace("___", " ").replace("__", " ").replace("_", " ").strip()
    # Strip a trailing 'Soil' so 'Alluvial Soil soil' doesn't appear in
    # the rendered template (the template already adds 'soil').
    if s.lower().endswith(" soil"):
        s = s[: -len(" soil")].strip()
    return " ".join(s.split()).lower() or raw
