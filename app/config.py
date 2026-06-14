"""Phase 10 UI configuration — model repos, switches, dropdown menus.

The disease model is the **C-PD** retrain (`iks-disease-plantdoc-crop`):
trained on leaf CROPS so it attends to the LEAF, not the background
(Phase 5/9). The UI pipeline is therefore CROP-FIRST: a pretrained YOLO
leaf detector crops the leaf before the classifier sees it.

Two design rules from Phase 9, applied here:
  - **heatmap only for disease** — healthy leaves show no Grad-CAM.
  - **advisory only for disease** — healthy leaves skip the IKS query
    (a "treatment for a healthy leaf" query is meaningless).

No GPU / network calls happen at import — this is a constants module.
"""

from __future__ import annotations

# --------------------------------------------------------------------- #
# Model repos & switches
# --------------------------------------------------------------------- #

#: Disease classifier HF Hub repo — the C-PD (leaf-attention) retrain.
DISEASE_MODEL_REPO: str = "ankit-iiitdmj/iks-disease-plantdoc-crop"

#: C-PD is the 27-class PlantDoc model (no native ``no_leaf`` class), so the
#: guardrail uses the segmentation fallback in ``app.guardrail.is_leaf``.
HAS_NO_LEAF_CLASS: bool = False

#: Pretrained YOLO leaf detector (no training) for the crop-first pipeline.
YOLO_LEAF_REPO: str = "foduucom/plant-leaf-detection-and-classification"

#: Detection confidence floor — 0.10 reliably detects on stock/field leaves.
YOLO_CONF: float = 0.10

#: The 10 HEALTHY PlantDoc classes (the other 17 are diseases). Drives the
#: "heatmap/advisory only for disease" gating.
HEALTHY_CLASSES: frozenset[str] = frozenset({
    "Apple leaf", "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf",
    "Peach leaf", "Raspberry leaf", "Soyabean leaf", "Strawberry leaf",
    "Tomato leaf", "grape leaf",
})

#: Soil multi-task classifier HF Hub repo.
SOIL_MODEL_REPO: str = "ankit-iiitdmj/iks-soil-multitask-v2"

#: Llama generator. Llama-3.1-8B 4-bit is the Phase 7 default; if VRAM
#: overflows on the T4 (16 GB) we fall back to Llama-3.2-3B by editing
#: this one line — RAGPipeline.model_name forwards straight to it.
LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"

#: HF dataset id for the IKS chunk corpus (206 chunks across 4 books).
CORPUS_REPO: str = "ankit-iiitdmj/iks-corpus-chunks"


# --------------------------------------------------------------------- #
# Dropdowns
# --------------------------------------------------------------------- #

#: Crop dropdown. Free-form text accepted via "Other" — Strategy B's
#: rewrite handles unknown crops gracefully. The curated list mirrors
#: the crops the test datasets actually cover.
CROP_CHOICES: tuple[str, ...] = (
    "rice",
    "wheat",
    "maize",
    "tomato",
    "potato",
    "apple",
    "grape",
    "mango",
    "soybean",
    "cotton",
    "sugarcane",
    "other",
)

#: Causal-pathway dropdown. Values match
#: ``src.integration.causation.CausalPathway`` exactly so the choice
#: flows through to ``MultimodalContext`` without translation.
#:
#: ``unknown`` is the default — per master plan §13 / contribution C5,
#: the system does NOT infer cause from images; the user supplies it.
CAUSAL_CHOICES: tuple[tuple[str, str], ...] = (
    ("unknown", "Not sure / not specified"),
    ("soil_driven", "Soil-driven (root / soil health)"),
    ("pest_vector", "Pest vector (insects / animals)"),
    ("contagion", "Contagion (spreading from neighbouring plants)"),
)


# --------------------------------------------------------------------- #
# Retrieval + generation defaults
# --------------------------------------------------------------------- #

#: Default Phase 8 query construction strategy. Phase 8 winner: Strategy
#: B (LLM-mediated, 0.59-0.96 retrieval score) over A (template, 0.01-
#: 0.04). Sidebar lets the user toggle to A for ablation comparison.
DEFAULT_STRATEGY: str = "B"  # "A" | "B"

#: Number of chunks retrieved + handed to the generator. Matches Phase 7
#: tuning — past 5 the LLM dilutes citations.
DEFAULT_TOP_K: int = 5

#: Generation cap. The §17 grounded-advisor prompt is short; 512 covers a
#: step-by-step protocol with citations. The Streamlit error handler
#: retries at 256 on OOM.
DEFAULT_MAX_NEW_TOKENS: int = 512

#: Decoding temperature. Strategy B query rewrite uses 0.2 (deterministic);
#: advisor generation matches.
DEFAULT_TEMPERATURE: float = 0.2


# --------------------------------------------------------------------- #
# Guardrail tuning
# --------------------------------------------------------------------- #

#: Acceptable leaf-foreground range when ``HAS_NO_LEAF_CLASS`` is False.
#: Outside [8%, 92%] the segmentation almost certainly failed and the
#: upload is rejected as "doesn't look like a leaf photo". Wider than
#: ``src.disease.segment``'s [5%, 95%] QC band to be USER-friendly: only
#: REALLY obvious non-leaf uploads get blocked.
LEAF_FOREGROUND_MIN: float = 0.08
LEAF_FOREGROUND_MAX: float = 0.92

#: Segmentation style for the fallback guardrail. PlantDoc-style cluttered
#: uploads need rembg/U2Net, not the lab-classical pipeline.
GUARDRAIL_SEGMENT_STYLE: str = "field"


# --------------------------------------------------------------------- #
# UI strings
# --------------------------------------------------------------------- #

APP_TITLE: str = "VṚKṢA · IKS Agricultural Advisor"

APP_TAGLINE: str = "Leaf + soil vision → grounded advice from classical Indian agronomy"

DISCLAIMER: str = (
    "**Research prototype.** Recommendations are derived from classical Indian "
    "agricultural treatises (Vrikshayurveda, Brihat Samhita, Krishi Parashara, "
    "Upavanavinoda) — NOT a substitute for professional agronomic advice. "
    "Verify before field use."
)

NOT_A_LEAF_MESSAGE: str = (
    "This doesn't look like a clear leaf photo. Please upload a close-up "
    "image of a single leaf (fill ~30-80% of the frame), then try again."
)

MODEL_NOTE: str = (
    "Disease model: C-PD (crop-retrained, leaf-attention). Pipeline crops the "
    "leaf, classifies, and shows a Grad-CAM heatmap only when a disease is "
    "detected. Healthy leaves get no heatmap and no treatment query."
)


def is_healthy(class_name: str) -> bool:
    """True if the predicted disease class is actually a HEALTHY leaf class."""
    return class_name in HEALTHY_CLASSES


#: Normalisations for crop names derived from a disease label's first token.
_CROP_ALIASES: dict[str, str] = {"soyabean": "soybean", "bell": "bell pepper"}


def crop_from_disease(class_name: str) -> str:
    """Derive the crop from a PlantDoc class name (the crop is the first token).

    The disease label already names the crop — ``"Corn rust leaf"`` → ``"corn"``,
    ``"Apple Scab Leaf"`` → ``"apple"``, ``"Bell_pepper leaf spot"`` →
    ``"bell pepper"``. The UI uses this instead of the (possibly stale) crop
    dropdown so the retrieval query can't say "rice" for a corn leaf.
    """
    first = class_name.replace("_", " ").split()[0].lower()
    return _CROP_ALIASES.get(first, first)


__all__ = [
    "APP_TAGLINE",
    "APP_TITLE",
    "CAUSAL_CHOICES",
    "CORPUS_REPO",
    "CROP_CHOICES",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_STRATEGY",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DISCLAIMER",
    "DISEASE_MODEL_REPO",
    "GUARDRAIL_SEGMENT_STYLE",
    "HAS_NO_LEAF_CLASS",
    "HEALTHY_CLASSES",
    "LEAF_FOREGROUND_MAX",
    "LEAF_FOREGROUND_MIN",
    "LLM_MODEL_NAME",
    "MODEL_NOTE",
    "NOT_A_LEAF_MESSAGE",
    "SOIL_MODEL_REPO",
    "YOLO_CONF",
    "YOLO_LEAF_REPO",
    "is_healthy",
]
