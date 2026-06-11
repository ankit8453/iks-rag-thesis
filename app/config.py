"""Phase 10 UI configuration — model repos, switches, dropdown menus.

The disease model is behind a config switch. Today the live demo uses the
ORIGINAL 27-class Phase 5 model (the published-SOTA-frontier 72.3% top-1
checkpoint with clean class names). The Phase 5-R 28-class retrain ships
in parallel as a diagnostic study (Grad-CAM shortcut + background
randomization ablation) — adopting it later is a TWO-LINE change here:

    DISEASE_MODEL_REPO = "ankit-iiitdmj/iks-disease-r-plantdoc"
    HAS_NO_LEAF_CLASS  = True

Nothing else in the app needs touching: the guardrail switches its own
implementation off this flag (native ``no_leaf`` class vs segmentation
fallback) and ``app/loaders.py`` passes the repo through unchanged.

No GPU / network calls happen at import — this is a constants module.
"""

from __future__ import annotations

# --------------------------------------------------------------------- #
# Model repos & switches
# --------------------------------------------------------------------- #

#: Disease classifier HF Hub repo. SEE MODULE DOCSTRING for the swap recipe.
DISEASE_MODEL_REPO: str = "ankit-iiitdmj/iks-disease-plantdoc"

#: True when the disease model has a native ``no_leaf`` reject class
#: (Phase 5-R). False when using the OLD 27-class model — in which case
#: ``app.guardrail.is_leaf`` falls back to ``src.disease.segment``.
HAS_NO_LEAF_CLASS: bool = False

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

APP_TITLE: str = "IKS Agricultural Advisor — Live Demo"

DISCLAIMER: str = (
    "**Research prototype.** This advisor returns recommendations derived "
    "from classical Indian agricultural treatises (Vrikshayurveda, Brihat "
    "Samhita, Krishi Parashara, Upavanavinoda). It is NOT a substitute for "
    "professional agronomic advice. Verify every recommendation with a "
    "qualified agricultural expert before field use."
)

NOT_A_LEAF_MESSAGE: str = (
    "This doesn't look like a clear leaf photo. Please upload a close-up "
    "image of a single leaf (fill ~30-80% of the frame), then try again."
)

OLD_MODEL_HONESTY_NOTE: str = (
    "Disease model: Phase 5 (72.3% top-1). Phase 9 Grad-CAM diagnostics "
    "showed this model can still attend to background regions on some "
    "PlantDoc-style images — interpret the heatmaps with that caveat. The "
    "Phase 5-R retrain ships as a separate diagnostic study."
)


__all__ = [
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
    "LEAF_FOREGROUND_MAX",
    "LEAF_FOREGROUND_MIN",
    "LLM_MODEL_NAME",
    "NOT_A_LEAF_MESSAGE",
    "OLD_MODEL_HONESTY_NOTE",
    "SOIL_MODEL_REPO",
]
