"""Phase 10 — full-system Streamlit UI.

Single entry point wiring Phases 5 (disease) + 6 (soil) + 7 (RAG) +
8 (integration) + 9 (explainability) into one interactive demo.

Run locally::

    streamlit run app/streamlit_app.py

Run on Colab via the launcher notebook ``notebooks/phase10_launch_ui.ipynb``;
it boots streamlit on a port and exposes it via localtunnel.
"""

from __future__ import annotations

# Streamlit prepends THIS file's directory (app/) to sys.path, which
# shadows the top-level ``app`` package and makes ``from app import
# config`` fail with ModuleNotFoundError. Inject the repo root BEFORE
# any ``from app import ...`` line so the package resolves correctly.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import io
import traceback
from typing import Any

import streamlit as st
from PIL import Image

from app import config as app_config
from app.guardrail import is_leaf
from app.loaders import load_all, report_vram
from src.integration.causation import CausalContext, CausalPathway
from src.integration.config import (
    LLMMediatedStrategyConfig,
    TemplateStrategyConfig,
)
from src.integration.context import MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.integration.strategy_template import TemplateStrategy
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


# --------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------- #

st.set_page_config(
    page_title=app_config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(app_config.APP_TITLE)
st.caption(app_config.OLD_MODEL_HONESTY_NOTE)


# --------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------- #

with st.sidebar:
    st.header("Inputs")
    crop = st.selectbox(
        "Crop", options=list(app_config.CROP_CHOICES), index=0,
    )
    if crop == "other":
        crop = st.text_input("Custom crop name", value="rice").strip() or "rice"

    causal_labels = [label for _, label in app_config.CAUSAL_CHOICES]
    causal_values = [value for value, _ in app_config.CAUSAL_CHOICES]
    causal_idx = st.selectbox(
        "Suspected cause",
        options=list(range(len(causal_labels))),
        format_func=lambda i: causal_labels[i],
        index=0,
    )
    causal_value = causal_values[causal_idx]

    causal_notes = st.text_area(
        "Notes (optional)", value="", height=80,
        help="Free-text observation: locality, season, prior interventions.",
    )

    st.divider()
    st.header("Retrieval strategy")
    strategy = st.radio(
        "Query construction",
        options=["B", "A"],
        index=0 if app_config.DEFAULT_STRATEGY == "B" else 1,
        format_func=lambda s: (
            "Strategy B — LLM-mediated (default)" if s == "B"
            else "Strategy A — Template"
        ),
        help=(
            "Strategy B is the Phase 8 retrieval winner (0.59-0.96 vs A's "
            "0.01-0.04). Toggle to A for an ablation comparison."
        ),
    )
    show_heatmaps = st.checkbox("Show Grad-CAM heatmaps", value=True)
    top_k = st.slider("Chunks to retrieve", 1, 10, app_config.DEFAULT_TOP_K)


# --------------------------------------------------------------------- #
# Load engines (cached)
# --------------------------------------------------------------------- #

with st.spinner("Loading models (one-time per session)..."):
    bundle = load_all()
status_line = report_vram(prefix="✓ Models loaded — ")
if status_line:
    st.success(status_line)


# --------------------------------------------------------------------- #
# Uploaders
# --------------------------------------------------------------------- #

col_leaf, col_soil = st.columns(2)
with col_leaf:
    st.subheader("Leaf photo")
    leaf_file = st.file_uploader(
        "Upload leaf image", type=["jpg", "jpeg", "png", "webp"],
        key="leaf_uploader",
    )
    if leaf_file is not None:
        st.image(leaf_file, use_container_width=True)
with col_soil:
    st.subheader("Soil photo")
    soil_file = st.file_uploader(
        "Upload soil image", type=["jpg", "jpeg", "png", "webp"],
        key="soil_uploader",
    )
    if soil_file is not None:
        st.image(soil_file, use_container_width=True)

analyze = st.button(
    "Analyze",
    type="primary",
    disabled=(leaf_file is None or soil_file is None),
    use_container_width=True,
)


# --------------------------------------------------------------------- #
# Analysis flow
# --------------------------------------------------------------------- #


def _to_pil(uploader_file: Any) -> Image.Image:
    """Streamlit ``UploadedFile`` → PIL.Image (RGB)."""
    data = uploader_file.read()
    uploader_file.seek(0)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _free_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_query(
    context: MultimodalContext, strategy_choice: str, llm: Any,
) -> str:
    if strategy_choice == "A":
        return TemplateStrategy(TemplateStrategyConfig()).build_query(context)
    return LLMMediatedStrategy(LLMMediatedStrategyConfig()).build_query(
        context, llm=llm,
    )


def _run_rag_with_oom_retry(query: str, top_k: int) -> Any:
    """Run RAG; on OOM, free cache + retry with smaller token budget."""
    try:
        return bundle.rag_pipeline.answer(query, k=top_k)
    except Exception as exc:  # broad on purpose
        msg = str(exc).lower()
        if "out of memory" not in msg and "cuda" not in msg:
            raise
        _LOGGER.warning("OOM on first generation, retrying smaller: %s", exc)
        _free_cuda_cache()
        gen = getattr(bundle.rag_pipeline, "generator", None)
        old = getattr(gen, "max_new_tokens", None) if gen is not None else None
        try:
            if gen is not None and old is not None:
                gen.max_new_tokens = 256
            return bundle.rag_pipeline.answer(query, k=top_k)
        finally:
            if gen is not None and old is not None:
                gen.max_new_tokens = old


if analyze and leaf_file is not None and soil_file is not None:
    leaf_img = _to_pil(leaf_file)
    soil_img = _to_pil(soil_file)

    # ----- Step 1: guardrail ---------------------------------------- #
    with st.spinner("Checking leaf upload..."):
        ok, reason = is_leaf(
            leaf_img,
            has_no_leaf_class=app_config.HAS_NO_LEAF_CLASS,
            disease_engine=bundle.disease_engine,
            foreground_min=app_config.LEAF_FOREGROUND_MIN,
            foreground_max=app_config.LEAF_FOREGROUND_MAX,
            segment_style=app_config.GUARDRAIL_SEGMENT_STYLE,
        )
    if not ok:
        st.error(app_config.NOT_A_LEAF_MESSAGE)
        st.caption(f"Detail: {reason}")
        st.stop()

    # ----- Step 2: vision inference --------------------------------- #
    with st.spinner("Running disease + soil inference..."):
        disease_result = bundle.disease_engine.predict(leaf_img)
        soil_result = bundle.soil_engine.predict(soil_img, with_embedding=True)

    st.subheader("Predictions")
    col_d, col_s = st.columns(2)
    with col_d:
        d_pred = disease_result.prediction
        st.metric("Disease", d_pred.class_name, f"{d_pred.confidence:.0%}")
        with st.expander("Top-5 candidates"):
            for name, prob in (disease_result.top_k or [])[:5]:
                st.write(f"- **{name}** — {prob:.0%}")
    with col_s:
        s_pred = soil_result.prediction
        st.metric("Soil type", s_pred.soil_type)
        st.write(f"Moisture: **{s_pred.moisture_appearance}**")
        st.write(f"Texture: **{s_pred.texture}**")
        with st.expander("Per-head confidence"):
            for head, conf in (s_pred.per_head_confidence or {}).items():
                st.write(f"- {head}: {conf:.0%}")

    # ----- Step 3: Grad-CAM ----------------------------------------- #
    if show_heatmaps:
        with st.expander("Why the model decided this (Grad-CAM heatmaps)"):
            with st.spinner("Computing 4 heatmaps..."):
                try:
                    from src.explain.gradcam import (
                        disease_gradcam, soil_gradcam,
                    )
                    cam_cols = st.columns(2)
                    with cam_cols[0]:
                        st.caption("Disease (predicted class)")
                        cam = disease_gradcam(leaf_img, bundle.disease_engine)
                        st.image(cam.overlay_rgb, use_container_width=True)
                    with cam_cols[1]:
                        for head in ("soil_type", "moisture", "texture"):
                            st.caption(f"Soil — {head}")
                            cam = soil_gradcam(
                                soil_img, bundle.soil_engine, head=head,
                            )
                            st.image(cam.overlay_rgb, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Grad-CAM step skipped: {exc}")

    # ----- Step 4: build context + query ---------------------------- #
    context = MultimodalContext(
        disease_pred=disease_result.prediction,
        soil_pred=soil_result.prediction,
        crop_type=crop,
        causal_context=CausalContext(
            pathway=CausalPathway(causal_value),
            notes=(causal_notes.strip() or None),
        ),
        disease_emb=None,
        soil_emb=soil_result.embedding,
    )

    with st.spinner("Reformulating query for IKS corpus retrieval..."):
        try:
            query = _build_query(context, strategy, llm=bundle.llm)
        except Exception as exc:
            st.warning(
                f"Strategy {strategy} failed ({exc}); falling back to Strategy A."
            )
            query = TemplateStrategy(
                TemplateStrategyConfig(),
            ).build_query(context)

    st.subheader("Retrieval query")
    st.code(query, language="markdown")

    # ----- Step 5: grounded RAG ------------------------------------- #
    with st.spinner("Retrieving + grounding answer in IKS corpus..."):
        try:
            rag = _run_rag_with_oom_retry(query, top_k=top_k)
        except Exception as exc:
            st.error(
                "Generation failed. Try a shorter query or restart the "
                "session if VRAM is exhausted."
            )
            with st.expander("Diagnostics"):
                st.code(traceback.format_exc())
            st.stop()

    st.subheader("Grounded recommendation")
    st.markdown(rag.answer)
    if rag.citations:
        st.caption("Cited: " + " · ".join(rag.citations))

    # ----- Step 6: retrieved chunks with highlights ----------------- #
    with st.expander(f"Show the {len(rag.retrieved)} retrieved source chunks"):
        try:
            from src.explain.chunk_highlight import explain_chunks
            explained = explain_chunks(query, rag.retrieved)
            for ex in explained:
                st.markdown(
                    f"**#{ex.rank}** — {ex.source_text} "
                    f"ch.{ex.chapter} v.{ex.verse_or_section}  "
                    f"_(score {ex.score:.3f})_"
                )
                st.markdown(ex.text_with_markers)
                if ex.matched_terms:
                    st.caption("Matched: " + ", ".join(ex.matched_terms))
                st.divider()
        except Exception as exc:
            st.warning(f"Chunk highlighter skipped: {exc}")
            for ch in rag.retrieved:
                st.markdown(
                    f"**{getattr(ch, 'source_text', '?')}** "
                    f"ch.{getattr(ch, 'chapter', '?')} "
                    f"v.{getattr(ch, 'verse_or_section', '?')}"
                )
                st.write(getattr(ch, "text", ""))
                st.divider()

    # ----- Step 7: disclaimer --------------------------------------- #
    st.info(app_config.DISCLAIMER)

elif not analyze:
    st.info("Upload a leaf photo and a soil photo, then click **Analyze**.")
