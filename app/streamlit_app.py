"""Phase 10 — full-system Streamlit UI (modern / futuristic).

Wires Phases 5 (disease) + 6 (soil) + 7 (RAG) + 8 (integration) +
9 (explainability) into one interactive demo, with the validated
Phase 9 pipeline:

  - C-PD disease model + YOLO leaf crop (crop-first).
  - eigen-smoothed Grad-CAM, shown ONLY when a disease is predicted.
  - advisory (Strategy-B query → grounded IKS answer) ONLY for disease;
    healthy leaves get "no treatment needed".

Run locally::  streamlit run app/streamlit_app.py
Run on Colab via ``notebooks/phase10_launch_ui.ipynb`` (cloudflared tunnel).
"""

from __future__ import annotations

# Streamlit prepends app/ to sys.path, shadowing the top-level ``app``
# package. Inject the repo root before any ``from app import ...``.
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
from app import crop_soil
from app.guardrail import is_leaf
from app.loaders import load_all, report_vram
from src.soil.model import SoilPrediction
from src.integration.causation import CausalContext, CausalPathway
from src.integration.config import LLMMediatedStrategyConfig, TemplateStrategyConfig
from src.integration.context import MultimodalContext
from src.integration.strategy_llm_mediated import LLMMediatedStrategy
from src.integration.strategy_template import TemplateStrategy
from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

# --------------------------------------------------------------------- #
# Page setup + futuristic theme
# --------------------------------------------------------------------- #

st.set_page_config(page_title=app_config.APP_TITLE, page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(1100px 520px at 15% -10%, #103b3a 0%, rgba(16,59,58,0) 55%),
          radial-gradient(900px 500px at 95% 0%, #1a2c52 0%, rgba(26,44,82,0) 50%),
          linear-gradient(180deg, #070d12 0%, #05080b 100%);
        color: #e6f0f2;
      }
      #MainMenu, footer {visibility: hidden;}
      .hero-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing: .5px; margin-bottom: 0;
        background: linear-gradient(90deg, #34e0a1, #3ec6ff 55%, #b18cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }
      .hero-sub { color: #93a7af; font-size: 1.02rem; margin-top: 2px; }
      .glass {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
        backdrop-filter: blur(9px);
        box-shadow: 0 10px 34px rgba(0,0,0,0.40);
      }
      .glass-ok  { border-left: 4px solid #34e0a1; }
      .glass-bad { border-left: 4px solid #ff7a7a; }
      .pill {
        display:inline-block; padding:3px 12px; border-radius:999px;
        font-size:.78rem; font-weight:700; letter-spacing:.4px;
      }
      .pill-ok  { background:rgba(52,224,161,.16); color:#5ff0bf; border:1px solid rgba(52,224,161,.4); }
      .pill-bad { background:rgba(255,122,122,.16); color:#ff9d9d; border:1px solid rgba(255,122,122,.4); }
      .big { font-size:1.7rem; font-weight:800; margin:6px 0 2px; }
      .muted { color:#8fa3ab; font-size:.85rem; }
      .stButton>button {
        border-radius: 12px; font-weight: 700; border: 0;
        background: linear-gradient(90deg, #34e0a1, #3ec6ff);
        color: #04121a;
      }
      .stButton>button:disabled { background:#243038; color:#5b6b73; }
      h1,h2,h3,h4 { color:#eaf4f6; }
      .stProgress > div > div { background-image: linear-gradient(90deg,#34e0a1,#3ec6ff); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="hero-title">🌿 {app_config.APP_TITLE}</div>'
    f'<div class="hero-sub">{app_config.APP_TAGLINE}</div>',
    unsafe_allow_html=True,
)
st.write("")


# --------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Inputs")
    # Two entry modes: Full (leaf + soil photo) vs Quick check (leaf only, with
    # the soil reading auto-filled to typical baseline conditions for the crop).
    mode = st.radio(
        "Analysis mode", options=["full", "quick"], index=0,
        format_func=lambda m: ("Full — leaf + soil photo" if m == "full"
                               else "Quick check — leaf only (typical soil)"),
        help="Quick check skips the soil photo and uses typical soil conditions "
             "for the crop instead of a measured sample.",
    )
    quick = mode == "quick"
    causal_labels = [label for _, label in app_config.CAUSAL_CHOICES]
    causal_values = [value for value, _ in app_config.CAUSAL_CHOICES]
    causal_idx = st.selectbox(
        "Suspected cause", options=list(range(len(causal_labels))),
        format_func=lambda i: causal_labels[i], index=0,
    )
    causal_value = causal_values[causal_idx]
    causal_notes = st.text_area("Notes (optional)", value="", height=80)

    st.divider()
    st.header("🔎 Retrieval")
    strategy = st.radio(
        "Query construction", options=["B", "A"],
        index=0 if app_config.DEFAULT_STRATEGY == "B" else 1,
        format_func=lambda s: ("Strategy B — LLM-mediated (default)" if s == "B"
                               else "Strategy A — Template (baseline)"),
        help="Strategy B (the contribution) bridges modern labels → classical vocabulary.",
    )
    show_heatmaps = st.checkbox("Show Grad-CAM heatmaps", value=True)
    top_k = st.slider("Chunks to retrieve", 1, 10, app_config.DEFAULT_TOP_K)
    st.divider()
    st.caption(app_config.MODEL_NOTE)


# --------------------------------------------------------------------- #
# Load engines (cached, once per session)
# --------------------------------------------------------------------- #

with st.spinner("Booting models (one-time per session)…"):
    bundle = load_all()
_status = report_vram(prefix="Models online — ")
if _status:
    st.markdown(f'<div class="glass glass-ok">⚡ {_status}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Uploaders
# --------------------------------------------------------------------- #

# --- Plant selection = the SCOPE GATE ------------------------------------- #
# The classifier has a fixed class set and cannot say "I don't know" — it always
# returns its closest class. So we do NOT infer support from the model. The
# farmer declares the plant (they reliably know their own crop), which tells us
# with certainty whether it is inside the trained scope BEFORE anything runs.
# The model is then used for what it is actually good at: naming the disease.
_supported = app_config.supported_crops(getattr(bundle.disease_engine, "class_names", []))
st.markdown("#### 🌱 Which plant is this?")
sel_col, note_col = st.columns([1, 2])
with sel_col:
    selected_crop = st.selectbox(
        "Plant", options=[*_supported, app_config.OTHER_CROP], index=0,
        label_visibility="collapsed",
    )
with note_col:
    st.caption(f"These {len(_supported)} plants are what the model was trained on — "
               "its honest scope. Not listed? Choose “Other”.")
other_crop_name = ""
if selected_crop == app_config.OTHER_CROP:
    other_crop_name = st.text_input(
        "Type the plant name", value="", placeholder="e.g. brinjal",
        help="We can't diagnose this plant yet, but recording the name helps us add it later.",
    ).strip()
st.write("")

col_leaf, col_soil = st.columns(2)
with col_leaf:
    st.markdown("#### 🍃 Leaf photo")
    leaf_file = st.file_uploader("Upload leaf", type=["jpg", "jpeg", "png", "webp"],
                                 key="leaf_uploader", label_visibility="collapsed")
    if leaf_file is not None:
        st.image(leaf_file, use_container_width=True)
with col_soil:
    if quick:
        st.markdown("#### 🟤 Soil — *skipped (Quick check)*")
        st.info("Quick check uses **typical soil conditions** for the crop. "
                "For a measured soil reading, switch to **Full** mode.")
        soil_file = None
    else:
        st.markdown("#### 🟤 Soil photo")
        soil_file = st.file_uploader("Upload soil", type=["jpg", "jpeg", "png", "webp"],
                                     key="soil_uploader", label_visibility="collapsed")
        if soil_file is not None:
            st.image(soil_file, use_container_width=True)

# "Same-location" safeguard (Full mode only): a mismatched leaf+soil pair (leaf
# from one place, soil from another) would silently produce a wrong, unfair
# result. Require the user to confirm both photos are from the same spot.
if quick:
    same_location = True
    can_analyze = leaf_file is not None
else:
    same_location = st.checkbox(
        "I confirm the leaf photo and the soil photo are from the **same plant / field**.",
        value=False,
        help="Prevents a mismatched leaf+soil pair from producing a misleading result.",
    )
    can_analyze = (leaf_file is not None and soil_file is not None and same_location)

# For an out-of-scope plant we still need the typed name — it is what makes the
# saved sample useful for a future retraining round.
if selected_crop == app_config.OTHER_CROP and not other_crop_name:
    can_analyze = False

analyze = st.button("✨  Analyze", type="primary", disabled=not can_analyze,
                    use_container_width=True)
if selected_crop == app_config.OTHER_CROP and not other_crop_name and leaf_file is not None:
    st.caption("☝️ Type the plant name above to continue.")
if (not quick) and leaf_file is not None and soil_file is not None and not same_location:
    st.caption("☝️ Tick the same-location confirmation above to enable Analyze.")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _to_pil(uploader_file: Any) -> Image.Image:
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


def _build_query(context: MultimodalContext, strategy_choice: str, llm: Any) -> str:
    if strategy_choice == "A":
        return TemplateStrategy(TemplateStrategyConfig()).build_query(context)
    return LLMMediatedStrategy(LLMMediatedStrategyConfig()).build_query(context, llm=llm)


def _run_rag_with_oom_retry(query: str, k: int) -> Any:
    try:
        return bundle.rag_pipeline.answer(query, k=k)
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" not in msg and "cuda" not in msg:
            raise
        _LOGGER.warning("OOM on generation, retrying smaller: %s", exc)
        _free_cuda_cache()
        gen = getattr(bundle.rag_pipeline, "generator", None)
        old = getattr(gen, "max_new_tokens", None) if gen is not None else None
        try:
            if gen is not None and old is not None:
                gen.max_new_tokens = 256
            return bundle.rag_pipeline.answer(query, k=k)
        finally:
            if gen is not None and old is not None:
                gen.max_new_tokens = old


# --------------------------------------------------------------------- #
# Analysis flow
# --------------------------------------------------------------------- #

if analyze and leaf_file is not None and (quick or soil_file is not None):
    leaf_img = _to_pil(leaf_file)
    soil_img = _to_pil(soil_file) if soil_file is not None else None

    # 0) SCOPE GATE — runs before any model, because the classifier cannot say
    # "I don't know": it would return its closest trained class and present a
    # confident, wrong diagnosis. The farmer's declaration settles scope exactly.
    if selected_crop == app_config.OTHER_CROP:
        st.markdown(
            f'<div class="glass glass-bad">🚫 '
            f'{app_config.OUT_OF_SCOPE_MESSAGE.format(crop=other_crop_name.title())}'
            f'</div>', unsafe_allow_html=True)
        st.caption("Supported plants: " + ", ".join(_supported))
        st.stop()

    # 1) guardrail
    with st.spinner("Checking the leaf upload…"):
        ok, reason = is_leaf(
            leaf_img, has_no_leaf_class=app_config.HAS_NO_LEAF_CLASS,
            disease_engine=bundle.disease_engine,
            foreground_min=app_config.LEAF_FOREGROUND_MIN,
            foreground_max=app_config.LEAF_FOREGROUND_MAX,
            segment_style=app_config.GUARDRAIL_SEGMENT_STYLE,
        )
    if not ok:
        st.markdown(f'<div class="glass glass-bad">🚫 {app_config.NOT_A_LEAF_MESSAGE}<br>'
                    f'<span class="muted">Detail: {reason}</span></div>', unsafe_allow_html=True)
        st.stop()

    # 2) crop-first vision inference
    with st.spinner("Detecting leaf → cropping → classifying…"):
        leaf_crop, found = bundle.cropper.crop(leaf_img)
        disease_result = bundle.disease_engine.predict(leaf_crop)
        d_pred = disease_result.prediction
        healthy = app_config.is_healthy(d_pred.class_name)
        # The farmer declared the plant, and they know their own crop — so that
        # is what we act on. The disease label also implies a crop; we use it
        # only as a CROSS-CHECK, surfacing a warning when the two disagree
        # rather than silently overriding the person.
        detected_crop = selected_crop
        model_crop = app_config.crop_from_disease(d_pred.class_name)
        crop_mismatch = model_crop != selected_crop
        cs_row = crop_soil.find(detected_crop)
        if quick:
            # Quick check: no soil photo — auto-fill the crop's typical baseline
            # soil reading (primary soil + driest acceptable moisture + primary
            # texture). Clearly flagged downstream as "typical, not measured".
            soil_emb = None
            if cs_row is not None:
                b = crop_soil.baseline(cs_row)
                s_pred = SoilPrediction(soil_type=b["soil_type"].capitalize(),
                                        moisture_appearance=b["moisture"],
                                        texture=b["texture"])
                soil_source = "typical"
            else:
                s_pred = SoilPrediction(soil_type="unspecified",
                                        moisture_appearance="moderate", texture="mixed")
                soil_source = "unknown"
        else:
            soil_result = bundle.soil_engine.predict(soil_img, with_embedding=True)
            s_pred = soil_result.prediction
            soil_emb = soil_result.embedding
            soil_source = "measured"

    # 3) status banner
    if healthy:
        st.markdown(
            f'<div class="glass glass-ok"><span class="pill pill-ok">✓ HEALTHY</span>'
            f'<div class="big">{d_pred.class_name}</div>'
            f'<span class="muted">No disease detected ({d_pred.confidence:.0%} confidence) — '
            f'no treatment required.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="glass glass-bad"><span class="pill pill-bad">⚠ DISEASE</span>'
            f'<div class="big">{d_pred.class_name}</div>'
            f'<span class="muted">{d_pred.confidence:.0%} confidence · crop: {detected_crop}</span></div>',
            unsafe_allow_html=True)

    # 3b) cross-check: the plant the farmer chose vs the plant the disease
    # label implies. Non-blocking on purpose — the farmer is the authority on
    # their own crop; we flag the disagreement instead of silently deciding.
    if crop_mismatch:
        st.markdown(
            f'<div class="glass glass-bad">⚠ '
            f'{app_config.CROP_MISMATCH_MESSAGE.format(selected=selected_crop, detected=model_crop)}'
            f'</div>', unsafe_allow_html=True)

    # 4) prediction detail cards
    col_d, col_s = st.columns(2)
    with col_d:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**🍃 Leaf — top candidates**")
        for name, prob in (disease_result.top_k or [])[:5]:
            st.write(f"- {name} — {prob:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_s:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        soil_title = "🟤 Soil — *typical (Quick check)*" if quick else "🟤 Soil"
        st.markdown(f"**{soil_title}**")
        st.write(f"- Type: **{s_pred.soil_type}**")
        st.write(f"- Moisture: **{s_pred.moisture_appearance}**")
        st.write(f"- Texture: **{s_pred.texture}**")
        if soil_source == "typical":
            st.caption(f"Typical baseline for **{detected_crop}** — not a measured soil sample.")
        elif soil_source == "unknown":
            st.caption(f"'{detected_crop}' is not in the crop–soil reference; using neutral defaults.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 4b) crop–soil suitability — Full mode only (Quick mode is baseline-by-construction)
    if not quick and cs_row is not None:
        verdict = crop_soil.check_suitability(
            cs_row, soil_type=s_pred.soil_type, texture=s_pred.texture,
            moisture=s_pred.moisture_appearance,
        )
        st.markdown("#### 🧭 Crop–soil suitability")
        if verdict["ok"]:
            st.markdown(f'<div class="glass glass-ok">✓ The detected soil looks suitable for '
                        f'<b>{detected_crop}</b>.</div>', unsafe_allow_html=True)
        else:
            bullets = "".join(f"<li>{m}</li>" for m in verdict["messages"])
            st.markdown(f'<div class="glass glass-bad">⚠ Possible crop–soil mismatch for '
                        f'<b>{detected_crop}</b>:<ul>{bullets}</ul>'
                        f'<span class="muted">Indicative only — auto-derived from the crop–soil '
                        f'reference, pending expert validation.</span></div>',
                        unsafe_allow_html=True)

    # 5) Grad-CAM — heatmap ONLY for disease
    if show_heatmaps:
        st.markdown("#### 🔬 Where the model looked")
        from src.explain.gradcam import disease_gradcam_eigen, soil_gradcam
        cam_cols = st.columns(4)
        with cam_cols[0]:
            st.caption("Leaf crop")
            st.image(leaf_crop, use_container_width=True)
        with cam_cols[1]:
            if healthy:
                st.caption("Disease heatmap")
                st.info("Healthy — no disease region to highlight.")
            else:
                st.caption("Disease Grad-CAM (lesion)")
                try:
                    dcam = disease_gradcam_eigen(leaf_crop, bundle.disease_engine)
                    st.image(dcam.overlay_rgb, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Grad-CAM skipped: {exc}")
        # soil heads — only when a real soil photo was provided (Full mode)
        if quick or soil_img is None:
            with cam_cols[2]:
                st.caption("Soil heatmaps")
                st.info("Skipped in Quick check (no soil photo).")
        else:
            for col, head in zip(cam_cols[2:], ("soil_type", "moisture")):
                with col:
                    st.caption(f"Soil — {head}")
                    try:
                        scam = soil_gradcam(soil_img, bundle.soil_engine, head=head)
                        st.image(scam.overlay_rgb, use_container_width=True)
                    except Exception as exc:
                        st.warning(f"skipped: {exc}")

    # 6) advisory — ONLY for disease (gated)
    st.markdown("#### 📜 IKS advisory")
    if healthy:
        st.markdown('<div class="glass glass-ok">✓ The leaf appears healthy — '
                    'no treatment advisory is generated. Keep monitoring.</div>',
                    unsafe_allow_html=True)
    else:
        context = MultimodalContext(
            disease_pred=d_pred, soil_pred=s_pred, crop_type=detected_crop,
            causal_context=CausalContext(pathway=CausalPathway(causal_value),
                                         notes=(causal_notes.strip() or None)),
            disease_emb=None, soil_emb=soil_emb,
        )
        with st.spinner("Bridging the query to classical vocabulary (Strategy B)…"):
            try:
                query = _build_query(context, strategy, llm=bundle.llm)
            except Exception as exc:
                st.warning(f"Strategy {strategy} failed ({exc}); using Strategy A.")
                query = TemplateStrategy(TemplateStrategyConfig()).build_query(context)
        st.markdown("**Retrieval query**")
        st.code(query, language="markdown")

        with st.spinner("Retrieving + grounding in the IKS corpus…"):
            try:
                rag = _run_rag_with_oom_retry(query, k=top_k)
            except Exception:
                st.error("Generation failed (try restarting the session if VRAM is exhausted).")
                with st.expander("Diagnostics"):
                    st.code(traceback.format_exc())
                st.stop()

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Grounded recommendation**")
        st.markdown(rag.answer)
        if rag.citations:
            st.caption("Cited: " + " · ".join(rag.citations))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander(f"Show the {len(rag.retrieved)} retrieved source chunks"):
            try:
                from src.explain.chunk_highlight import explain_chunks
                for ex in explain_chunks(query, rag.retrieved):
                    st.markdown(f"**#{ex.rank}** — {ex.source_text} "
                                f"ch.{ex.chapter} v.{ex.verse_or_section} _(score {ex.score:.3f})_")
                    st.markdown(ex.text_with_markers)
                    if ex.matched_terms:
                        st.caption("Matched: " + ", ".join(ex.matched_terms))
                    st.divider()
            except Exception as exc:
                st.warning(f"Chunk highlighter skipped: {exc}")

    st.markdown(f'<div class="glass"><span class="muted">{app_config.DISCLAIMER}</span></div>',
                unsafe_allow_html=True)

elif not analyze:
    _hint = ("⬆️ Upload a <b>leaf</b> photo, then press <b>Analyze</b> "
             "(Quick check — typical soil for the crop)." if quick else
             "⬆️ Upload a <b>leaf</b> photo and a <b>soil</b> photo, confirm same location, "
             "then press <b>Analyze</b>.")
    st.markdown(f'<div class="glass">{_hint}</div>', unsafe_allow_html=True)
