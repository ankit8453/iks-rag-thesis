"""Streamlit-cached loaders for the Phase 10 UI.

All five heavyweight resources are built ONCE per Streamlit session via
``@st.cache_resource``:

* Disease engine (EfficientNet-B4 @ 380, fp32, on GPU)
* Soil engine (EfficientNet-B0 @ 224, fp32 multi-task, on GPU)
* IKS ChromaDB collection (RAM, built from HF dataset on first call)
* Hybrid retriever (CPU embedder + sparse BM25 + CPU reranker)
* RAGPipeline (Llama-3.1-8B 4-bit on GPU + the retriever above)

T4 memory plan — the Phase 8 OOM came from stacking everything on GPU:

* Llama 4-bit on GPU         ~5.5 GB
* Disease B4 on GPU          ~0.4 GB
* Soil B0 on GPU             ~0.1 GB
* bge-large embedder on CPU  (NOT GPU — saves ~1.5 GB)
* bge reranker on CPU        (NOT GPU — saves ~0.5 GB)
* ChromaDB in RAM            ~50 MB

Total GPU footprint ~6.1 GB; comfortably under the T4's 16 GB envelope.

This module deliberately does NOT import streamlit at module load — the
``@st.cache_resource`` decorator is applied conditionally so the loaders
can be unit-tested without streamlit installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from app import config as app_config
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.disease.infer import DiseaseInferenceEngine  # noqa: F401
    from src.rag.pipeline import RAGPipeline  # noqa: F401
    from src.soil.infer import SoilInferenceEngine  # noqa: F401

_LOGGER = get_logger(__name__)


# --------------------------------------------------------------------- #
# Conditional cache decorator
# --------------------------------------------------------------------- #


def _cache_resource(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``st.cache_resource(fn)`` when streamlit is importable,
    otherwise return ``fn`` unchanged so loaders work in plain Python /
    unit tests."""
    try:
        import streamlit as st  # type: ignore
        return st.cache_resource(fn)  # type: ignore[attr-defined]
    except Exception:  # streamlit absent OR script-context missing in tests
        return fn


# --------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------- #


@dataclass
class EngineBundle:
    """All five heavyweight resources, ready for use.

    Attributes
    ----------
    disease_engine, soil_engine
        Vision inference engines.
    rag_pipeline
        Phase 7 ``RAGPipeline`` wired to its own retriever + Llama generator.
    llm
        The same ``GroundedGenerator`` instance the pipeline uses, exposed
        here so Strategy B can call ``.complete(prompt)`` without a
        second load.
    device
        ``"cuda"`` / ``"cpu"`` the vision engines were placed on.
    """

    disease_engine: Any
    soil_engine: Any
    rag_pipeline: Any
    llm: Any
    device: str


# --------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------- #


@_cache_resource
def load_disease_engine(
    repo: str = app_config.DISEASE_MODEL_REPO,
    device: str = "cuda",
    work_dir: Path | None = None,
) -> Any:
    """Load the disease inference engine from HF Hub.

    Returns the engine cached for the Streamlit session — subsequent
    calls with the same args are free.
    """
    from src.disease.infer import DiseaseInferenceEngine  # local import

    _LOGGER.info("Loading disease engine: repo=%s device=%s", repo, device)
    engine = DiseaseInferenceEngine(
        model_source=repo, device=device, work_dir=work_dir,
    )
    _LOGGER.info(
        "Disease engine ready: %d classes (first=%r)",
        engine.num_classes,
        engine.class_names[0] if engine.class_names else "?",
    )
    return engine


@_cache_resource
def load_soil_engine(
    repo: str = app_config.SOIL_MODEL_REPO,
    device: str = "cuda",
    work_dir: Path | None = None,
) -> Any:
    """Load the soil multi-task inference engine from HF Hub."""
    from src.soil.infer import SoilInferenceEngine  # local import

    _LOGGER.info("Loading soil engine: repo=%s device=%s", repo, device)
    engine = SoilInferenceEngine(
        model_source=repo, device=device, work_dir=work_dir,
    )
    _LOGGER.info(
        "Soil engine ready: heads=[soil_type=%d, moisture=%d, texture=%d]",
        len(engine.soil_type_classes),
        len(engine.moisture_classes),
        len(engine.texture_classes),
    )
    return engine


@_cache_resource
def load_rag_pipeline(
    corpus_repo: str = app_config.CORPUS_REPO,
    llm_model_name: str = app_config.LLM_MODEL_NAME,
    persist_dir: Path | None = None,
    default_k: int = app_config.DEFAULT_TOP_K,
) -> Any:
    """Build the ChromaDB + retriever + generator stack and return the pipeline.

    The ChromaDB collection is rebuilt from the HF corpus dataset on
    first call (cheap: ~1 min for 206 chunks at bge-large on CPU). The
    embedder + reranker stay on CPU — they're the long-pole-of-VRAM and
    the Phase 8 OOM driver if pushed to GPU.
    """
    from src.rag.corpus_loader import build_chroma, load_chunks_from_hf
    from src.rag.generator import GroundedGenerator
    from src.rag.pipeline import RAGPipeline

    _LOGGER.info("Loading corpus from %s ...", corpus_repo)
    chunks = load_chunks_from_hf(repo=corpus_repo)
    _LOGGER.info("Corpus chunks loaded: %d", len(chunks))

    if persist_dir is None:
        # In-memory by default — ChromaDB rebuild is cheap, and Colab
        # /tmp is gone on session timeout anyway. Caller can pass a
        # persistent dir for laptop runs.
        persist_dir = Path("/tmp/iks_chroma_phase10")
    persist_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Building Chroma collection at %s ...", persist_dir)
    collection = build_chroma(chunks, persist_dir=persist_dir)

    _LOGGER.info("Building Llama generator: %s", llm_model_name)
    generator = GroundedGenerator(model_name=llm_model_name)

    pipeline = RAGPipeline(
        collection=collection, generator=generator, default_k=default_k,
    )
    return pipeline


# --------------------------------------------------------------------- #
# Convenience: build the whole bundle at once
# --------------------------------------------------------------------- #


def load_all(
    device: str = "cuda",
    work_dir: Path | None = None,
) -> EngineBundle:
    """Load disease + soil + RAG and return them in a single bundle.

    Call this once from the Streamlit script. Each loader is cached
    individually so re-runs of this function are free.
    """
    disease_engine = load_disease_engine(device=device, work_dir=work_dir)
    soil_engine = load_soil_engine(device=device, work_dir=work_dir)
    rag_pipeline = load_rag_pipeline()
    return EngineBundle(
        disease_engine=disease_engine,
        soil_engine=soil_engine,
        rag_pipeline=rag_pipeline,
        llm=rag_pipeline.generator,
        device=device,
    )


def report_vram(prefix: str = "") -> str:
    """Return a short human-readable VRAM status line.

    Returns an empty string on CPU-only / non-CUDA hosts. Used by the
    Streamlit script to print a "model load complete, GPU at X.XX GB"
    line right after :func:`load_all`.
    """
    try:
        import torch  # local import — never blocking
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()  # bytes
        used = total - free
        gb = lambda x: x / (1024 ** 3)  # noqa: E731
        return (
            f"{prefix}GPU memory: used {gb(used):.2f} GB / "
            f"total {gb(total):.2f} GB ({100 * used / total:.1f}%)"
        )
    except Exception:
        return ""


__all__ = [
    "EngineBundle",
    "load_all",
    "load_disease_engine",
    "load_rag_pipeline",
    "load_soil_engine",
    "report_vram",
]
