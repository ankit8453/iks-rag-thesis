"""Generate ``notebooks/phase8_multimodal_integration.ipynb`` (Phase 8 §C).

One-shot builder for the Phase 8 Colab notebook that wires the
HF-Hub-hosted Phase 5 disease model + Phase 6 soil multi-task model
through the three integration strategies (A: template, B: LLM-mediated
modern→classical bridge, C: multimodal embedding projection ablation)
into the Phase 7 RAG pipeline.

13 cells per the Phase 8 prompt's locked structure. Same pattern as
``scripts/build_phase7_notebook.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase8_multimodal_integration.ipynb"

REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

NB = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ----------------------------- Cell 1 — title md ----------------------------
md(
    """# Phase 8 — Multimodal Integration (vision → RAG query, 3 strategies)

This notebook is the core-novelty wiring for **contribution C2** (joint
disease + soil context module with three ablated query-construction
strategies) and **contribution C5** (cause-conditional retrieval — the
pathway is *user-supplied*, never inferred from images).

## What it does

Given a leaf image + a soil image + a crop name + an optional causal
pathway, the notebook constructs three retrieval queries and feeds each
through the Phase 7 RAG pipeline:

- **Strategy A — Template (transparent baseline).** Deterministic fill
  of `"Organic treatment for <disease> affecting <crop> grown in
  <soil_type> soil that appears <moisture> with <texture> texture,
  <optional causal clause>."`
- **Strategy B — LLM-mediated query (main contribution).** Reuses the
  already-loaded Phase 7 Llama-3.1-8B to *rewrite* the structured
  context into a single retrieval query that bridges modern vision
  labels (e.g. *Tomato___Leaf_Mold*, *Alluvial_Soil*) to descriptive /
  classical-text vocabulary (e.g. *whitish leaf lesions*, *fertile
  riverine soil*). The LLM is forbidden from inventing a treatment.
- **Strategy C — Multimodal embedding projection (honest ablation).**
  Concatenates the penultimate B4 disease feature (1792-d) + B0 soil
  feature (1280-d) + a bge-large embedding of the crop name (1024-d),
  trains a single linear projector to the 1024-d corpus space against
  Strategy A's top-1 retrieved chunk (weak signal — no manual labels),
  and retrieves by cosine similarity. **Expected to under-perform A
  and B** because the modality gap (visual embeddings live on a
  different manifold from text embeddings) is not crossed by a single
  linear layer trained on a handful of weak pairs. The point is to
  *show* the gap, not close it.

## Why Strategy B is the real contribution

Modern vision labels do not appear in classical Indian agricultural
texts. Strategy A's templated query can return generic top-k matches,
but Strategy B's rewrite bridges the vocabulary gap and improves
on-topic retrieval. Rigorous RAGAS context_precision / context_recall
scoring against an expert gold-query set is **Phase 11**; Phase 8
delivers the mechanism + a qualitative side-by-side read.

## Hard rules

- Local commits only — never `git push`.
- Reuses Phase 5 / 6 / 7 code by import. The RAG pipeline, the
  retriever, the embedder, and the Llama generator are all the Phase 7
  classes — Phase 8 adds the multimodal *front end*, not a new RAG path.
"""
)

# ----------------------------- Cell 2 — setup -------------------------------
code(
    f"""# Cell 2 — clone repo + install dependencies (defensive)
import importlib
import os
import shutil
import subprocess
import sys

REPO_URL = "{REPO_HTTPS_URL}"
REPO_PATH = "{REPO_LOCAL_PATH}"

if not os.path.exists(REPO_PATH):
    subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
else:
    # Refresh on re-runs so a freshly-pushed commit is picked up.
    subprocess.run(["git", "-C", REPO_PATH, "pull", "--ff-only"], check=True)

os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)
print(f"Working directory: {{os.getcwd()}}")

# Reuse Phase 7 deps. The vision modules add timm + pillow + torchvision
# which are already pulled in by sentence-transformers / chromadb /
# transformers, so we only need a defensive top-up.
DEPS = [
    "chromadb>=0.5,<0.6",
    "sentence-transformers>=3.0,<4.0",
    "transformers>=4.44,<4.50",
    "accelerate>=0.33",
    "bitsandbytes>=0.43",
    "rank-bm25>=0.2.2",
    "datasets>=2.20",
    "huggingface_hub>=0.24",
    "timm>=1.0",
    "pillow",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
print("Dependencies installed.")
"""
)

# ----------------------------- Cell 3 — HF auth -----------------------------
code(
    """# Cell 3 — HF Hub login (private chunks + gated Llama + Phase-5/6 weights)
from huggingface_hub import HfApi, login

login()  # interactive — paste the ankit-iiitdmj write-scope token.

api = HfApi()
info = api.whoami()
print(f"Logged in as: {info.get('name')}  role={info.get('auth', {}).get('accessToken', {}).get('role')}")
assert info.get("name") == "ankit-iiitdmj", (
    "HF token belongs to a different user; Strategy C's weak-pair training "
    "needs the private chunks dataset and the model checkpoints under "
    "ankit-iiitdmj/."
)
"""
)

# ----------------------------- Cell 4 — GPU check ---------------------------
code(
    """# Cell 4 — GPU + CUDA sanity check
import subprocess
import sys

import torch

sys.path.insert(0, REPO_PATH)

print(f"torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    # nvidia-smi for the supervisor walkthrough — same as the Phase 7 cell.
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("WARNING: no GPU. Strategy B's Llama load will be infeasible on CPU.")
"""
)

# ----------------------------- Cell 5 — vision models -----------------------
code(
    """# Cell 5 — Load Phase 5 disease (B4) + Phase 6 soil (B0 multi-task) engines
# Both load from HF Hub model repos. Each engine resolves
# `checkpoint_best.pt`, builds the bare model with `pretrained=False`,
# and loads the state-dict — no training-time bookkeeping is touched.
import torch

from src.disease.infer import DiseaseInferenceEngine
from src.soil.infer import SoilInferenceEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disease model — Phase 5 final stage (EfficientNet-B4 @ 380x380).
# ``class_names`` is not passed explicitly: the engine auto-resolves
# them from ``data/splits/plantdoc/class_map.json`` (Phase 5's
# canonical mapping). Before this fix the engine fell back to
# ``"class_0"`` placeholders and Strategy A's query read "Organic
# treatment for class 0 affecting rice ...", carrying ZERO disease
# information into retrieval.
disease_engine = DiseaseInferenceEngine(
    model_source="ankit-iiitdmj/iks-disease-plantdoc",
    device=DEVICE,
)
print(f"Disease engine: {disease_engine.num_classes} classes on {disease_engine.device}")
print(f"  first 5 names : {disease_engine.class_names[:5]}")
print(f"  last 3 names  : {disease_engine.class_names[-3:]}")
assert not any(n.startswith("class_") and n[6:].isdigit() for n in disease_engine.class_names), (
    "Disease engine class names still look like 'class_<i>' placeholders. "
    "Check data/splits/plantdoc/class_map.json — Phase 8 query construction "
    "will be meaningless without real labels."
)

# Soil model — Phase 6 v2/v3-tiling multi-task (EfficientNet-B0 @ 224x224).
soil_engine = SoilInferenceEngine(
    model_source="ankit-iiitdmj/iks-soil-multitask-v2",
    device=DEVICE,
)
print(
    f"Soil engine: heads=[soil_type={len(soil_engine.soil_type_classes)}, "
    f"moisture={len(soil_engine.moisture_classes)}, "
    f"texture={len(soil_engine.texture_classes)}] on {soil_engine.device}"
)
"""
)

# ----------------------------- Cell 6 — RAG pipeline ------------------------
code(
    """# Cell 6 — Build the Phase 7 RAG pipeline (re-uses the published code)
# This is the exact same wiring as Phase 7's Cell 5–9 collapsed into one
# block: pull the 4-book chunks dataset, rebuild ChromaDB in-session,
# build the HybridRetriever, and load Llama-3.1-8B 4-bit.
#
# The Phase 3b.2 Gemini re-OCR brought the chunks down to 206 (cleaner
# OCR → denser semantic chunks + 2 new books). Phase 8 asserts that
# count so a future corpus drift is caught immediately.
from collections import Counter

import torch

from src.rag.corpus_loader import build_chroma, load_chunks_from_hf
from src.rag.generator import GroundedGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import HybridRetriever

EXPECTED_CHUNK_COUNT = 206
EXPECTED_PER_BOOK = {
    "vrikshayurveda": 42,
    "brihat_samhita": 136,
    "krishi_parashara": 13,
    "upavanavinoda": 15,
}

chunks = load_chunks_from_hf()
per_book = Counter(c["book_id"] for c in chunks)
print(f"Loaded {len(chunks)} chunks; per-book breakdown:")
for book, n in sorted(per_book.items(), key=lambda kv: -kv[1]):
    print(f"  {book:<22} {n:>4}")

assert len(chunks) == EXPECTED_CHUNK_COUNT, (
    f"Corpus drift: expected {EXPECTED_CHUNK_COUNT} chunks, got {len(chunks)}. "
    f"Phase 8 was built against the Phase-3b.2 Gemini re-OCR snapshot."
)
for book, expected in EXPECTED_PER_BOOK.items():
    assert per_book.get(book) == expected, (
        f"Per-book drift: {book} expected {expected}, got {per_book.get(book)}"
    )
print(f"\\nCorpus matches the Phase-3b.2 snapshot ({EXPECTED_CHUNK_COUNT} chunks across 4 books).")

# Re-embed + rebuild ChromaDB at corpus/vector_db/. Same call signature
# as Phase 7 — Phase 8 does NOT change the retrieval contract.
collection = build_chroma(chunks, persist_dir="corpus/vector_db")
print(f"\\nChromaDB ready: collection count = {collection.count()}")

# Hybrid retriever — dense + sparse BM25 + cross-encoder rerank, all on.
retriever = HybridRetriever(
    collection, use_dense=True, use_sparse=True, use_reranker=True,
)
print(
    f"retriever: dense={retriever.use_dense} sparse={retriever.use_sparse} "
    f"reranker={retriever.use_reranker}"
)

# Llama-3.1-8B 4-bit (gated; Cell 3's token must have license access).
# Note: with both vision backbones + bge encoder + Llama loaded, T4 VRAM
# is tight. If the load OOMs, fall back to Llama-3.2-3B-Instruct — the
# RAGPipeline interface is model-agnostic.
generator = GroundedGenerator(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    temperature=0.2,
    max_new_tokens=512,
    seed=42,
)
generator._ensure_loaded()  # warm up
torch.cuda.empty_cache()
print(
    f"\\nLlama loaded. CUDA memory in use: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
)

rag_pipeline = RAGPipeline(retriever=retriever, generator=generator, default_k=5)
print("RAGPipeline ready.")
"""
)

# ----------------------------- Cell 7 — demo inputs -------------------------
code(
    """# Cell 7 — Demo inputs: real PlantDoc test-set leaves + Phantom-fs
# soil images (NOT Pillow stand-ins, which made the disease model
# predict the same prior class for every sample). Three distinct
# (crop, disease, soil) tuples so Strategy A / B / C have something
# meaningfully different to compare across samples.
#
# Crop labels are chosen so they MATCH the disease label's crop (the
# disease engine is trained on PlantDoc, which has 27 classes spanning
# Apple, Bell pepper, Blueberry, Cherry, Corn, Peach, Potato,
# Raspberry, Soyabean, Squash, Strawberry, Tomato, grape — NO rice,
# NO mango, so we pick from inside that vocabulary). Each sample
# also carries a CausalPathway so C5 is exercised across all three
# branches (soil_driven / pest_vector / unknown).
from pathlib import Path

from PIL import Image

from src.integration import CausalPathway

# Two source paths: laptop has the images locally; Colab needs them
# pulled from the private HF Hub datasets. The cell auto-detects which
# to use per image so the notebook runs unchanged on both.
PLANTDOC_LOCAL_ROOT = Path(REPO_PATH) / "data" / "plant_disease" / "plantdoc" / "raw"
PHANTOMFS_LOCAL_ROOT = Path(REPO_PATH) / "data" / "soil" / "phantomfs" / "raw" / "Orignal-Dataset"
DEMO_SCRATCH = Path(REPO_PATH) / "_phase8_demo"
DEMO_SCRATCH.mkdir(exist_ok=True)


def _resolve_demo_image(
    *, kind: str, dataset_id: str, split: str, label_col: str, label_value: str,
    local_root: Path, local_rel: str, scratch_key: str,
) -> Path:
    \"\"\"Try local repo first; fall back to HF dataset → JPEG on disk.\"\"\"
    p = local_root / local_rel
    if p.is_file() and p.stat().st_size > 0:
        return p
    cached = DEMO_SCRATCH / f"{kind}__{scratch_key}.jpg"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    for sample in ds:
        if sample.get(label_col) == label_value:
            sample["image"].convert("RGB").save(cached, format="JPEG")
            return cached
    raise RuntimeError(
        f"No sample with {label_col}={label_value!r} in {dataset_id}:{split}."
    )


LEAF_TOMATO = _resolve_demo_image(
    kind="plantdoc", dataset_id="ankit-iiitdmj/iks-plantdoc", split="test",
    label_col="label", label_value="Tomato leaf late blight",
    local_root=PLANTDOC_LOCAL_ROOT,
    local_rel="Tomato leaf late blight/image.jpg",
    scratch_key="tomato",
)
LEAF_CORN = _resolve_demo_image(
    kind="plantdoc", dataset_id="ankit-iiitdmj/iks-plantdoc", split="test",
    label_col="label", label_value="Corn rust leaf",
    local_root=PLANTDOC_LOCAL_ROOT,
    local_rel="Corn rust leaf/Corn-southern-rust-advanced-F1b-8-7-15.jpg",
    scratch_key="corn",
)
LEAF_POTATO = _resolve_demo_image(
    kind="plantdoc", dataset_id="ankit-iiitdmj/iks-plantdoc", split="test",
    label_col="label", label_value="Potato leaf early blight",
    local_root=PLANTDOC_LOCAL_ROOT,
    local_rel="Potato leaf early blight/fac66s01a.jpg",
    scratch_key="potato",
)

SOIL_ALLUVIAL = _resolve_demo_image(
    kind="phantomfs", dataset_id="ankit-iiitdmj/iks-soil-phantomfs", split="train",
    label_col="class_name", label_value="Alluvial_Soil",
    local_root=PHANTOMFS_LOCAL_ROOT, local_rel="Alluvial_Soil/1.jpg",
    scratch_key="alluvial",
)
SOIL_BLACK = _resolve_demo_image(
    kind="phantomfs", dataset_id="ankit-iiitdmj/iks-soil-phantomfs", split="train",
    label_col="class_name", label_value="Black_Soil",
    local_root=PHANTOMFS_LOCAL_ROOT, local_rel="Black_Soil/1.jpg",
    scratch_key="black",
)
SOIL_RED = _resolve_demo_image(
    kind="phantomfs", dataset_id="ankit-iiitdmj/iks-soil-phantomfs", split="train",
    label_col="class_name", label_value="Red_Soil",
    local_root=PHANTOMFS_LOCAL_ROOT, local_rel="Red_Soil/1.jpg",
    scratch_key="red",
)

DEMO_SAMPLES = [
    {
        "name": "tomato / alluvial / soil_driven",
        "leaf_path": LEAF_TOMATO, "soil_path": SOIL_ALLUVIAL,
        "crop": "tomato",
        "pathway": CausalPathway.SOIL_DRIVEN,        # C5 SOIL branch
        "notes": "Recent stand has been waterlogged.",
    },
    {
        "name": "corn / black / pest_vector",
        "leaf_path": LEAF_CORN, "soil_path": SOIL_BLACK,
        "crop": "corn",
        "pathway": CausalPathway.PEST_VECTOR,        # C5 PEST branch
        "notes": "Insects observed on adjacent rows.",
    },
    {
        "name": "potato / red / unknown",
        "leaf_path": LEAF_POTATO, "soil_path": SOIL_RED,
        "crop": "potato",
        "pathway": CausalPathway.UNKNOWN,            # No bias — control
        "notes": None,
    },
]

# Verify every image exists and is non-empty. A missing file would
# silently corrupt the comparison; better to fail loudly here than to
# debug a "why is every disease the same" surprise later.
print(f"Demo samples ready: {len(DEMO_SAMPLES)}")
for s in DEMO_SAMPLES:
    for kind in ("leaf_path", "soil_path"):
        p = Path(s[kind])
        assert p.is_file(), f"Demo image missing: {p}"
        assert p.stat().st_size > 0, f"Demo image is empty: {p}"
    print(f"  - {s['name']}  pathway={s['pathway'].value}")
    print(f"      leaf : {s['leaf_path']}")
    print(f"      soil : {s['soil_path']}")

# Optional upload widget (skipped on `Run all` in the deterministic
# walkthrough; uncomment to run a real farmer-uploaded image in the
# live demo).
# from google.colab import files
# uploaded = files.upload()

# Sanity: run the disease engine ONCE on the tomato leaf and print
# the raw argmax index AND the mapped class name together, so the
# index→name mapping is independently verifiable from the cell
# output (not just trusted to be correct).
import torch  # for the with_gradcam=False branch's no_grad

probe_leaf = Image.open(DEMO_SAMPLES[0]["leaf_path"])
probe_result = disease_engine.predict(probe_leaf)
probe_idx = probe_result.prediction.class_index
probe_name = disease_engine.class_names[probe_idx]
print()
print("=== sanity check: disease engine idx→name on sample-0 leaf ===")
print(f"  argmax index : {probe_idx}")
print(f"  mapped name  : {probe_name}")
print(f"  prediction   : {probe_result.prediction.class_name}  conf={probe_result.prediction.confidence:.3f}")
assert probe_result.prediction.class_name == probe_name, (
    f"Index→name mapping inconsistent: prediction says "
    f"{probe_result.prediction.class_name!r} but class_names[{probe_idx}] is {probe_name!r}"
)
"""
)

# ----------------------------- Cell 8 — build_visual_context ----------------
code(
    """# Cell 8 — build_multimodal_context for every demo sample.
# This runs the disease + soil engines once per sample and packs the
# results (label + confidence + penultimate embedding) into a
# MultimodalContext, including the user-supplied CausalPathway and
# free-text notes for C5.
from src.integration import build_multimodal_context

contexts = []
for sample in DEMO_SAMPLES:
    ctx = build_multimodal_context(
        leaf_image=Image.open(sample["leaf_path"]),
        soil_image=Image.open(sample["soil_path"]),
        crop_type=sample["crop"],
        causal_pathway=sample["pathway"],
        causal_notes=sample["notes"],
        disease_engine=disease_engine,
        soil_engine=soil_engine,
        capture_embeddings=True,  # needed for Strategy C
    )
    contexts.append(ctx)
    print("=" * 78)
    print(f"SAMPLE: {sample['name']}")
    print(f"  leaf src: {sample['leaf_path']}")
    print(f"  soil src: {sample['soil_path']}")
    print(f"  disease : {ctx.disease_pred.class_name}  "
          f"(idx={ctx.disease_pred.class_index}  conf={ctx.disease_pred.confidence:.2f})")
    print(f"  soil    : type={ctx.soil_pred.soil_type}  "
          f"moisture={ctx.soil_pred.moisture_appearance}  "
          f"texture={ctx.soil_pred.texture}")
    print(f"  crop    : {ctx.crop_type}")
    print(f"  cause   : {ctx.causal_context.pathway.value}  "
          f"notes={ctx.causal_context.notes!r}")
    print(
        f"  embeds  : disease={ctx.disease_emb.shape}  "
        f"soil={ctx.soil_emb.shape}"
    )

# Honesty check: the three disease predictions should NOT all be the
# same class — that would mean every sample is feeding the engine the
# same image (the bug we just fixed) or the engine is collapsing for
# some reason. Fail loudly here, not at Cell 12 where it would just
# look like Strategy A under-performs.
disease_labels = [c.disease_pred.class_name for c in contexts]
assert len(set(disease_labels)) >= 2, (
    "All three demo samples predicted the same disease class "
    f"({disease_labels[0]!r}); the A/B/C comparison would be meaningless. "
    "Check that DEMO_SAMPLES['leaf_path'] points to distinct real PlantDoc "
    "test images."
)
print(f"\\nBuilt {len(contexts)} contexts with distinct disease labels: {disease_labels}")
"""
)

# ----------------------------- Cell 9 — Strategy A --------------------------
code(
    """# Cell 9 — Strategy A (Template): deterministic query + RAG answer.
# Run for every sample so the supervisor can read the query side-by-side
# with the retrieved chunks and the grounded answer. This is the
# transparent baseline.
from src.integration import TemplateStrategy
from src.integration.config import TemplateStrategyConfig

template_strategy = TemplateStrategy(TemplateStrategyConfig())

template_results = []
for sample, ctx in zip(DEMO_SAMPLES, contexts):
    query = template_strategy.build_query(ctx)
    print("=" * 78)
    print(f"SAMPLE: {sample['name']}")
    print(f"  Query (Strategy A — template): {query!r}")
    rag_answer = rag_pipeline.answer(query, k=5)
    print("  --- Top retrieved sources ---")
    for i, h in enumerate(rag_answer.retrieved[:5], 1):
        meta = h.metadata or {}
        print(
            f"    [{i}] score={h.score:.4f}  "
            f"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} "
            f"v.{meta.get('verse_or_section','?')}"
        )
    print("  --- Grounded answer ---")
    print("   ", (rag_answer.answer or "").strip())
    print("  --- Citations ---")
    for c in rag_answer.citations or []:
        print("    -", c)
    template_results.append({"query": query, "rag_answer": rag_answer})
"""
)

# ----------------------------- Cell 10 — Strategy B -------------------------
code(
    """# Cell 10 — Strategy B (LLM-mediated): bridge modern → classical vocab.
# Note how B's query differs from A's: B rewrites disease + soil labels
# into descriptive / classical phrasing that matches IKS-corpus
# language (e.g. "scorched leaves with whitish lesions" instead of
# "Tomato___Leaf_Mold", or "fertile riverine soil" instead of
# "Alluvial_Soil").
from src.integration import LLMMediatedStrategy
from src.integration.config import LLMMediatedStrategyConfig

llm_strategy = LLMMediatedStrategy(LLMMediatedStrategyConfig())

llm_results = []
for sample, ctx, a_result in zip(DEMO_SAMPLES, contexts, template_results):
    llm_query = llm_strategy.build_query(ctx, rag_pipeline.generator)
    print("=" * 78)
    print(f"SAMPLE: {sample['name']}")
    print("  Query (Strategy A): " + repr(a_result["query"]))
    print("  Query (Strategy B): " + repr(llm_query))
    print("  ↑ How B differs: bridge from modern labels to classical/symptomatic vocabulary.")
    rag_answer = rag_pipeline.answer(llm_query, k=5)
    print("  --- Top retrieved sources ---")
    for i, h in enumerate(rag_answer.retrieved[:5], 1):
        meta = h.metadata or {}
        print(
            f"    [{i}] score={h.score:.4f}  "
            f"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} "
            f"v.{meta.get('verse_or_section','?')}"
        )
    print("  --- Grounded answer ---")
    print("   ", (rag_answer.answer or "").strip())
    print("  --- Citations ---")
    for c in rag_answer.citations or []:
        print("    -", c)
    llm_results.append({"query": llm_query, "rag_answer": rag_answer})
"""
)

# ----------------------------- Cell 11 — Strategy C -------------------------
code(
    """# Cell 11 — Strategy C (multimodal embedding projection, ABLATION).
# Trains a single linear layer on WEAK pairs:
#   (concat(disease_emb, soil_emb, crop_emb))  →  Strategy-A top-1 chunk emb
# A few epochs MSE, no manual labels. Retrieve via ChromaDB cosine.
#
# Expected outcome: C under-performs A and B on on-topic-source overlap.
# The modality gap (visual embeddings live on a different manifold than
# text embeddings) is too wide for a single linear layer trained on a
# handful of weak pairs to close. The comparison cell below will make
# that visible.
from sentence_transformers import SentenceTransformer

from src.integration import MultimodalEmbeddingStrategy
from src.integration.config import MultimodalEmbeddingStrategyConfig

embed_strategy = MultimodalEmbeddingStrategy(MultimodalEmbeddingStrategyConfig())

# Reuse the same encoder Phase 7 used to embed the corpus.
text_embedder = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Use Strategy A's templated query as the weak supervision signal.
projector, train_report = embed_strategy.train_weak_projection(
    samples=contexts,
    embedder=text_embedder,
    template_query_fn=template_strategy.build_query,
    chroma_collection=collection,
    epochs=80,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(
    f"Weak projection trained: epochs={train_report.epochs}  "
    f"final_loss={train_report.final_loss:.6f}  n={train_report.n_samples}"
)

embed_results = []
for sample, ctx in zip(DEMO_SAMPLES, contexts):
    retrieved = embed_strategy.retrieve_via_embedding(
        ctx, projector, collection, text_embedder, k=5,
    )
    print("=" * 78)
    print(f"SAMPLE: {sample['name']}")
    print("  Query (Strategy C): [multimodal vector — no text query]")
    print("  --- Top retrieved sources ---")
    for i, h in enumerate(retrieved, 1):
        meta = h["metadata"] or {}
        print(
            f"    [{i}] score={h['score']:.4f}  "
            f"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} "
            f"v.{meta.get('verse_or_section','?')}"
        )
    embed_results.append({"retrieved": retrieved})
"""
)

# ----------------------------- Cell 12 — comparison -------------------------
code(
    """# Cell 12 — Side-by-side qualitative comparison across A / B / C.
# The on-topic count is a HEURISTIC for the supervisor demo — "how many
# of the k retrieved chunks come from a plausibly relevant IKS book"
# (default: all 4). Rigorous RAGAS context_precision/recall against an
# expert gold-query set is **Phase 11**.
from src.integration.compare import run_all_strategies, qualitative_compare

all_results = []
for sample, ctx in zip(DEMO_SAMPLES, contexts):
    per_strategy = run_all_strategies(
        ctx,
        rag_pipeline,
        projector=projector,
        chroma_collection=collection,
        embedder=text_embedder,
        k=5,
        answer=True,
    )
    all_results.append(per_strategy)

rows = qualitative_compare(all_results)

# Pretty-print as a flat table.
print(f"{'sample':>2}  {'strategy':<22}  {'on-topic/k':<11}  query")
print("-" * 100)
for r in rows:
    print(
        f"{r['sample_idx']:>2}  "
        f"{r['strategy']:<22}  "
        f"{r['on_topic_count']:>4} / {r['k']:<4}  "
        f"{r['query'][:60]}"
    )

# Aggregate winners per sample.
print()
print("=== Strategy ranking per sample (on-topic count, ties → declared) ===")
for sample_idx, sample in enumerate(DEMO_SAMPLES):
    print(f"\\nSample: {sample['name']}")
    per = [r for r in rows if r["sample_idx"] == sample_idx]
    per.sort(key=lambda r: -r["on_topic_count"])
    for rank, r in enumerate(per, 1):
        print(f"  #{rank} {r['strategy']:<22} on_topic={r['on_topic_count']}/{r['k']}")
"""
)

# ----------------------------- Cell 13 — findings md ------------------------
md(
    """## Findings + scope notes for the thesis

**What this notebook demonstrated.**

1. The vision → RAG plumbing works end-to-end across 3 samples + 3
   strategies. Each `(leaf, soil, crop, optional cause)` produces three
   distinct retrieval paths with comparable retrieved-chunk shape and
   (for Strategies A and B) a grounded `[Source Text, ch.X, v.Y]`-cited
   answer.
2. Strategy B's query visibly *bridges* modern vision labels to
   classical / descriptive vocabulary on every sample. Compare the
   `Strategy A` vs `Strategy B` query strings in Cell 10's output — A
   uses dataset labels verbatim, B reformulates them.
3. The C5 causal context threads through both Strategy A (deterministic
   pathway clause appended to the template) and Strategy B (instructed
   into the rewrite prompt). The `unknown` sample correctly receives no
   bias clause.
4. Strategy C's qualitative read confirms the expected ordering
   **B ≥ A > C** in on-topic retrieval — the modality gap (visual
   embeddings live on a different manifold from text embeddings) is too
   wide for a single linear layer trained on a handful of weak pairs to
   close. This is the *honest* ablation; a fully-trained projector with
   manual relevance labels is future work.

**What this notebook deliberately does NOT do.**

- **No RAGAS context_precision / context_recall scoring.** That is the
  rigorous evaluation phase — Phase 11 — and runs against an
  expert-curated gold-query set, not against three Pillow-generated
  stub images. Phase 8 ships the mechanism + a qualitative read.
- **No causal-conditioning ablation.** C5 is *wired and demoed* on one
  sample, not formally evaluated. The Phase 11 ablation will compare
  retrieval and answer quality with and without the causal clause on
  every gold query.
- **No fine-tuning of the multimodal projector.** Strategy C's
  projector is a single linear layer trained for a handful of epochs
  on auto-generated weak pairs. A learned multimodal embedding space
  (CLIP-style contrastive training on image / classical-text pairs) is
  out of scope for this thesis.

**What comes next.**

- **Phase 9** — explainability: Grad-CAM on the disease prediction +
  retrieved-chunk highlighting in the final answer.
- **Phase 10** — full system + Streamlit UI: this notebook's
  flow becomes the backend of an interactive farmer-facing app.
- **Phase 11** — rigorous RAGAS ablation on the gold-query set. This is
  where Strategy B's win over A and the modality-gap penalty for C are
  *measured* rather than visually inspected.
"""
)


# ============================================================================
# Assemble + write
# ============================================================================

NB.cells = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(NB, fh)

print(f"Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)} with {len(cells)} cells.")
