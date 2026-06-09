"""Generate ``notebooks/phase9_explainability.ipynb`` (Phase 9 §C).

12 cells per the Phase 9 prompt's locked structure. Same builder
pattern as ``build_phase7_notebook.py`` and ``build_phase8_notebook.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase9_explainability.ipynb"

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
    """# Phase 9 — Explainability Layer (Grad-CAM + retrieved-chunk highlighting)

Phase 9 (master plan §18) adds the **honest interpretability** claim
the paper relies on (§35). Two surfaces:

1. **Vision Grad-CAM** — *where* each model looked. One heatmap for the
   disease classifier (Phase 5 EfficientNet-B4) and three heatmaps for
   the multi-task soil model (Phase 6 EfficientNet-B0) — one per head
   (``soil_type``, ``moisture``, ``texture``). The soil model's
   multi-output forward returns a dict, which breaks
   ``pytorch_grad_cam``'s ``ClassifierOutputTarget`` directly, so
   each head is wrapped via :class:`~src.explain.gradcam.SoilHeadWrapper`
   to expose a single-tensor forward.
2. **Retrieved-chunk highlighting** — *why* each chunk was retrieved.
   Pure lexical query↔chunk term overlap (no second model), wrapped
   in ``**…**`` Markdown markers so the Streamlit / notebook caller
   can render them in any colour scheme. The on-topic-overlap count
   from Phase 8 was a coarse "did at least one of the 4 books show
   up?" check; Phase 9's matched terms tell the supervisor "the
   retriever picked this chunk because both the query and the chunk
   mention *kunapajala* and *tree*".

## Why both matter for the paper

- Grad-CAM defends the vision modelling against the "you might just be
  memorising background pixels" objection — heatmaps on diseased leaves
  should attend to the lesion, not the corner. This is how the soil
  multi-task choice is also defensible: the same backbone routes
  attention differently for each head.
- Chunk highlighting defends the retrieval choice. The RAG generator
  cites ``[Source Text, ch.X, v.Y]`` (master plan §17), but a citation
  without provenance is just a label. The retrieval panel attaches
  the *evidence* (which query terms triggered which chunk + how
  similar) directly to each citation.

## Scope notes (deliberately deferred)

- **No rigorous interpretability benchmark.** Pointing-game on
  segmentation masks, faithfulness sweeps, etc. are Phase 11. Phase 9
  ships the surfaces.
- **Retrieval explanation is lexical, not answer-grounded.** Aligning
  the chunk to the LLM's *answer* sentences would mix retrieval and
  generation failure modes — Phase 9 isolates retrieval; Phase 11
  brings in RAGAS faithfulness for the generation side.

## Hard rules (master plan §16)

- Local commits only — never `git push`.
- Models / corpus are read-only here. The Phase 5 / 6 / 7 / 8 modules
  are imported, not reimplemented.
"""
)

# ----------------------------- Cell 2 — setup -------------------------------
code(
    f"""# Cell 2 — clone repo + install dependencies (defensive)
import os
import subprocess
import sys

REPO_URL = "{REPO_HTTPS_URL}"
REPO_PATH = "{REPO_LOCAL_PATH}"

if not os.path.exists(REPO_PATH):
    subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
else:
    subprocess.run(["git", "-C", REPO_PATH, "pull", "--ff-only"], check=True)

os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)
print(f"Working directory: {{os.getcwd()}}")

# Phase 9 adds grad-cam + matplotlib on top of the Phase 7/8 dep set.
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
    "grad-cam>=1.5",
    "matplotlib>=3.7",
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

info = HfApi().whoami()
print(f"Logged in as: {info.get('name')}")
assert info.get("name") == "ankit-iiitdmj", (
    "HF token belongs to a different user — Phase 9 needs the same private "
    "datasets and gated model access used in Phase 7/8."
)
"""
)

# ----------------------------- Cell 4 — GPU check ---------------------------
code(
    """# Cell 4 — GPU + CUDA sanity check
import torch

print(f"torch: {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
else:
    print("WARNING: no GPU. Grad-CAM itself is light, but the Phase 7 Llama is")
    print("not loadable on CPU; the retrieval explanation cells need the LLM,")
    print("so switch the runtime to T4 before continuing.")
"""
)

# ----------------------------- Cell 5 — vision models -----------------------
code(
    """# Cell 5 — Load Phase 5 disease (B4) + Phase 6 soil (B0 multi-task) engines.
# Both engines load from HF Hub model repos and ship in FULL precision —
# only Llama is 4-bit quantised. Grad-CAM needs gradients to flow
# through the conv layers; quantised weights would break that, so the
# vision modules deliberately stay in fp16/fp32.
import torch

from src.disease.infer import DiseaseInferenceEngine
from src.soil.infer import SoilInferenceEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

disease_engine = DiseaseInferenceEngine(
    model_source="ankit-iiitdmj/iks-disease-plantdoc",
    device=DEVICE,
)
print(f"Disease engine: {disease_engine.num_classes} classes on {disease_engine.device}")
print(f"  first 5 names : {disease_engine.class_names[:5]}")
assert not any(n.startswith("class_") and n[6:].isdigit() for n in disease_engine.class_names), (
    "Disease engine class names still look like 'class_<i>' placeholders. "
    "Check data/splits/plantdoc/class_map.json."
)

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
    """# Cell 6 — Phase 7 RAG pipeline. Same wiring as Phase 8 Cell 6:
# corpus pulled from the private HF dataset (206 chunks across 4
# books, Gemini re-OCR'd in Phase 3b.2), re-embedded into ChromaDB,
# wrapped in the HybridRetriever (dense + sparse + reranker), and
# composed with a Llama-3.1-8B 4-bit grounded generator.
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
    f"Corpus drift: expected {EXPECTED_CHUNK_COUNT} chunks, got {len(chunks)}."
)
for book, expected in EXPECTED_PER_BOOK.items():
    assert per_book.get(book) == expected, (
        f"Per-book drift: {book} expected {expected}, got {per_book.get(book)}"
    )

collection = build_chroma(chunks, persist_dir="corpus/vector_db")
print(f"ChromaDB ready: collection count = {collection.count()}")
retriever = HybridRetriever(collection, use_dense=True, use_sparse=True, use_reranker=True)
generator = GroundedGenerator(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    temperature=0.2,
    max_new_tokens=512,
    seed=42,
)
generator._ensure_loaded()
torch.cuda.empty_cache()
print(f"Llama loaded. CUDA memory in use: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

rag_pipeline = RAGPipeline(retriever=retriever, generator=generator, default_k=5)
print("RAGPipeline ready.")
"""
)

# ----------------------------- Cell 7 — demo inputs -------------------------
code(
    """# Cell 7 — Demo inputs: REAL distinct PlantDoc + Phantomfs images.
# The Phase 8 Cell-7 bug (Pillow stand-ins → every sample predicting
# the same prior class) made Grad-CAM heatmaps explain a WRONG label.
# Phase 9 reuses the same three real images and refuses to proceed if
# any path is missing — a Grad-CAM over a wrong-disease placeholder
# is a misleading figure.
#
# Two source paths:
#   - LOCAL  : repo's data/plant_disease/ and data/soil/ trees (the
#              laptop has them; they're gitignored so Colab does NOT).
#   - HF Hub : private datasets ankit-iiitdmj/iks-plantdoc and
#              ankit-iiitdmj/iks-soil-phantomfs. Same images, served
#              from Parquet. The cell auto-detects which to use.
from pathlib import Path

from PIL import Image

from src.integration import CausalPathway

PLANTDOC_LOCAL_ROOT = Path(REPO_PATH) / "data" / "plant_disease" / "plantdoc" / "raw"
PHANTOMFS_LOCAL_ROOT = Path(REPO_PATH) / "data" / "soil" / "phantomfs" / "raw" / "Orignal-Dataset"
DEMO_SCRATCH = Path(REPO_PATH) / "_phase9_demo"
DEMO_SCRATCH.mkdir(exist_ok=True)

# Target (label, local-fallback-file, dataset-source) tuples per sample.
PLANTDOC_TARGETS = {
    "tomato_leaf": (
        "Tomato leaf late blight",
        "Tomato leaf late blight/image.jpg",
    ),
    "corn_leaf": (
        "Corn rust leaf",
        "Corn rust leaf/Corn-southern-rust-advanced-F1b-8-7-15.jpg",
    ),
    "potato_leaf": (
        "Potato leaf early blight",
        "Potato leaf early blight/fac66s01a.jpg",
    ),
}
PHANTOMFS_TARGETS = {
    # HF dataset's class_name column drops the "_Soil" suffix; the
    # local raw folder keeps it. Both are listed per target.
    "alluvial_soil": ("Alluvial", "Alluvial_Soil/1.jpg"),
    "black_soil":   ("Black",   "Black_Soil/1.jpg"),
    "red_soil":     ("Red",     "Red_Soil/1.jpg"),
}


def _resolve_local(root: Path, rel: str) -> Path | None:
    \"\"\"Return the local copy if present and non-empty, else None.\"\"\"
    p = root / rel
    if p.is_file() and p.stat().st_size > 0:
        return p
    return None


def _fetch_from_hf(
    dataset_id: str, split: str, label_col: str, label_value: str,
    out_path: Path,
) -> Path:
    \"\"\"Download the first sample with matching label and save as JPG.

    Cached after first run via the HF datasets library — subsequent
    re-runs reuse the cached parquet, no re-download.\"\"\"
    from datasets import load_dataset

    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    ds = load_dataset(dataset_id, split=split)
    for sample in ds:
        if sample.get(label_col) == label_value:
            sample["image"].convert("RGB").save(out_path, format="JPEG")
            return out_path
    raise RuntimeError(
        f"No sample with {label_col}={label_value!r} found in "
        f"{dataset_id}:{split}."
    )


def _resolve_plantdoc(name: str) -> Path:
    label, rel = PLANTDOC_TARGETS[name]
    local = _resolve_local(PLANTDOC_LOCAL_ROOT, rel)
    if local is not None:
        return local
    out = DEMO_SCRATCH / f"plantdoc__{name}.jpg"
    return _fetch_from_hf(
        "ankit-iiitdmj/iks-plantdoc", "test", "label", label, out,
    )


def _resolve_phantomfs(name: str) -> Path:
    label, rel = PHANTOMFS_TARGETS[name]
    local = _resolve_local(PHANTOMFS_LOCAL_ROOT, rel)
    if local is not None:
        return local
    out = DEMO_SCRATCH / f"phantomfs__{name}.jpg"
    return _fetch_from_hf(
        "ankit-iiitdmj/iks-soil-phantomfs", "train", "class_name", label, out,
    )


print("=== Resolving demo image sources (local repo → HF Hub fallback) ===")
DEMO_SAMPLES = [
    {
        "name": "tomato_alluvial_soil_driven",
        "leaf_path": _resolve_plantdoc("tomato_leaf"),
        "soil_path": _resolve_phantomfs("alluvial_soil"),
        "crop": "tomato",
        "pathway": CausalPathway.SOIL_DRIVEN,
    },
    {
        "name": "corn_black_pest_vector",
        "leaf_path": _resolve_plantdoc("corn_leaf"),
        "soil_path": _resolve_phantomfs("black_soil"),
        "crop": "corn",
        "pathway": CausalPathway.PEST_VECTOR,
    },
    {
        "name": "potato_red_unknown",
        "leaf_path": _resolve_plantdoc("potato_leaf"),
        "soil_path": _resolve_phantomfs("red_soil"),
        "crop": "potato",
        "pathway": CausalPathway.UNKNOWN,
    },
]

print()
print("=== Demo sample sources + predicted disease names ===")
sample_disease_names = []
for s in DEMO_SAMPLES:
    for kind in ("leaf_path", "soil_path"):
        p = Path(s[kind])
        assert p.is_file(), (
            f"Demo image missing: {p}  "
            f"(Phase 9 refuses to render Grad-CAM over a stand-in)"
        )
        assert p.stat().st_size > 0, f"Demo image is empty: {p}"
    # Predict per-sample so the supervisor can see distinct labels.
    pred = disease_engine.predict(Image.open(s["leaf_path"])).prediction
    sample_disease_names.append(pred.class_name)
    print(f"- {s['name']}")
    print(f"    leaf src     : {s['leaf_path']}")
    print(f"    soil src     : {s['soil_path']}")
    print(f"    crop         : {s['crop']}")
    print(f"    pred disease : {pred.class_name}  (idx={pred.class_index}  conf={pred.confidence:.3f})")

assert len(set(sample_disease_names)) >= 2, (
    "All three samples predicted the same disease class "
    f"{sample_disease_names!r} — Grad-CAM figures would explain the same "
    "label three times. Pick distinct PlantDoc test images that the model "
    "actually attends to differently."
)
print(f"\\nSample disease labels are distinct: {sample_disease_names}")
"""
)

# ----------------------------- Cell 8 — disease Grad-CAM --------------------
code(
    """# Cell 8 — Disease Grad-CAM per sample.
# `disease_gradcam` runs the full-precision B4 backbone WITH gradients
# enabled, picks the predicted class as the Grad-CAM target, and
# returns an overlay (uint8 H×W×3), the raw heatmap (float32 H×W in
# [0,1]), the human-readable label, and the raw argmax index (so the
# index→name table is auditable per-sample).
import matplotlib.pyplot as plt
from PIL import Image

from src.explain.gradcam import disease_gradcam

disease_cams: dict[str, "GradCAMResult"] = {}

fig, axes = plt.subplots(len(DEMO_SAMPLES), 2, figsize=(8, 4 * len(DEMO_SAMPLES)))
if len(DEMO_SAMPLES) == 1:
    axes = axes.reshape(1, 2)

for row, sample in enumerate(DEMO_SAMPLES):
    cam = disease_gradcam(sample["leaf_path"], disease_engine)
    disease_cams[sample["name"]] = cam
    leaf = Image.open(sample["leaf_path"]).convert("RGB")
    leaf = leaf.resize((disease_engine.image_size, disease_engine.image_size))
    axes[row, 0].imshow(leaf)
    axes[row, 0].set_title(f"{sample['name']}\\n(original)")
    axes[row, 0].axis("off")
    axes[row, 1].imshow(cam.overlay_rgb)
    axes[row, 1].set_title(
        f"disease Grad-CAM\\n{cam.pred_label} (idx={cam.pred_index}, conf={cam.pred_conf:.2f})"
    )
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()
print(f"\\nGenerated {len(disease_cams)} disease Grad-CAM overlays.")
"""
)

# ----------------------------- Cell 9 — soil Grad-CAM x3 --------------------
code(
    """# Cell 9 — Soil Grad-CAM × 3 heads per sample.
# Each head is wrapped via SoilHeadWrapper so pytorch_grad_cam sees a
# single-tensor forward. soil_gradcam runs the model in eval mode,
# pulls the per-head argmax label from the engine (so the explanation
# targets the SAME label the rest of the pipeline saw), and returns
# the overlay + heatmap + label/conf.
import matplotlib.pyplot as plt
from PIL import Image

from src.explain.gradcam import SOIL_HEADS, soil_gradcam

soil_cams_per_sample: dict[str, dict] = {}

for sample in DEMO_SAMPLES:
    print(f"--- {sample['name']} ---")
    cams = {}
    for head in SOIL_HEADS:
        cam = soil_gradcam(sample["soil_path"], soil_engine, head=head)
        cams[head] = cam
        print(f"  {head:<10} idx={cam.pred_index}  label={cam.pred_label!r}  conf={cam.pred_conf:.2f}")
    soil_cams_per_sample[sample["name"]] = cams

    # Render the 4-tile row: original soil + 3 head heatmaps.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    soil_img = Image.open(sample["soil_path"]).convert("RGB")
    soil_img = soil_img.resize((soil_engine.config.image_size, soil_engine.config.image_size))
    axes[0].imshow(soil_img)
    axes[0].set_title(f"{sample['name']} (original)")
    axes[0].axis("off")
    for col, head in enumerate(SOIL_HEADS, start=1):
        cam = cams[head]
        axes[col].imshow(cam.overlay_rgb)
        axes[col].set_title(f"{head}\\n{cam.pred_label} ({cam.pred_conf:.2f})")
        axes[col].axis("off")
    plt.tight_layout()
    plt.show()

print(f"\\nGenerated {sum(len(c) for c in soil_cams_per_sample.values())} soil-head Grad-CAM overlays.")
"""
)

# ----------------------------- Cell 10 — retrieval explanation --------------
code(
    """# Cell 10 — Build the query (Phase 8 Strategy A) and explain top-k retrieval.
# Per sample: build the multimodal context, render Strategy A's
# template query, retrieve top-5, then expand each chunk with the
# matched-query-term overlay.
from PIL import Image

from src.explain.chunk_highlight import explain_chunks
from src.integration import (
    TemplateStrategy,
    build_multimodal_context,
)
from src.integration.config import TemplateStrategyConfig

template_strategy = TemplateStrategy(TemplateStrategyConfig())

retrieval_per_sample: dict[str, dict] = {}

for sample in DEMO_SAMPLES:
    ctx = build_multimodal_context(
        leaf_image=Image.open(sample["leaf_path"]),
        soil_image=Image.open(sample["soil_path"]),
        crop_type=sample["crop"],
        causal_pathway=sample["pathway"],
        disease_engine=disease_engine,
        soil_engine=soil_engine,
        capture_embeddings=False,    # Strategy C is NOT exercised here.
    )
    query = template_strategy.build_query(ctx)
    retrieved = rag_pipeline.retriever.retrieve(query, k=5)
    explained = explain_chunks(query, retrieved)
    retrieval_per_sample[sample["name"]] = {
        "query": query, "retrieved": retrieved, "explained": explained,
    }
    print("=" * 78)
    print(f"SAMPLE: {sample['name']}")
    print(f"  QUERY: {query!r}")
    for row in explained:
        print(
            f"  #{row.rank}  score={row.score:.3f}  "
            f"{row.source_text} ch.{row.chapter} v.{row.verse_or_section}"
        )
        print(f"      matched: {row.matched_terms or '(no overlap)'}")
"""
)

# ----------------------------- Cell 11 — combined figure save ---------------
code(
    """# Cell 11 — Combined figures saved to results/explainability/<sample>/
# These PNGs are what the Phase 10 Streamlit UI will surface and the
# paper figures will reuse. The directory is committed to the repo
# (results/ is for tracked PNGs / metrics per master plan §41).
from pathlib import Path

import numpy as np
from PIL import Image

from src.explain.visualize import (
    DEFAULT_OUT_ROOT,
    render_retrieval_panel,
    render_vision_panel,
    save_explanation,
)

saved_paths: list[tuple[Path, Path]] = []

for sample in DEMO_SAMPLES:
    name = sample["name"]
    disease_cam = disease_cams[name]
    soil_cams = soil_cams_per_sample[name]

    leaf = Image.open(sample["leaf_path"]).convert("RGB")
    leaf = leaf.resize((disease_engine.image_size, disease_engine.image_size))
    leaf_rgb = np.asarray(leaf, dtype=np.uint8)

    soil = Image.open(sample["soil_path"]).convert("RGB")
    soil = soil.resize((soil_engine.config.image_size, soil_engine.config.image_size))
    soil_rgb = np.asarray(soil, dtype=np.uint8)

    vision_fig = render_vision_panel(
        sample_name=name,
        original_leaf=leaf_rgb,
        disease_cam=disease_cam,
        original_soil=soil_rgb,
        soil_cams=soil_cams,
    )
    retrieval_fig = render_retrieval_panel(
        sample_name=name,
        query=retrieval_per_sample[name]["query"],
        explained_chunks=retrieval_per_sample[name]["explained"],
    )
    paths = save_explanation(
        sample_name=name,
        vision_fig=vision_fig,
        retrieval_fig=retrieval_fig,
    )
    saved_paths.append(paths)
    print(f"{name:<40s} → {paths[0].name}, {paths[1].name}")

print()
print(f"All figures saved under {DEFAULT_OUT_ROOT.relative_to(REPO_PATH)}")
"""
)

# ----------------------------- Cell 12 — findings md ------------------------
md(
    """## What the figures should show + what's deferred

**What to look for in the saved panels**

- ``results/explainability/<sample>/vision_panel.png`` — the disease
  heatmap should concentrate over the lesion / discoloured region
  (NOT the corner or the background). The three soil-head heatmaps
  should attend to different regions: ``soil_type`` typically focuses
  on the colour-rich area; ``moisture`` on glossier / damper-looking
  patches; ``texture`` on the granular surface itself. If all three
  soil-head heatmaps look identical, the multi-task head is
  collapsing — flag for retraining (out of Phase 9 scope).
- ``results/explainability/<sample>/retrieval_panel.png`` — the
  similarity bar chart should be monotone descending (it's the
  retrieved top-k by score). The matched terms next to each chunk
  tell the supervisor *why* the retriever picked the chunk; chunks
  with ``(no overlap)`` are pure dense-retrieval hits — those are
  expected and not a bug, but worth flagging when they dominate.

**What this notebook deliberately does NOT do**

- **Pointing-game / faithfulness benchmarks.** Quantitative
  Grad-CAM evaluation (how well does the heatmap localise the GT
  lesion mask?) needs segmentation labels we don't have. Phase 11
  will revisit this with the gold-query set.
- **Answer-grounded chunk highlighting.** Phase 9 explains the
  retrieval step only; aligning chunks to the LLM's *answer*
  sentences mixes retrieval and generation failure modes. The
  generation-side audit comes via RAGAS faithfulness in Phase 11.
- **Sentence-level chunk highlighting inside the answer.** Possible
  to add later, but the demo gets more mileage out of the
  retrieval panel + Grad-CAM than a third explanation surface.

**What comes next**

- **Phase 10** — Streamlit UI that surfaces the Grad-CAM panels and
  the retrieval panel live. The figures rendered here are the
  components that UI will display; the saved PNGs are the paper's
  reference figures.
- **Phase 11** — rigorous evaluation: RAGAS context_precision /
  context_recall on the gold-query set, faithfulness on the
  generated answers, and a quantitative Grad-CAM pointing-game if a
  segmentation mask subset of PlantDoc becomes available.
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
