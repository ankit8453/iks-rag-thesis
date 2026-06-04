"""Generate ``notebooks/phase7_rag_pipeline.ipynb`` (Phase 7 §C).

One-shot builder for the Phase 7 Colab notebook that wires the
HF-Hub-hosted corpus chunks through the hybrid retriever and the
Llama-3.1-8B 4-bit grounded generator. 12 cells per the prompt's
locked structure.

Cell 8 (retriever smoke) deliberately runs BEFORE the LLM loads so a
retrieval failure fails fast — the model download is the most
expensive step and should not happen behind a broken retriever.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase7_rag_pipeline.ipynb"

REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

NB = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# --------------------------------------------------------------------- #
# Cell 1 — Title + goal + design notes
# --------------------------------------------------------------------- #
md(
    "# Phase 7 — Grounded RAG Pipeline (Colab)\n"
    "\n"
    "Hybrid retrieval + Llama-3.1-8B-Instruct 4-bit grounded generator over\n"
    "the Phase 3 IKS corpus (285 chunks: 78 Vrikshayurveda + 207 Brihat\n"
    "Samhita 12-chapter subset).\n"
    "\n"
    "## Design (master plan §17)\n"
    "\n"
    "- **Retrieval:** dense (BAAI/bge-large-en-v1.5 over ChromaDB) +\n"
    "  sparse (BM25) → Reciprocal-Rank-Fusion (k=60) → cross-encoder\n"
    "  rerank (BAAI/bge-reranker-base). All three stages are toggleable\n"
    "  for Phase 11 §27 ablations.\n"
    "- **Generation:** Llama-3.1-8B-Instruct in 4-bit (nf4 + double-quant\n"
    "  + bf16 compute). Master plan §17 grounded-advisor system prompt:\n"
    "  answer ONLY from retrieved passages, cite source + chapter + verse,\n"
    "  step-by-step organic protocol, refuse out-of-corpus questions.\n"
    "- **Corpus transport:** chunks live in the private HF dataset\n"
    "  `ankit-iiitdmj/iks-corpus-chunks` (the laptop's `corpus/vector_db/`\n"
    "  cannot reach Colab). This notebook rebuilds ChromaDB in-session\n"
    "  from the dataset.\n"
    "\n"
    "## Platform\n"
    "\n"
    "Phase 7 runs on Colab (Linux). On Windows, `chromadb` and `torch` /\n"
    "`sentence_transformers` segfault in the same Python process — that's\n"
    "captured in the memory entry `feedback-chromadb-torch-windows-dll`.\n"
    "On Linux the single-process design is fine.\n"
    "\n"
    "## ⚠️ Before you start\n"
    "\n"
    "- **Runtime:** GPU (T4 free tier is enough for 4-bit 8B; expect\n"
    "  ~5–6 GB VRAM).\n"
    "- **HF Hub token:** Write token belonging to `ankit-iiitdmj` (needed\n"
    "  for the private chunks dataset AND the gated Llama-3.1 weights).\n"
    "- **Llama-3.1 license:** you must have accepted it at\n"
    "  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before\n"
    "  Cell 9 runs. If you haven't, Cell 9 will fail with a 403 on the\n"
    "  weights download.\n"
)

# --------------------------------------------------------------------- #
# Cell 2 — Setup
# --------------------------------------------------------------------- #
code(
    "# Cell 2 — setup: clone repo + install dependencies (defensive)\n"
    f"REPO_URL = \"{REPO_HTTPS_URL}\"\n"
    f"REPO_PATH = \"{REPO_LOCAL_PATH}\"\n"
    "\n"
    "import os, subprocess, sys\n"
    "\n"
    "if os.path.isdir(REPO_PATH) and not os.path.isfile(os.path.join(REPO_PATH, \"requirements.txt\")):\n"
    "    print(f\"Removing partial clone at {REPO_PATH} ...\")\n"
    "    subprocess.run([\"rm\", \"-rf\", REPO_PATH], check=True)\n"
    "if not os.path.isdir(REPO_PATH):\n"
    "    subprocess.run([\"git\", \"clone\", REPO_URL, REPO_PATH], check=True)\n"
    "\n"
    "os.chdir(REPO_PATH)\n"
    "print(\"Repo root contents:\", sorted(os.listdir(\".\")))\n"
    "\n"
    "# Phase 7 runtime deps. Colab pre-installs torch / numpy / pandas /\n"
    "# sklearn so we skip those. `bitsandbytes` is needed for 4-bit\n"
    "# Llama; `rank-bm25` for sparse retrieval.\n"
    "_pip_packages = [\n"
    "    \"transformers>=4.44\",\n"
    "    \"bitsandbytes>=0.43\",\n"
    "    \"accelerate>=0.33\",\n"
    "    \"sentence-transformers>=3.0\",\n"
    "    \"chromadb>=0.5\",\n"
    "    \"rank-bm25>=0.2.2\",\n"
    "    \"huggingface_hub>=0.24\",\n"
    "    \"datasets>=2.20\",\n"
    "    \"pyyaml>=6.0\",\n"
    "]\n"
    "proc = subprocess.run(\n"
    "    [sys.executable, \"-m\", \"pip\", \"install\", *_pip_packages],\n"
    "    capture_output=True, text=True,\n"
    ")\n"
    "if proc.returncode != 0:\n"
    "    print(\"PIP STDOUT (tail):\\n\" + proc.stdout[-3000:])\n"
    "    print(\"PIP STDERR (tail):\\n\" + proc.stderr[-3000:])\n"
    "    raise SystemExit(\"pip install failed — see tails above.\")\n"
    "print(\"setup ok\")\n"
)

# --------------------------------------------------------------------- #
# Cell 3 — HF Hub auth
# --------------------------------------------------------------------- #
code(
    "# Cell 3 — HF Hub login (private chunks dataset + gated Llama weights)\n"
    "from huggingface_hub import login, HfApi\n"
    "login()  # Colab inline widget — paste your Write token\n"
    "\n"
    "_whoami = HfApi().whoami()\n"
    "assert _whoami[\"name\"] == \"ankit-iiitdmj\", (\n"
    "    f\"HF Hub token belongs to {_whoami['name']!r}, expected 'ankit-iiitdmj'.\"\n"
    ")\n"
    "print(f\"HF Hub ok: user={_whoami['name']}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 4 — GPU check
# --------------------------------------------------------------------- #
code(
    "# Cell 4 — GPU check (4-bit 8B Llama needs ~5–6 GB; T4 is fine)\n"
    "import subprocess, torch, sys\n"
    "sys.path.insert(0, REPO_PATH)\n"
    "\n"
    "subprocess.run([\"nvidia-smi\"], check=False)\n"
    "print()\n"
    "assert torch.cuda.is_available(), \"No GPU detected — switch runtime to GPU before running Cell 9.\"\n"
    "dev = torch.cuda.get_device_properties(0)\n"
    "vram_gib = dev.total_memory / 1024**3\n"
    "print(f\"GPU: {dev.name}, VRAM: {vram_gib:.1f} GiB\")\n"
    "if vram_gib < 14:\n"
    "    print(\"WARN: <14 GiB VRAM — Llama-3.1-8B 4-bit may OOM on contexts; consider \"\n"
    "          \"`model_name=meta-llama/Llama-3.2-3B-Instruct` when constructing the generator.\")\n"
)

# --------------------------------------------------------------------- #
# Cell 5 — Corpus rebuild header
# --------------------------------------------------------------------- #
md(
    "## Corpus rebuild\n"
    "\n"
    "Pull the private 285-row chunks dataset from HF Hub and re-embed it\n"
    "into a fresh ChromaDB collection at `corpus/vector_db/`. Re-running\n"
    "the cell upserts by deterministic sha1 `chunk_id` — no duplicates.\n"
)

# --------------------------------------------------------------------- #
# Cell 6 — Load chunks + build Chroma
# --------------------------------------------------------------------- #
code(
    "# Cell 6 — Load chunks from HF and rebuild ChromaDB locally\n"
    "from src.rag.corpus_loader import load_chunks_from_hf, build_chroma\n"
    "\n"
    "chunks = load_chunks_from_hf()  # ankit-iiitdmj/iks-corpus-chunks\n"
    "print(f\"Loaded {len(chunks)} chunks; first chunk:\")\n"
    "print(\" \", {k: chunks[0][k] for k in (\"book_id\", \"chapter\", \"verse_or_section\", \"chunk_id\")})\n"
    "\n"
    "collection = build_chroma(chunks, persist_dir=\"corpus/vector_db\")\n"
    "print(f\"\\nCollection populated: count={collection.count()}\")\n"
    "assert collection.count() == len(chunks), \"Chroma count mismatch — re-run Cell 6.\"\n"
)

# --------------------------------------------------------------------- #
# Cell 7 — Build HybridRetriever
# --------------------------------------------------------------------- #
code(
    "# Cell 7 — Build HybridRetriever (dense + sparse + reranker, all on)\n"
    "from src.rag.retriever import HybridRetriever\n"
    "\n"
    "retriever = HybridRetriever(collection, use_dense=True, use_sparse=True, use_reranker=True)\n"
    "print(\n"
    "    f\"HybridRetriever ready: dense={retriever.use_dense} \"\n"
    "    f\"sparse={retriever.use_sparse} reranker={retriever.use_reranker}\"\n"
    ")\n"
)

# --------------------------------------------------------------------- #
# Cell 8 — Retriever smoke (BEFORE LLM load)
# --------------------------------------------------------------------- #
code(
    "# Cell 8 — Retriever smoke. Runs BEFORE Cell 9 so a retrieval failure\n"
    "# fails fast (no need to pay the ~5 GB Llama download cost).\n"
    "SMOKE_QUERIES = [\n"
    "    \"how to treat a diseased tree\",                    # Vrikshayurveda / Brihat ch.55\n"
    "    \"signs that predict rainfall\",                     # Brihat ch.21-28\n"
    "    \"how to find underground water\",                   # Brihat ch.54\n"
    "    \"yellow leaf disease and the correct soil for it\", # joint disease + soil (Phase 8 preview)\n"
    "    \"organic protocol for sandy loam crops\",          # cross-source retrieval\n"
    "]\n"
    "\n"
    "for q in SMOKE_QUERIES:\n"
    "    hits = retriever.retrieve(q, k=5)\n"
    "    print(\"=\" * 78)\n"
    "    print(f\"QUERY: {q!r}\")\n"
    "    for i, h in enumerate(hits, 1):\n"
    "        meta = h.metadata or {}\n"
    "        src = f\"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} v.{meta.get('verse_or_section','?')}\"\n"
    "        snip = (h.text or '').replace('\\n', ' ')[:140]\n"
    "        print(f\"  [{i}] score={h.score:.4f} stage={h.retriever}  {src}\")\n"
    "        print(f\"        {snip}\")\n"
    "    print()\n"
)

# --------------------------------------------------------------------- #
# Cell 9 — Load Llama-3.1-8B 4-bit
# --------------------------------------------------------------------- #
code(
    "# Cell 9 — Load Llama-3.1-8B 4-bit (gated — Cell 3's token must have license access)\n"
    "import torch\n"
    "from src.rag.generator import GroundedGenerator\n"
    "\n"
    "generator = GroundedGenerator(\n"
    "    model_name=\"meta-llama/Llama-3.1-8B-Instruct\",\n"
    "    load_in_4bit=True,\n"
    "    temperature=0.2,\n"
    "    max_new_tokens=512,\n"
    "    seed=42,\n"
    ")\n"
    "generator._ensure_loaded()  # noqa: SLF001 — warm up here so VRAM is visible BEFORE Cell 10\n"
    "torch.cuda.empty_cache()\n"
    "mem = torch.cuda.memory_allocated() / 1024**3\n"
    "print(f\"Llama-3.1-8B 4-bit loaded. CUDA memory in use: {mem:.2f} GiB\")\n"
)

# --------------------------------------------------------------------- #
# Cell 10 — End-to-end RAG on 5 demo queries
# --------------------------------------------------------------------- #
code(
    "# Cell 10 — End-to-end RAG: retriever → grounded generator, 5 demo queries\n"
    "from src.rag.pipeline import RAGPipeline\n"
    "\n"
    "pipeline = RAGPipeline(retriever=retriever, generator=generator, default_k=5)\n"
    "\n"
    "DEMO_QUERIES = [\n"
    "    # In-corpus expected to work well:\n"
    "    \"How should a diseased tree with falling branches be treated?\",\n"
    "    \"What signs in the sky predict imminent rainfall?\",\n"
    "    \"How does the classical text guide finding underground water?\",\n"
    "    # Joint disease + soil — foreshadows Phase 8 (multimodal context):\n"
    "    \"What organic protocol should be used for a tree showing yellow leaves growing in mixed sandy-loam soil?\",\n"
    "    # OUT-OF-CORPUS faithfulness check: the model MUST refuse rather than hallucinate.\n"
    "    \"What is the recommended drone-spraying schedule for monoculture rice fields?\",\n"
    "]\n"
    "\n"
    "for q in DEMO_QUERIES:\n"
    "    print(\"=\" * 78)\n"
    "    print(f\"QUERY: {q}\")\n"
    "    result = pipeline.answer(q, k=5)\n"
    "    print(\"--- ANSWER ---\")\n"
    "    print(result.answer)\n"
    "    print(\"--- CITATIONS ---\")\n"
    "    for c in result.citations:\n"
    "        print(\"  -\", c)\n"
    "    print(\"--- CHUNKS USED ---\")\n"
    "    for cid in result.used_chunk_ids:\n"
    "        # show the metadata of every chunk the model actually cited\n"
    "        match = next((r for r in result.retrieved if r.chunk_id == cid), None)\n"
    "        if match is not None:\n"
    "            meta = match.metadata\n"
    "            print(f\"  {cid[:10]}  {meta.get('source_text')} ch.{meta.get('chapter')} v.{meta.get('verse_or_section')}\")\n"
    "    print(\"--- TOP RETRIEVED (for inspection) ---\")\n"
    "    for i, h in enumerate(result.retrieved[:3], 1):\n"
    "        meta = h.metadata\n"
    "        src = f\"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} v.{meta.get('verse_or_section','?')}\"\n"
    "        snip = (h.text or '').replace('\\n',' ')[:120]\n"
    "        print(f\"  [{i}] score={h.score:.4f} {src} :: {snip}\")\n"
    "    print()\n"
)

# --------------------------------------------------------------------- #
# Cell 11 — How to read the outputs
# --------------------------------------------------------------------- #
md(
    "## How to read these results\n"
    "\n"
    "- **In-corpus queries 1–3** should each produce a numbered organic\n"
    "  protocol with one or more `[Source Text, ch.X, v.Y]` citations\n"
    "  matching the top retrieved chunks. The `CHUNKS USED` row tells you\n"
    "  which retrieved chunks the answer actually cited.\n"
    "- **Joint query 4** is a preview of Phase 8 (multimodal context):\n"
    "  the question deliberately mentions both a disease symptom and a\n"
    "  soil type, so retrieval should pull chunks from both Vrikshayurveda\n"
    "  (disease) and Brihat Samhita (soil/exploration). The Phase 8\n"
    "  notebook will inject vision-module predictions into the query at\n"
    "  exactly this seam.\n"
    "- **Out-of-corpus query 5** is the §17 faithfulness sanity check.\n"
    "  The model MUST emit the locked refusal sentence — *\"The retrieved\n"
    "  classical-text passages do not contain enough information to\n"
    "  answer this question. Please consult a qualified agricultural\n"
    "  expert.\"* — rather than hallucinating a drone-spraying schedule.\n"
    "  If it hallucinates a treatment instead, log it as a Phase 11 RAGAS\n"
    "  faithfulness issue and do NOT hide it.\n"
)

# --------------------------------------------------------------------- #
# Cell 12 — Phase 7 complete + what's next
# --------------------------------------------------------------------- #
md(
    "## Phase 7 complete\n"
    "\n"
    "Pipeline lives at:\n"
    "\n"
    "- `src/rag/corpus_loader.py` (`load_chunks_from_hf`, `build_chroma`)\n"
    "- `src/rag/retriever.py` (`HybridRetriever`, `RetrievedChunk`)\n"
    "- `src/rag/generator.py` (`GroundedGenerator`, §17 prompt)\n"
    "- `src/rag/pipeline.py` (`RAGPipeline`)\n"
    "\n"
    "### Next: Phase 8 (multimodal integration)\n"
    "\n"
    "The query-construction step is the seam — Phase 8 will compose the\n"
    "user question with the Phase 5 disease classifier's prediction and\n"
    "the Phase 6 soil classifier's prediction (e.g. *\"yellow leaves on\n"
    "Bottle Gourd in sandy-loam soil\"*) and route the enriched query\n"
    "through this same `RAGPipeline`.\n"
    "\n"
    "### Swapping LLMs\n"
    "\n"
    "If VRAM is tight (T4 free tier with long contexts), pass\n"
    "`model_name=\"meta-llama/Llama-3.2-3B-Instruct\"` to either\n"
    "`GroundedGenerator(...)` or `RAGPipeline(model_name=...)`. The pipeline\n"
    "is generator-agnostic, so the rest of the code is unchanged.\n"
    "\n"
    "### Adding more books\n"
    "\n"
    "Once Krishi Parashara, Upavanavinoda, Kashyapiyakrishisukti, or the\n"
    "TBD-sixth text are processed by Phase 3, re-run\n"
    "`python scripts/push_corpus_chunks.py` on the laptop to push the\n"
    "expanded corpus to `ankit-iiitdmj/iks-corpus-chunks`. The next Colab\n"
    "Cell 6 run picks them up automatically — no code change anywhere in\n"
    "the RAG pipeline.\n"
)


# Finalise notebook
NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(NB, fh)
print(f"wrote {NOTEBOOK_PATH} ({len(cells)} cells)")
