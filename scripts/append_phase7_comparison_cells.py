"""Append a self-contained Phase-3b.2 comparison section to the
phase 7 notebook so the supervisor demo shows
Tesseract-era outputs (already frozen in the .ipynb) directly above
Gemini-era outputs.

Idempotent: if cells with the marker tag already exist, they are
replaced rather than duplicated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NB_PATH = Path("phase7_rag_pipeline.ipynb")  # root copy with frozen outputs
MARKER_TAG = "phase3b2_comparison"


def _md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [MARKER_TAG]},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"tags": [MARKER_TAG]},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELL_HEADER_MD = """\
---

# Phase 3b.2 re-run -- Gemini-OCR'd corpus (comparison vs above)

> All cells above this point were produced from the **Tesseract**-OCR'd
> 285-chunk corpus (Phase 3, May 2026). The outputs you see frozen above
> in cells 6-10 are from that run. They are kept here intentionally so
> the comparison is visible in one document.

**What changed in Phase 3b.2 (June 2026):**

1. **New OCR engine** -- the four book PDFs were re-OCR'd with
   **Gemini 3.5 Flash** (paid tier, ~Rs 41 total spend) instead of
   local Tesseract. Gemini was prompted to skip Devanagari script,
   preserve verse numbers exactly, and use context to spell Sanskrit
   plant names consistently. The Tesseract output had repeating
   Devanagari ink-bleed noise inside English sentences which was
   degrading retrieval and refusal behaviour.
2. **Two new books added.** The corpus now spans **four** classical
   treatises, not two:
   - Vrikshayurveda (Surapala, tr. Sadhale 1996)
   - Brihat Samhita Part 1 (Varahamihira, tr. Bhat, 12 chapters)
   - **NEW**: Krishi Parashara (Majumdar & Banerji 1960, PDF pp.94-119)
   - **NEW**: Upavanavinoda (Sarngadhara, tr. Majumdar, PDF pp.77-96)
3. **Fewer but cleaner chunks**: 285 -> 206. Tesseract's mis-OCR'd
   numbers were producing spurious "new verse" boundaries; Gemini's
   clean output lets the chunker pack proper semantic blocks.

**How to read the cells below:**

The next 4 code cells rebuild the pipeline with the new corpus and
re-run **the exact same 5 demo queries** that appear in cell 10
above. Compare answers, citations, and retrieved-chunk snippets
side-by-side with the Tesseract-era outputs above.

> **Note on running this section:** these cells are self-contained --
> they re-import everything and rebuild a fresh retriever
> (`retriever_gemini`) and pipeline (`pipeline_gemini`) at a separate
> ChromaDB path (`corpus/vector_db_gemini/`). If you want to preserve
> the frozen outputs in cells 6-10, do **not** re-run those cells in
> Colab -- start execution from this section by clicking on the next
> cell and using `Runtime -> Run from selected cell`. You will need
> to re-run **only** cell 3 (HF login) once before this section so
> the auth token is in memory.
"""

CELL_LOAD_CHUNKS = """\
# Phase 3b.2 -- Cell A: force-reload chunks from HF (the dataset was
# replaced after the Gemini re-OCR, so the on-disk HF cache must be
# bypassed). Expect 206 chunks across 4 books.
from datasets import load_dataset
from src.rag.corpus_loader import DEFAULT_CHUNKS_REPO, REQUIRED_FIELDS

ds_gemini = load_dataset(
    DEFAULT_CHUNKS_REPO,
    split="train",
    download_mode="force_redownload",   # bypass any old-snapshot cache
)
chunks_gemini = []
for row in ds_gemini:
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    assert not missing, f"missing fields: {missing}"
    chunks_gemini.append({k: row[k] for k in REQUIRED_FIELDS})

print(f"Loaded {len(chunks_gemini)} chunks from {DEFAULT_CHUNKS_REPO}")
from collections import Counter
per_book = Counter(c["book_id"] for c in chunks_gemini)
print()
print("Per-book breakdown (Gemini OCR):")
for bid, n in sorted(per_book.items(), key=lambda kv: -kv[1]):
    print(f"  {bid:<22} {n:>4} chunks")
print()
print("First chunk preview:")
first = chunks_gemini[0]
print(f"  source_text     : {first['source_text']}")
print(f"  chapter / verse : {first['chapter']} / {first['verse_or_section']}")
print(f"  text (first 200): {first['text'][:200]!r}")
"""

CELL_BUILD_CHROMA = """\
# Phase 3b.2 -- Cell B: re-embed and upsert into a SEPARATE ChromaDB
# collection so the comparison run does NOT clobber any in-process
# `collection` variable created by Cell 6 above. Expect 206 vectors.
from src.rag.corpus_loader import build_chroma

collection_gemini = build_chroma(
    chunks_gemini,
    persist_dir="corpus/vector_db_gemini",
    collection_name="iks_corpus_gemini",
)
print()
print(f"ChromaDB ready: count={collection_gemini.count()} (expected 206)")
assert collection_gemini.count() == len(chunks_gemini), \\
    f"Chroma count mismatch: {collection_gemini.count()} vs {len(chunks_gemini)}"
"""

CELL_BUILD_RETRIEVER = """\
# Phase 3b.2 -- Cell C: hybrid retriever (dense + sparse + reranker)
# over the new collection. Same configuration as cell 7 above so the
# comparison is apples-to-apples.
from src.rag.retriever import HybridRetriever

retriever_gemini = HybridRetriever(
    collection_gemini,
    use_dense=True, use_sparse=True, use_reranker=True,
)
print(
    f"retriever_gemini ready: dense={retriever_gemini.use_dense} "
    f"sparse={retriever_gemini.use_sparse} reranker={retriever_gemini.use_reranker}"
)
"""

CELL_RUN_QUERIES = """\
# Phase 3b.2 -- Cell D: re-run the SAME 5 demo queries from cell 10
# above with the Gemini-OCR pipeline. Compare the answers, citations,
# and retrieved chunks against the Tesseract-era outputs in cell 10.
#
# Reuses the existing `generator` (Llama-3.1-8B-Instruct 4-bit) if it
# was already loaded earlier in this Colab session. If not, loads it
# fresh -- one-time ~5 min cost.
import torch

try:
    generator  # noqa: F821 -- re-use if cell 9 already ran
    print("Reusing already-loaded `generator` from earlier in the session.")
except NameError:
    print("`generator` not in memory -- loading Llama-3.1-8B 4-bit fresh ...")
    from src.rag.generator import GroundedGenerator
    generator = GroundedGenerator(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        temperature=0.2,
        max_new_tokens=512,
        seed=42,
    )
    generator._ensure_loaded()  # noqa: SLF001
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Llama loaded. CUDA memory in use: {mem:.2f} GiB")

from src.rag.pipeline import RAGPipeline
pipeline_gemini = RAGPipeline(
    retriever=retriever_gemini, generator=generator, default_k=5,
)

DEMO_QUERIES = [
    # Same five queries as cell 10:
    "How should a diseased tree with falling branches be treated?",
    "What signs in the sky predict imminent rainfall?",
    "How does the classical text guide finding underground water?",
    "What organic protocol should be used for a tree showing yellow leaves growing in mixed sandy-loam soil?",
    "What is the recommended drone-spraying schedule for monoculture rice fields?",
]

for q in DEMO_QUERIES:
    print("=" * 78)
    print(f"QUERY [GEMINI-OCR corpus]: {q}")
    result = pipeline_gemini.answer(q, k=5)
    print("--- ANSWER ---")
    print(result.answer)
    print("--- CITATIONS ---")
    for c in result.citations:
        print("  -", c)
    print("--- CHUNKS USED ---")
    for cid in result.used_chunk_ids:
        match = next((r for r in result.retrieved if r.chunk_id == cid), None)
        if match is not None:
            meta = match.metadata
            print(f"  {cid[:10]}  {meta.get('source_text')} ch.{meta.get('chapter')} v.{meta.get('verse_or_section')}")
    print("--- TOP RETRIEVED (for inspection) ---")
    for i, h in enumerate(result.retrieved[:3], 1):
        meta = h.metadata
        src = f"{meta.get('source_text','?')} ch.{meta.get('chapter','?')} v.{meta.get('verse_or_section','?')}"
        snip = (h.text or '').replace('\\n',' ')[:120]
        print(f"  [{i}] score={h.score:.4f} {src} :: {snip}")
    print()
"""

CELL_HOW_TO_READ_MD = """\
## How to compare the two runs

For each of the five queries in cell D above, look at the **same**
query in cell 10 (Tesseract-era) and contrast:

| What to compare | Tesseract era (cell 10) | Gemini era (cell D) |
|---|---|---|
| Did the model answer or refuse? | Q2 (rainfall) and Q3 (water) often refused or gave thin answers because chunks were Devanagari-noisy | Should now produce a real grounded answer for in-corpus questions |
| Citations | Heavy on Vrik + Brihat only (the only two books) | Can now also cite **Krishi Parashara** and **Upavanavinoda** |
| Retrieved chunk text | Look for stray Devanagari characters or garbled diacritics | Clean English, with proper diacritics (Asvattha, Prajapati, etc.) |
| Out-of-corpus Q5 (drones) | MUST emit the locked refusal sentence | Should still emit the same refusal -- this is the faithfulness guard |

If any **in-corpus** query that worked above now refuses, that is a
regression to flag. If any **out-of-corpus** query stops refusing and
starts hallucinating, that is a far more serious regression.

### Cost + provenance receipt for the thesis

- OCR engine    : Gemini 3.5 Flash (paid, gemini-3.5-flash)
- Pages OCR'd   : 466 across 4 books (Vrik 101 + Brihat 319 selected + KP 26 + UV 20)
- Spend         : ~Rs 41 INR (~$0.49 USD)
- Re-run cost   : free -- chunks pulled from the same private HF dataset
- Resume safety : per-page on-disk cache made the run idempotent across
                  3 sessions (one was killed by an IDE close, one by a
                  free-tier 429 before billing propagated)
- Pipeline      : unchanged from cell 5-9 above (hybrid retrieval + Llama-3.1-8B 4-bit grounded generator with the locked SYSTEM_PROMPT_V17)
"""


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Drop any previously-appended comparison cells so this script is
    # idempotent.
    nb["cells"] = [
        c for c in nb["cells"]
        if MARKER_TAG not in (c.get("metadata", {}).get("tags") or [])
    ]

    new_cells = [
        _md(CELL_HEADER_MD),
        _code(CELL_LOAD_CHUNKS),
        _code(CELL_BUILD_CHROMA),
        _code(CELL_BUILD_RETRIEVER),
        _code(CELL_RUN_QUERIES),
        _md(CELL_HOW_TO_READ_MD),
    ]
    nb["cells"].extend(new_cells)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Appended {len(new_cells)} cells (tag={MARKER_TAG!r}) to {NB_PATH}")
    print(f"Total cells now: {len(nb['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
