# Claude Code Prompt — Phase 8: Multimodal Integration (vision → RAG query, 3 strategies)

> Paste below the horizontal rule into Claude Code in a fresh session on the laptop. It builds `src/integration/` and a Colab notebook that wires the Phase 5 disease model + Phase 6 soil model into the Phase 7 RAG pipeline via three query-construction strategies. Agent time ~40 min. Colab run ~30–45 min (model downloads dominate).

---

## CONTEXT

This is the core-novelty phase (master plan §16, contribution C2): the joint disease–soil–crop context construction that turns vision outputs into a retrieval query. It also wires in contribution C5 (cause-conditional retrieval) as an optional query input.

**What already exists (reuse, do NOT rebuild):**
- Phase 5 disease model: EfficientNet-B4 @ 380×380, on HF `ankit-iiitdmj/iks-disease-plantdoc` (final stage). Inference code in `src/disease/`.
- Phase 6 soil model: EfficientNet-B0 multi-task (soil_type, moisture, texture), on HF `ankit-iiitdmj/iks-soil-multitask-v2`. Inference code in `src/soil/`.
- Phase 7 RAG pipeline: `src/rag/pipeline.py` `RAGPipeline` (HybridRetriever dense+BM25+rerank → Llama-3.1-8B-Instruct 4-bit grounded generator). Corpus = 4 books in private HF dataset `ankit-iiitdmj/iks-corpus-chunks`, rebuilt into ChromaDB in-session.

**Platform: Colab/Linux, single-process** (same reason as Phase 7: chromadb+torch segfault on Windows; LLM needs T4). Laptop is used only to author code + push the commit.

**Research-backed scope decision (locked):** the corpus is TEXT-ONLY, so this is *multimodal input → text retrieval*, not image retrieval. Current literature shows vision→text query construction reliably beats multimodal-embedding retrieval for a text corpus, and embedding projection suffers a modality gap. Therefore:
- **Strategy A (Template)** and **Strategy B (LLM-mediated query expansion)** are the MAIN contribution.
- **Strategy C (multimodal embedding projection)** is implemented as the ABLATION COMPARISON that demonstrates *why* text-query wins — not as the primary path.

**The real problem these strategies solve:** modern vision labels (e.g. "rice blast") do NOT appear in the classical texts, which use symptom descriptions and terms like *kunapajala*, *panchagavya*. Query construction must BRIDGE modern labels → classical vocabulary. Strategy B exists precisely for this bridge.

**Hard rules:**
- All paths via `src.utils.paths`. Logging via `src.utils.logging_setup.get_logger`.
- **Local commits only — never `git push`.**
- Reuse Phase 5/6/7 code by import; do NOT reimplement vision inference or the RAG pipeline.
- Corpus + models are READ-ONLY here.
- Open weights only.

---

# Mission

Build `src/integration/` that converts (leaf image, soil image, crop type, optional causal context) into a retrieval query via three strategies, feeds each into the Phase 7 RAG pipeline, and produces a side-by-side qualitative comparison. Rigorous RAGAS scoring is deferred to Phase 11 — Phase 8 delivers the mechanism + a qualitative read.

## Locked Decisions

| # | Decision |
|---|---|
| 1 | Three strategies: A (template), B (LLM-mediated expansion) = main; C (multimodal embedding projection) = ablation. |
| 2 | C5 causal-context is an OPTIONAL input to query construction, default `"unspecified"`; values: soil_deficiency / pest_attack / spread_from_neighbours / unspecified. When set, it conditions the query (cause-conditional retrieval). |
| 3 | Strategy B reuses the already-loaded Llama-3.1-8B (no second model) with a query-rewriting prompt focused on the modern→classical vocabulary bridge. |
| 4 | Strategy C: concatenate disease-CNN + soil-CNN penultimate embeddings + a crop embedding → linear projection to the corpus embedding dim (1024, bge-large). Train the projection on AUTO-GENERATED weak pairs (each sample's Strategy-A query embedding ↔ its top retrieved chunk embedding) — NO manual annotation. Explicitly documented as a weak-signal ablation; a fully-trained version is future work. |
| 5 | Vision inference reused from `src/disease` and `src/soil` (import, don't rewrite). |
| 6 | Output of each strategy flows through the existing `RAGPipeline` unchanged. |
| 7 | Colab notebook + `src/integration/` code. Single local commit. No push. |

---

## Deliverables

### `src/integration/context.py`
- `@dataclass VisualContext`: disease_label, disease_conf, soil_type, moisture, texture, crop_type, causal_context="unspecified", plus optional raw embedding tensors (disease_emb, soil_emb).
- `build_visual_context(leaf_img, soil_img, crop_type, causal_context="unspecified") -> VisualContext`:
  - Run disease model (import from src.disease inference) → label + confidence + penultimate embedding.
  - Run soil model (import from src.soil inference) → soil_type, moisture, texture + penultimate embedding.
  - Return populated VisualContext.

### `src/integration/strategy_template.py` (Strategy A)
- `build_query_template(ctx: VisualContext) -> str`:
  - Deterministic fill, e.g.: `"Organic treatment for {disease_label} affecting {crop_type} grown in {soil_type} soil that appears {moisture} with {texture} texture."` + if causal_context != unspecified, append a clause conditioning on the pathway (e.g. soil_deficiency → "...with emphasis on soil restoration and nourishment"; pest_attack → "...with emphasis on pest control"; spread_from_neighbours → "...with emphasis on preventing spread between plants").
- Transparent, no model call.

### `src/integration/strategy_llm.py` (Strategy B)
- `build_query_llm(ctx: VisualContext, llm) -> str`:
  - Prompt the (passed-in, already-loaded) Llama-3.1-8B to rewrite the structured context into ONE retrieval query that bridges modern terminology to classical Indian agricultural vocabulary (symptom descriptions, traditional treatment concepts). The prompt MUST instruct: do not invent a treatment, only reformulate the query; prefer descriptive/symptomatic phrasing likely to match classical texts; incorporate the causal pathway if provided.
  - Return the rewritten query string. Deterministic (temp 0.2, fixed seed).

### `src/integration/strategy_embed.py` (Strategy C — ablation)
- `class MultimodalProjector(nn.Module)`: linear (disease_dim + soil_dim + crop_emb_dim → 1024).
- `crop_embedding(crop_type) -> tensor`: bge-large embedding of the crop name (reuse the corpus embedder).
- `train_projection_weak(samples, retriever, embedder) -> MultimodalProjector`:
  - For a set of demo samples, compute Strategy-A query → its top-1 retrieved chunk embedding (the weak target). Fit the projector to map concatenated visual embeddings → that target embedding (a few epochs MSE). NO manual labels.
- `retrieve_via_embedding(ctx, projector, collection, k) -> list[Chunk]`: project visual embeddings → query vector → ChromaDB similarity. Returns chunks.
- Module docstring states clearly: weak-signal ablation demonstrating the modality gap; not the primary retrieval path.

### `src/integration/compare.py`
- `run_all_strategies(ctx, rag_pipeline, projector=None, k=5) -> dict`:
  - Strategy A: build query → `rag_pipeline.retrieve` → top-k chunks + final grounded answer.
  - Strategy B: same with LLM-built query.
  - Strategy C: `retrieve_via_embedding` → top-k chunks (+ optionally generate).
  - Return per-strategy {query, retrieved_chunk_ids, retrieved_sources, answer}.
- `qualitative_compare(results) -> table`: for each strategy, show the query, the source texts of the top-k chunks, and a simple on-topic overlap count (how many of the k chunks come from a plausibly-relevant book/chapter given the disease+soil). This is a QUALITATIVE read, not RAGAS.

### `notebooks/phase8_multimodal_integration.ipynb` (13 cells, nbformat)
1. **Markdown** — Phase 8 goal (C2 + C5), research-backed scope (A/B main, C ablation), the modern→classical bridge problem.
2. **Setup** — clone, pip install (reuse Phase 7 deps + nothing heavy new).
3. **HF auth** — token (private chunks dataset + gated Llama + model checkpoints).
4. **GPU check**.
5. **Load vision models** — disease B4 + soil B0 multi-task from HF; print they loaded.
6. **Load RAG pipeline** — Phase 7 `RAGPipeline` (rebuild ChromaDB from `iks-corpus-chunks`, now 4 books; hybrid retriever; Llama-3.1-8B). Print corpus chunk count (should be == 206 now that 2 more books are in).
7. **Demo inputs** — provide 3–4 (leaf image, soil image, crop) sample pairs from the dataset test sets, plus an optional upload cell. For each, set a causal_context to exercise C5 on at least one.
8. **build_visual_context** on the samples — print the structured outputs (disease, soil_type, moisture, texture, causal_context).
9. **Strategy A** — show the templated query + retrieved sources + grounded answer for each sample.
10. **Strategy B** — show the LLM-bridged query + retrieved sources + grounded answer; highlight how B's query differs from A's (the vocabulary bridge).
11. **Strategy C (ablation)** — train the weak projection, retrieve, show top-k sources; note qualitatively whether it under/over-performs A/B.
12. **Side-by-side comparison** — `qualitative_compare` table across A/B/C for all samples; short written read on which retrieves more on-topic chunks and why (expected: B ≥ A > C).
13. **Markdown** — findings; explicit note that rigorous RAGAS context_precision/recall scoring + the expert gold-query set are Phase 11; C5 causal conditioning is wired and demoed but full causal evaluation is later; next = Phase 9 (explainability) / Phase 10 (full system + Streamlit).

### Tests (`tests/integration/`)
- `test_template.py`: build_query_template fills all fields; causal_context clause appears only when set; unspecified yields no causal clause.
- `test_compare_smoke.py`: with a stub RAG pipeline + stub vision context (monkeypatched, no models), run_all_strategies returns the expected per-strategy dict shape and threads causal_context through. No model downloads, no GPU.
- Run `pytest tests/integration/ -q`.

### progress.md + commit
- Append Phase 8 entry: 3 strategies built (A/B main, C ablation); vision→RAG wired; C5 causal hook added (optional, default unspecified); qualitative comparison done; rigorous eval deferred to Phase 11.
- Stage: `src/integration/*`, `notebooks/phase8_multimodal_integration.ipynb`, `tests/integration/*`, `progress.md`.
- **Do NOT stage** chunk text / vector_db / PDFs / model weights.
- Commit message: `"Phase 8: multimodal integration (template + LLM-mediated query, embedding ablation) + C5 causal hook"`
- **No git push.**

---

## End Checks (must all pass)
- [ ] `python -c "from src.integration.compare import run_all_strategies; from src.integration.context import build_visual_context; print('ok')"` imports (no model load).
- [ ] `python -c "import nbformat; print(len(nbformat.read('notebooks/phase8_multimodal_integration.ipynb',as_version=4).cells))"` == 13.
- [ ] `pytest tests/integration/ -q` passes (no GPU, no network).
- [ ] Notebook Cell 6 prints a corpus chunk count == 206 (confirms the 2 newly-OCR'd books are in the dataset).
- [ ] `git status`: no chunk text / vector_db / PDF / weights staged.
- [ ] `git log --oneline -1` starts with "Phase 8:".
- [ ] No `git push`.

## Working Style
- Plan first (numbered). No push.
- Reuse Phase 5/6/7 code by import — do not duplicate vision inference or the RAG pipeline.
- Strategy B's value is the modern→classical vocabulary bridge — make that explicit in its prompt and in Cell 10's commentary.
- Strategy C is honestly framed as a weak-signal ablation (modality gap), not a tuned competitor.
- C5 causal-context threads through all three strategies as an optional, defaulted input.
- Stop and ask if: the `iks-corpus-chunks` dataset still shows only 206 chunks (means the 2 new books weren't pushed — the Phase 3b external-OCR text may not have been re-ingested + re-pushed); the disease/soil checkpoints are unreachable; T4 VRAM overflows (fall back to Llama-3.2-3B via the model-agnostic hook).

## What success looks like
1. A (leaf, soil, crop, optional cause) → three queries → three retrievals → grounded answers, side by side.
2. Strategy B visibly bridges modern labels to classical vocabulary; B ≥ A > C qualitatively.
3. C5 causal context demonstrably changes the query/retrieval on at least one sample.
4. Tests pass without GPU/network; one unpushed commit.
5. Ready for Phase 9 (explainability) and Phase 11 (rigorous RAGAS ablation on a gold-query set — the formal version of C2's contribution).

Begin.
