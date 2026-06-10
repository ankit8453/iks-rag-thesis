# Weekly Progress Log

## Overview
This document tracks weekly progress on the IKS Agricultural Advisory System thesis project. Each week includes completed tasks, blockers, and goals for the next week.

---

## Phase 5-R Part 2: background-randomization retrain + no-leaf reject + Grad-CAM audit

Phase 5-R Part 1 (commit b8269c1) shipped the segmentation + composite + QC pipeline; per-dataset verdicts: PlantVillage classical ✓, PlantDoc rembg ✓, Paddy Doctor train as-is (full-canopy field photos, no fg/bg split worth randomising). Phase 9 Grad-CAM identified the real problem the retrain has to fix: only 3 of 256 PlantDoc test images had the disease-classifier's CAM peak inside the central 60% box — the model attends to corners / backgrounds / watermarks instead of the leaf. Part 2 builds the cascade trainer that decorrelates background from class by compositing leaves onto a random soil / urban background each epoch, plus an audit that says **keep or revert** based on whether the central-attention rate actually improves.

### What the code does (read top-to-bottom)

- `src/disease/segment_cache.py` — idempotent batch-segmenter + mask cache. Writes `data/plant_disease/_masks/<dataset>/<relpath>.png` once via the Part 1 `segment()` router (PlantVillage classical, PlantDoc rembg). Persists a per-dataset `_segmentation_log.json` recording the `flagged_rel_paths` (foreground < 5% or > 95%); the trainer reads that list at startup via `load_flagged_set()` so flagged rows fall back to raw at train time and a single mis-segmentation never poisons a batch. Paddy Doctor is deliberately NOT cached (no randomisation planned).
- `src/disease/randomized_dataset.py` — `RandomizedDiseaseDataset` is a `torch.utils.data.Dataset`-compatible wrapper with three per-sample modes: `"randomize"` (load image + cached mask + random bg from `build_background_pool()`, composite via Part 1's `composite_leaf_on_bg`), `"raw"` (Paddy path, pass-through), and `"no_leaf"` (Pandey `Background_without_leaves` rows + bare-soil hold-out, raw, label = 27). Per-epoch seed via `set_epoch(epoch)` so the same `(epoch, idx)` reproduces the same composite — necessary for the OLD-vs-NEW Grad-CAM comparison to be apples-to-apples. `build_samples_from_split`, `build_no_leaf_samples`, `load_class_map_with_no_leaf` (extends the 27-class PlantDoc map by appending `no_leaf` at index 27, existing ids do NOT shift) are the helpers the trainer calls.
- `src/disease/train_cascade_r.py` — `STAGE_INFO_R` table mirrors the original `STAGE_INFO` but adds a `mode` field per stage and points all checkpoints at a NEW namespace `models/disease_r/iks-disease-r-{plantvillage,paddy-doctor,plantdoc}/` so the old `iks-disease-*` models are untouched and revert is trivial. `build_loaders_for_stage` constructs train/val with the requested mode (val carries no-leaf rows at the PlantDoc stage; test is ALWAYS raw so the top-1 number is directly comparable to the original Phase 5's 71%). `train_one_stage_r` reuses the original `train_one_stage` core (same hyperparams, same seed, same CheckpointManager), warm-starts from the prior stage via `strict=False` (so the 38→10→28 head change doesn't break the load). `run_cascade` runs all three stages sequentially.
- `src/disease/gradcam_audit.py` — re-runs the Phase 9 central-60% test on the full PlantDoc test split for both engines. `audit_engine(name, engine)` walks every test row, runs `disease_gradcam`, takes the CAM peak position, tallies the central-attention rate + top-1 accuracy. `run_old_vs_new()` does both (falls back gracefully if the new checkpoint is missing). `keep_or_revert(old, new)` applies the locked rule: **central-attention gain ≥ +5 pp AND top-1 drop ≤ 3 pp ⇒ keep; otherwise revert**.

### Decisions locked from Part 1 (no re-litigation in Part 2)

- Paddy Doctor: **NO randomisation, train as-is.** Part 1's `paddy_qc.png` showed the classical pipeline can't split foreground from a full-canopy paddy stand, and rembg fragmented the rice plant in side experiments. The cascade trainer routes paddy through `mode="raw"`.
- Dr. Pandey's dataset: **only the `Background_without_leaves` folder is used.** The other 35 leaf classes are a confirmed PlantVillage re-pack (`docs/pandey_dataset_inspection.md`); merging them would re-amplify the very shortcut bias we're trying to break.
- Same architecture / hyperparameters / seed / test splits as the original Phase 5. Every difference between OLD and NEW must be attributable to the randomisation, not a config drift.

### Tests — 11 new + 31 prior = 42 passing in 62 s

- `tests/disease/test_segment_cache.py` — `mask_path_for` is stable + OS-independent; a flagged segmentation result still writes its mask to disk AND lands in the log's `flagged_rel_paths` (so `load_flagged_set()` returns it); the cache is idempotent (a re-run hits zero new segmentations); `load_flagged_set()` on a never-cached dataset returns the empty set. No GPU, no network — `segment()` is monkeypatched.
- `tests/disease/test_randomized_dataset.py` — `mode="raw"` returns the image unchanged (Paddy path), `mode="no_leaf"` carries the reject label (27), `mode="randomize"` actually composites (bg colour is visible in the output AND leaf colour is visible — neither solid), a row in `flagged_rel_paths` falls back to raw even when a mask file exists (no_leaf-style raw image even though the bg pool was bright white), per-epoch seeding is reproducible (same `(epoch, idx)` → identical pixel arrays; different epoch → different array), `build_no_leaf_samples` produces reject-labelled rows, `load_class_map_with_no_leaf` appends idx 27 without shifting existing class ids.
- The previous 31 disease tests (model + train + transforms + smoke + ...) all still pass — Phase 5-R is fully additive.

### Notebook — `notebooks/phase5r_retrain.ipynb` (12 cells per the prompt)

1. Markdown — experiment framing + locked keep-or-revert rule.
2. Clone + pip install (Phase 5/7 deps + rembg + onnxruntime + matplotlib) + HF login + GPU check.
3. Build background pool; show 8-image preview across phantomfs / sirajganj / pandey.
4. Batch-segment PlantVillage (classical) + PlantDoc (rembg); print % flagged per dataset; refuse to continue if PV > 10% flagged or PD > 20% (segmentation drift halts training).
5. Sanity: render 6 on-the-fly composites (3 PV + 3 PD) to confirm training inputs look right.
6. Stage 1 — pretrain B4 on randomised PlantVillage (38 classes).
7. Stage 2 — finetune on Paddy Doctor as-is (10 classes), warm-started from Stage 1.
8. Stage 3 — finetune on randomised PlantDoc + no_leaf (28 classes), warm-started from Stage 2.
9. Evaluate on RAW test splits — top-1 + no_leaf precision/recall — directly comparable to the original Phase 5's 71%.
10. Grad-CAM audit — central-attention rate OLD vs NEW on the SAME PlantDoc test images; persist `docs/phase5r_audit.json`.
11. Markdown — verdict table (filled by Cell 10) + keep/revert.
12. Markdown — next steps for either decision (push checkpoints + re-run Phase 9 / wire `no_leaf` into Phase 10 UI guardrail, OR document the bias as a Phase-11 follow-up).

### Cascade resource shape

T4 / Colab. Stage 1 (~43 k PV images × 30 epochs of randomised compositing) is the bottleneck — expect ~5 hr; Stage 2 (8.3 k paddy as-is) ~1.5 hr; Stage 3 (2 k PlantDoc + ~1 k no_leaf) ~1 hr; Grad-CAM audit on 256 PlantDoc test images ~7 min. The prompt budgets 5–9 hr over 1–2 sessions; the cascade is split across `train_one_stage_r` calls so a Colab session-timeout can resume from the warm-started checkpoint with no logic change.

### Phase 5-R Part 2 end checks (all green)

- `python -c "from src.disease.segment_cache import build_mask_cache; from src.disease.randomized_dataset import RandomizedDiseaseDataset; from src.disease.train_cascade_r import run_cascade; from src.disease.gradcam_audit import run_old_vs_new; print('ok')"` → `ok` (no model load).
- `python -c "import nbformat; print(len(nbformat.read('notebooks/phase5r_retrain.ipynb',as_version=4).cells))"` → `12`.
- `pytest tests/disease/ -q` → `42 passed in 61s` (31 prior + 11 new).
- `git status` — `src/disease/{segment_cache.py, randomized_dataset.py, train_cascade_r.py, gradcam_audit.py}`, `tests/disease/{test_segment_cache.py, test_randomized_dataset.py}`, `scripts/build_phase5r_notebook.py`, `notebooks/phase5r_retrain.ipynb`, `progress.md` staged. NO masks, NO weights, NO image data, NO push.
- Single local commit titled `"Phase 5-R Part 2: background-randomization retrain + no-leaf reject + Grad-CAM audit"`. **No git push.**

---

## Phase 9: explainability layer (Grad-CAM disease+3 soil heads, retrieved-chunk highlighting)

Phase 9 (master plan §18) delivers the honest-interpretability surfaces the paper relies on in §35: *where* each vision model looked (Grad-CAM heatmaps) and *why* each chunk was retrieved (per-chunk matched-term overlay + similarity score panel). Built `src/explain/` end-to-end, 12-cell Colab notebook `notebooks/phase9_explainability.ipynb`, 17 unit tests. Reuses Phase 5 disease + Phase 6 soil + Phase 7 RAG + Phase 8 integration by import; models + corpus stay read-only.

### Vision Grad-CAM — disease + 3 soil heads

- `src/explain/gradcam.py:find_target_layer` locates the timm EfficientNet Grad-CAM target by attribute walk: prefers `backbone.conv_head` (final 1×1 expansion conv before GAP — highest-resolution class-discriminative heatmap), falls back to `backbone.blocks[-1]` with a logged WARNING so any drop in heatmap resolution is auditable. Raises `AttributeError` on a degenerate backbone (no `.conv_head`, empty `.blocks`).
- `disease_gradcam(image_path, engine) -> GradCAMResult` runs the full-precision B4 backbone with gradients enabled (NOT inside `torch.no_grad`), preprocesses at 380×380 with ImageNet stats (matching `DiseaseInferenceEngine`'s pipeline), uses the engine's argmax prediction as the Grad-CAM target, and returns the overlay (uint8 H×W×3), raw heatmap (float32 H×W in [0,1]), the human-readable label via `engine.class_names`, the confidence, AND the raw argmax index — so the label-mapping is per-call auditable (defends against a regression of the Phase 8 placeholder-label bug).
- `SoilHeadWrapper(soil_model, head)` is the real novelty for the multi-task soil model. `SoilMultiTaskClassifier.forward` returns `dict[head_name, tensor]`; pytorch-grad-cam's `ClassifierOutputTarget` requires a single-tensor forward, so the wrapper builds an `nn.Module` that runs `model.backbone(x)` then ONLY the chosen head's linear layer. Validates the head name against the locked `SOIL_HEADS = ("soil_type", "moisture", "texture")` tuple at construction.
- `soil_gradcam(image_path, soil_engine, head) -> GradCAMResult` orchestrates it: preprocesses at 224×224 (matching the Phase 6 training pipeline), reads the per-head predicted label from `soil_engine.predict(...)` so the explanation targets the same label the rest of the pipeline saw, wraps + runs the CAM, and returns the same `GradCAMResult` shape as the disease version. Confidence is read from `SoilPrediction.per_head_confidence` so the caption matches Cell 9's per-head readout exactly.
- All Grad-CAM runs force `eval()` mode (no dropout / batchnorm perturbation) and explicitly enable gradients. Both vision models are FULL-PRECISION on disk (only Llama is 4-bit) so gradients flow without bitsandbytes detangling — captured in the gradcam module docstring as a Phase 9 invariant.

### Retrieved-chunk highlighting — lexical, audit-grade

- `src/explain/chunk_highlight.py:tokenize(text) -> set[str]` lower-cases, strips punctuation via a tight `[A-Za-z]+` regex, drops a locked English stopword set (template-noise words like `for / in / treatment / organic` plus soil noise-words `soil / type`), drops length-≤2 tokens, and applies a crude regular-plural normaliser (strip trailing `s` on len>3 words that don't end in `ss`) so `trees ↔ tree` matches. Irregular plurals (`leaves ↔ leaf`, `branches ↔ branch`) are deliberately NOT normalised — the chunk highlighter is "thin and transparent", not an NLP stack.
- `explain_chunks(query, retrieved_chunks) -> list[ExplainedChunk]` builds one row per hit with `rank`, `score`, `chunk_id`, `source_text / chapter / verse_or_section`, `matched_terms = sorted(query_tokens ∩ chunk_tokens)`, and `text_with_markers` — the chunk text with every matched term wrapped in `**…**` Markdown markers. The wrapper expands each matched token to also catch its regular plural form (so `leaf` in the query wraps `leafs` in the chunk; case-insensitive). Accepts both `RetrievedChunk`-style attribute access AND plain dicts (Strategy C's dict format from Phase 8).
- Why query↔chunk instead of answer↔chunk: aligning chunks to the LLM's *answer* sentences mixes two failure modes (bad retrieval vs bad generation) into one explanation signal. Aligning to the query isolates retrieval, which is what §18 actually wants explained. The generation-side faithfulness audit comes via RAGAS in Phase 11.

### Visualisation — matplotlib only

- `src/explain/visualize.py:render_vision_panel(sample_name, leaf_rgb, disease_cam, soil_rgb, soil_cams)` composes a 2-row figure: row 1 is original-leaf + disease-CAM (cols 0–1, rest hidden), row 2 is original-soil + 3 soil-head CAMs (all 4 cols). Each CAM tile is captioned with `<label> (<conf>)`.
- `render_retrieval_panel(sample_name, query, explained_chunks)` is a 2-column figure: left a horizontal bar chart of the top-k similarity scores (rank 1 at top, descending), right a monospace text listing of `#rank source ch.X v.Y` + `matched: <terms>` + a 240-char wrapped snippet of `text_with_markers`. The query is printed at the top of the listing so reviewers can read query and chunk side-by-side.
- `save_explanation(sample_name, vision_fig, retrieval_fig, out_dir=None)` writes both figures as 140-dpi PNGs under `results/explainability/<safe_sample_name>/` and closes the figures (notebook memory hygiene). Default root is `results/explainability/` — committed because `.png` files there are paper-reference figures, not raw data.
- Matplotlib-only so the same code surfaces inline in the Colab notebook AND inside the Phase 10 Streamlit UI's `st.pyplot` — no code duplication.

### Tests — 17 / 17 passing, no GPU, no network

- `tests/explain/test_smoke.py` (3) — every public symbol carries a non-empty docstring; `SOIL_HEADS` stays locked to the three Phase 6 heads; the legacy `compute_gradcam` / `highlight_chunks` back-compat shims still callable so older callers don't break.
- `tests/explain/test_chunk_highlight.py` (8) — `tokenize` lower-cases + drops punctuation, drops every word in the locked stopword set, regular-plural normalisation works for `trees ↔ tree` and refuses to over-normalise `grass` (ss-terminating); `explain_chunks` returns `matched_terms = query ∩ chunk` per row, wraps each matched term with `**…**` markers preserving original case, handles a no-overlap chunk cleanly (matched_terms == [], text_with_markers identical to raw text), accepts both `RetrievedChunk` attribute access AND plain dicts (the Strategy C return shape from Phase 8), assigns 1-based ranks in input order.
- `tests/explain/test_gradcam_targetlayer.py` (6) — `find_target_layer` prefers `.conv_head`, falls back to `.blocks[-1]` with a logged WARNING, raises `AttributeError` on a backbone with neither attribute or with an empty `.blocks` list; `SoilHeadWrapper` rejects an unknown head name and accepts each of the three locked Phase 6 head names. Uses tiny stub objects standing in for a timm EfficientNet — no torch needed for these tests (the actual `nn.Module` construction is exercised in the Colab notebook).
- `pytest tests/explain/ -q` runs in <0.1 s.

### Notebook — 12 cells per Phase 9 prompt §C

1. Markdown — goal (§18), what Grad-CAM + chunk highlighting show, paper relevance, deferred scope (no pointing-game, no answer-grounded highlighting, no sentence-level alignment inside the answer — all Phase 11).
2. Setup — clone + pip install (Phase 7/8 deps + `grad-cam>=1.5` + `matplotlib>=3.7`).
3. HF auth (private chunks + gated Llama + Phase 5/6 weights).
4. GPU check.
5. Load disease (B4) + soil (B0 multi-task) engines; assert disease class names are not `class_<i>` placeholders (defends against Phase 8 bug 1 regression).
6. Phase 7 RAG pipeline (corpus 206 chunks across 4 books, per-book counts asserted; HybridRetriever; Llama-3.1-8B 4-bit).
7. Demo inputs — three REAL distinct PlantDoc test images (`Tomato leaf late blight`, `Corn rust leaf`, `Potato leaf early blight`) + Phantom-fs soil images (Alluvial, Black, Red). Asserts every path exists, predicts per-sample, asserts the three disease labels are distinct — refuses to proceed otherwise (Grad-CAM over a wrong-disease placeholder is a misleading figure).
8. `disease_gradcam` per sample, with original + heatmap side-by-side, captioned with `<label> (idx=<i>, conf=<f>)`.
9. `soil_gradcam` × 3 heads per sample — prints the per-head label/idx/conf table, then a 4-tile row per sample (original + 3 head heatmaps).
10. Strategy A query construction via `build_multimodal_context` → `TemplateStrategy.build_query` → top-5 retrieval → `explain_chunks`. Prints per-sample `QUERY`, then per-chunk `#rank score source ch.X v.Y` + `matched: <terms>`.
11. `render_vision_panel` + `render_retrieval_panel` + `save_explanation` per sample. Saves to `results/explainability/<sample>/{vision_panel,retrieval_panel}.png`. Prints saved paths.
12. Markdown — how to read the saved panels (disease CAM should focus lesion; the three soil-head CAMs should attend different regions; bar chart should be monotone descending; chunks with `(no overlap)` are pure dense-retrieval hits, not a bug), scope deferrals, Phase 10 / 11 handoff.

### Phase 9 end checks (all green)

- `python -c "from src.explain.gradcam import disease_gradcam, soil_gradcam, SoilHeadWrapper; from src.explain.chunk_highlight import explain_chunks; print('ok')"` → `ok` (no model load).
- `python -c "import nbformat; print(len(nbformat.read('notebooks/phase9_explainability.ipynb',as_version=4).cells))"` → `12`.
- `pytest tests/explain/ -q` → `17 passed in 0.06s`.
- `git status` — `src/explain/*`, `notebooks/phase9_explainability.ipynb`, `tests/explain/*`, `results/explainability/.gitkeep`, `scripts/build_phase9_notebook.py`, `progress.md` staged. NO chunk text / vector_db / PDFs / weights.
- Single local commit titled `"Phase 9: explainability layer (Grad-CAM disease+3 soil heads, retrieved-chunk highlighting)"`. **No git push.**

---

## Phase 8: multimodal integration (template + LLM-mediated query, embedding ablation) + C5 causal hook

Phase 8 is the core-novelty wiring for **contribution C2** (joint disease + soil context module with three ablated query-construction strategies) and **contribution C5** (cause-conditional retrieval — pathway user-supplied, never image-derived). Built `src/integration/` end-to-end, the 13-cell Colab notebook `notebooks/phase8_multimodal_integration.ipynb`, and 22 unit tests. Reuses Phase 5 disease + Phase 6 soil + Phase 7 RAG code by import; corpus + models are read-only here. All paths under `src.utils.paths`; logging via `src.utils.logging_setup`. Decision matrix locked: existing class-based API kept (`MultimodalContext`, `CausalPathway` enum, `TemplateStrategy.build_query()` etc.); vestigial OLID causation-dataset + transforms code dropped (~280 LOC + 2 tests removed); Cell 6 asserts the post-Phase-3b.2 corpus snapshot (`206 chunks across 4 books` with explicit per-book counts) instead of the prompt's outdated `>285` check.

### Mechanism — three strategies on the same multimodal context

- **Strategy A — Template (`strategy_template.py`).** Deterministic plain-text render: `"Organic treatment for {disease} affecting {crop} grown in {soil_type} soil that appears {moisture} with {texture} texture[, <pathway clause>]."`. The `UNKNOWN` causal pathway deliberately yields **no** clause so the default is bias-free. Helper `_humanise_label` strips dataset noise (`___`, trailing `_Soil`) so queries don't read like CSV headers.
- **Strategy B — LLM-mediated (`strategy_llm_mediated.py`).** Reuses Phase 7's already-loaded Llama-3.1-8B (no second model load) to rewrite the structured context into ONE retrieval query that bridges modern vision labels (`Tomato___Leaf_Mold`, `Alluvial_Soil`) to descriptive / classical-text vocabulary (`scorched leaves with whitish lesions`, `fertile riverine soil`). Prompt rules baked in: must NOT propose a treatment, must stay within the structured context, must reflect the causal pathway only when not `UNKNOWN`. Temperature 0.2 + fixed seed → reproducible. The generator-agnostic `_invoke_llm` tries `.generate / .complete / __call__` so the strategy survives a future swap to Llama-3.2-3B or any other LLM exposing one of those methods.
- **Strategy C — Multimodal embedding projection (`strategy_multimodal_embedding.py`, honest ablation).** Concatenates penultimate B4 disease feature (1792-d) + B0 soil feature (1280-d) + bge-large crop-name embedding (1024-d), trains a single `MultimodalProjector` linear layer (4096 → 1024) on WEAK pairs — per-sample target is the bge-embedding of Strategy A's top-1 retrieved chunk. No manual relevance labels. Trained for 80 epochs MSE in the notebook (~10s). Retrieval is cosine similarity against the same ChromaDB collection Phase 7 uses. The module docstring explicitly frames C as *"the ablation that demonstrates *why* text queries win for a text corpus, not a competitor to A/B"*. The modality gap (visual embeddings live on a different manifold from bge text embeddings) is too wide for one linear layer trained on a handful of weak pairs to close; B ≥ A > C is the expected qualitative outcome.

### C5 causal hook — wired through all three strategies

`CausalPathway` enum (4 values: `SOIL_DRIVEN / PEST_VECTOR / CONTAGION / UNKNOWN`, default `UNKNOWN`) and the frozen `CausalContext(pathway, notes)` dataclass are user-input ONLY — the system never infers cause from images. Threading:

- Strategy A appends a deterministic pathway clause: `SOIL_DRIVEN → "with emphasis on soil restoration, nourishment, and root health"`, `PEST_VECTOR → "...pest deterrence..."`, `CONTAGION → "...preventing the spread..."`, `UNKNOWN → ""`. The empty clause for `UNKNOWN` is locked by `test_template_no_causal_clause_when_unknown`.
- Strategy B injects the pathway into the rewrite prompt via `_PATHWAY_HINTS` so the LLM can shape the query around it. `UNKNOWN` falls through as *"not specified"* so no cause is forced.
- Strategy C does not consume the pathway directly (the C-ablation is about the visual modality), but the same MultimodalContext threads through, so a future cause-conditional embedding can be slotted in without touching `compare.py`.

### Builder + side-by-side runner

- `build_multimodal_context(leaf_image, soil_image, crop_type, causal_pathway=UNKNOWN, ...)` orchestrates Phase 5 disease + Phase 6 soil inference into a populated `MultimodalContext`, including the penultimate visual embeddings needed by Strategy C. Accepts pre-built inference engines (the notebook builds them once and reuses) or constructs them on demand from HF repo IDs. Capture-embeddings is on by default — cheap, same forward pass.
- `compare.run_all_strategies(ctx, rag_pipeline, projector=None, ...)` is the public side-by-side driver. Strategy C is silently skipped when `projector` / `chroma_collection` / `embedder` aren't all supplied (so unit tests can exercise A + B without GPU). Returns `dict[strategy_name, StrategyResult]` with `query`, `retrieved_chunk_ids`, `retrieved_sources`, optional `answer`, optional `citations` — fields tuned so the notebook's comparison cell can pretty-print without further unpacking.
- `compare.qualitative_compare(results_per_sample, relevant_book_ids=None)` is the QUALITATIVE read for the supervisor demo — counts `on_topic_count` = how many of the k retrieved chunks come from a plausibly-relevant IKS book (defaults to all 4). Explicit on rigorous evaluation: *"the rigorous RAGAS context_precision/recall read happens in Phase 11"*.

### Phase 6 soil-inference plumbing finished as part of Phase 8

The Phase 8 prompt assumed Phase 6 single-image inference existed, but `src/soil/infer.py:predict_image()` was still stubbed with `NotImplementedError("Phase 6 — Week 19")`. Built it as part of Phase 8: `SoilInferenceEngine` loads `SoilMultiTaskClassifier` from HF (`ankit-iiitdmj/iks-soil-multitask-v2`), preprocesses to 224×224 with ImageNet stats, runs the 3 visual heads (soil_type 7-class / moisture 3-class / texture 3-class), and returns `SoilPrediction` + optional 1280-d backbone embedding. Same pattern as `DiseaseInferenceEngine`. Also extended `DiseaseInferenceEngine` with `predict_with_embedding()` returning the 1792-d penultimate B4 feature. Both extensions are additive — no existing API touched.

### Tests — 22 / 22 passing, no GPU, no network

`tests/integration/test_smoke.py` (8) covers import surface + `CausalPathway` enum invariance + `MultimodalProjector` shape + docstring guard against scope creep. `test_template.py` (10) is the Strategy-A behavioural matrix: every structured field appears in the rendered query, dataset-noise stripped, deterministic, causal-pathway clause matrix (parametrised over 4 pathway values), empty-crop edge case. `test_compare_smoke.py` (7) uses a stub `RAGPipeline` + stub LLM + stub retriever + stub `DiseasePrediction` / `SoilPrediction` (NO model load) to exercise `run_all_strategies` shape, C5 threading into Strategy A, Strategy B's reuse of `pipeline.generator`, the `relevant_book_ids` filter in `qualitative_compare`, and the explicit error when the pipeline has no generator. `pytest tests/integration/ -q` runs in <2 s.

### Phase 11 deferrals (documented in Cell 13 of the notebook)

- **No RAGAS context_precision / context_recall scoring.** That requires the expert-curated gold-query set Dr Pandey is putting together; will be Phase 11.
- **No formal causal-conditioning ablation.** C5 is wired + demoed on one sample; the formal "with vs without causal clause" sweep is Phase 11.
- **No trained multimodal embedding space.** A CLIP-style contrastive projector on image / classical-text pairs is out of scope for the thesis; Phase 8 ships the honest linear-layer ablation to show *why* B/A beat C, not to compete with them.

### Notebook structure — 13 cells per Phase 8 prompt §C

1. Markdown — goal (C2 + C5), research-backed scope (A/B main, C ablation), the modern→classical bridge problem.
2. Setup — clone repo + pip install (Phase 7 deps + timm + pillow).
3. HF auth (private chunks + gated Llama + Phase 5/6 weights).
4. GPU check + nvidia-smi.
5. Load disease (B4) + soil (B0 multi-task) engines from HF Hub.
6. Load Phase 7 RAG pipeline; **assert chunk count == 206 with 4-book breakdown** (the Phase-3b.2 snapshot).
7. Demo inputs — 3 Pillow stand-in samples covering 3 causal pathways (`SOIL_DRIVEN`, `PEST_VECTOR`, `UNKNOWN` control). Optional upload widget commented out for the deterministic run.
8. `build_multimodal_context` per sample — print structured outputs + embedding shapes.
9. Strategy A — templated query + retrieved sources + grounded answer.
10. Strategy B — LLM-bridged query + retrieved sources + grounded answer; **explicitly prints A's query alongside B's** so the vocabulary bridge is visible.
11. Strategy C — train weak projector + retrieve top-k; print final loss + per-sample top-5.
12. Side-by-side comparison — `qualitative_compare` flat table + per-sample strategy ranking by on-topic count.
13. Markdown — findings, scope notes (no RAGAS, no causal sweep, no trained embedding space), and what comes in Phase 9 / 10 / 11.

### Why no Colab run was attempted in this session

The Phase 8 prompt locks "Colab run ~30–45 min" as a separate step. The notebook is wired but unexecuted — the Colab run is gated on the Phase 7 re-run that's still pending (the supervisor demo flows from "show Phase 7 Tesseract → Phase 7 Gemini → Phase 8 multimodal" in order). Running Phase 8 first would burn Colab compute on a notebook the supervisor walkthrough won't reach for a day or two yet.

### Phase 8 end checks (all green)

- `python -c "from src.integration.compare import run_all_strategies; from src.integration.context import build_multimodal_context; print('ok')"` → `ok` (import without model load).
- `python -c "import nbformat; print(len(nbformat.read('notebooks/phase8_multimodal_integration.ipynb',as_version=4).cells))"` → `13`.
- `pytest tests/integration/ -q` → `22 passed in 1.70s`.
- `git status` — `src/integration/*`, `src/soil/infer.py`, `src/disease/infer.py`, `notebooks/phase8_multimodal_integration.ipynb`, `tests/integration/*`, `scripts/build_phase8_notebook.py`, `progress.md` staged. NO chunk text / vector_db / PDFs / weights.
- Single local commit titled `"Phase 8: multimodal integration (template + LLM-mediated query, embedding ablation) + C5 causal hook"`. **No git push.**

---

## Phase 3b: register Krishi Parashara + Upavanavinoda for external Gemini OCR

Phase 3 shipped 2 books (Vrikshayurveda + Brihat Samhita, 285 chunks) via local Tesseract OCR. Phase 7's first end-to-end queries surfaced a recurring failure mode — Tesseract's Devanagari-confused output occasionally derailed the grounded generator even when retrieval scores were excellent (rainfall query, etc.). Phase 3b registers two new books for a higher-quality OCR path (Gemini 3.5 Flash) **without** running any OCR or changing the chunker / metadata schema.

**What changed (config + one loader branch — no OCR in this commit):**

- `configs/corpus/books.yaml` — two previously-pending entries replaced with full `status: ready_external` registrations. Each carries `ocr_method: gemini_external`, `scope: pages`, a verified `page_range`, and a `text_source` path under `corpus/ocr_external/`:
  - **Krishi Parashara** (Majumdar & Banerji 1960, Bibliotheca Indica) — PDF pages 94–119 (English translation block).
  - **Upavanavinoda** (Sarngadhara, tr. Majumdar, IRI Indian Positive Sciences No. 1) — PDF pages 77–96 (English translation block).
- `src/rag/corpus/build_corpus.py` — added `_chunks_for_external_book(book)` and a branch in the main loop that runs *instead of* Tesseract for any `ocr_method: gemini_external` book. When `text_source` is absent the loader logs `"<id>: awaiting Gemini OCR at <path>, skipping."` and continues — never errors. When present, it light-cleans + chunks + embeds exactly like the Phase 3 path; chunker + metadata schema unchanged. `READY_STATES = {"ready", "ready_external"}` keeps the existing 285 chunks untouched on re-runs.
- `corpus/ocr_external/` — new directory with a tracked `README.md` documenting the contract (one `.md` per book, English-only, verse-numbered). `.gitignore` excludes `*.md` here (master plan §38 — copyrighted translation text never enters the public repo); the chunks travel via the private `ankit-iiitdmj/iks-corpus-chunks` HF dataset.
- `tests/rag/test_books_config.py` — 5 new tests guarding the registry shape, the Phase 3b contract on both new entries, and the missing-text-source skip path (returns `None` with an "awaiting" log, no exception). All pass.

**What's NOT in this commit (intentional):**

- No Gemini API calls — the actual re-OCR + transcript-cleaning + `<id>.md` writes happen in a separate Phase 3b.2 step (cost-gated; ~$0.0007/page × ~46 pages ≈ $0.03).
- No new chunks in ChromaDB — both new books are skipped today.
- No re-run of the full `build_corpus` pipeline (the existing 285 chunks are scheduled to be replaced once Gemini OCR lands; running the pipeline now would just re-embed about-to-be-discarded vectors. The pytest IS the proof that the skip path works).
- `kashyapiyakrishisukti` + `text_six_tbd` remain `status: pending` placeholders.

**Why this design:** the pipeline stays config-driven (master plan §15 — adding a book is one YAML entry, not a code change). External-OCR books and local-OCR books coexist transparently — the loader branches purely on `ocr_method`, everything downstream (clean → chunk → metadata-tag → embed → ChromaDB) is identical. Phase 7's grounded-generator code does not need to know how a chunk was OCR'd.

---

## Phase 7: grounded RAG pipeline (hybrid retrieval + Llama-3.1-8B), Colab + private chunk dataset

Phase 7 (master plan §17) wires the Phase 3 corpus into a complete grounded-advisor pipeline: hybrid retrieval (dense BGE + sparse BM25 + cross-encoder rerank with RRF fusion) → Llama-3.1-8B-Instruct 4-bit generator under a strict §17 grounded-advisor system prompt → source-cited answer plus the retrieved chunks shown for transparency.

### Platform note (recorded in memory `feedback-chromadb-torch-windows-dll`)

Phase 7 runs on **Colab / Linux, single-process**, not on the Windows laptop. On Windows `chromadb` and `torch`/`sentence_transformers` cannot coexist in one Python process — either import order produces a silent `0xC0000005` or a `cygrpc` DLL failure. On Linux the conflict doesn't manifest, so the Colab notebook uses the simple single-process design.

### PART 1 — Chunk transport (laptop, executed in this session)

`scripts/push_corpus_chunks.py` reads both Phase 3 JSONL files (`vrikshayurveda.jsonl` 78 + `brihat_samhita.jsonl` 207 = **285 rows**), validates that every row carries the full Phase 3 schema (`source_text`, `edition`, `chapter`, `verse_or_section`, `topic_tags`, `original_language`, `translator`, `chunk_id`, `text`), adds a `book_id` column, and pushes a single `train` split to the new private dataset **`ankit-iiitdmj/iks-corpus-chunks`**. Upload finished in 13 seconds (517 KB parquet). Re-running is idempotent — `Dataset.push_to_hub` overwrites the same snapshot. Master plan §38 forbids redistributing the translation text via the public GitHub repo; the private HF dataset is the §38-compliant transport.

### PART 2 — RAG code (`src/rag/`)

- `corpus_loader.py` — `load_chunks_from_hf(repo=DEFAULT_CHUNKS_REPO)` pulls the 285-row dataset into a list of dicts; `build_chroma(chunks, persist_dir)` re-embeds with `BAAI/bge-large-en-v1.5` and upserts into a fresh ChromaDB collection (`iks_corpus`), idempotent by deterministic sha1 `chunk_id`.
- `retriever.py` — `HybridRetriever(collection, use_dense, use_sparse, use_reranker, ...)` runs dense (Chroma) + sparse (rank-bm25 over an in-memory BM25Okapi index) → Reciprocal-Rank-Fusion at the canonical `k=60` → cross-encoder rerank via `BAAI/bge-reranker-base`. All three stages are **independently toggleable** so Phase 11 §27 ablations are config flips, not rewrites. `RetrievedChunk` is the shared dataclass (also consumed by `src/explain/chunk_highlight.py`).
- `generator.py` — `GroundedGenerator` loads `meta-llama/Llama-3.1-8B-Instruct` 4-bit via bitsandbytes (nf4 + double-quant + bf16 compute) and uses the locked `SYSTEM_PROMPT_V17` master-plan-§17 prompt: answer only from retrieved passages, cite `[Source Text, ch.X, v.Y]` (NOT `[chunk_id]` — citations are human-legible source coordinates), step-by-step organic protocol, refuse with the locked sentence when the retrieved passages don't cover the question, never introduce treatments absent from context. `extract_citations()` parses the answer back to `(source, chapter, verse)` strings; `used_chunk_ids` is the back-link to the retrieved chunks. `LlamaGenerator` is kept as a thin alias for backward compatibility.
- `pipeline.py` — `RAGPipeline(retriever, generator, default_k=5)` is a thin orchestrator with a **model-agnostic generator hook** (the Phase 8 multimodal-context plug-in point). `RAGAnswer.to_dict()` is JSON-friendly. Swapping LLM is one constructor arg.

The Phase-4 stub classes (`DenseRetriever`, `BM25Retriever`, `Chunker`, `Embedder`, `CrossEncoderReranker`) are kept as docstring-bearing aliases / stubs in `src.rag` so the existing `tests/rag/test_smoke.py` import surface keeps passing — the live retrieval logic lives entirely inside `HybridRetriever`.

### PART 3 — Notebook + tests

`notebooks/phase7_rag_pipeline.ipynb` (12 cells, generated by `scripts/build_phase7_notebook.py`): setup → HF auth → GPU check → corpus rebuild from HF Hub → HybridRetriever build → **retriever smoke (5 queries, runs BEFORE Cell 9 so a retrieval failure fails fast)** → Llama-3.1-8B 4-bit load (warm-up in cell so VRAM is visible) → end-to-end RAG on **5 demo queries** including the deliberately out-of-corpus query that the §17 prompt requires the model to refuse rather than hallucinate → reading guide + Phase 8 hand-off + LLM-swap notes.

Tests:

- `tests/rag/test_retriever.py` — 7 tests, no GPU / no network. Uses a stub embedder (1-hot over a tiny vocab) + a fake Chroma collection + a stub cross-encoder. Verifies: all-stages-off raises, RRF fuses higher when a chunk appears in BOTH stages, single-stage toggle changes the score-label, reranker reorders, and the RRF formula matches Cormack 2009 (`1/(60+1) + 1/(60+1)` for a double-match).
- `tests/rag/test_pipeline_smoke.py` — 5 tests. Stub retriever + stub generator. Verifies the pipeline's `RAGAnswer` shape, the query-pass-through, that `used_chunk_ids` only contains chunks whose `(source, chapter, verse)` actually appears in the answer, and that constructing without either a retriever or a collection raises.

`pytest tests/rag/test_retriever.py tests/rag/test_pipeline_smoke.py -q` → 12 passed. Full `tests/rag/` → **41 passed** (12 new + 29 from Phase 3).

### End checks all pass

- ✅ Private `iks-corpus-chunks` dataset exists with 285 rows + the documented schema.
- ✅ `python -c "from src.rag.pipeline import RAGPipeline; print('ok')"` imports cleanly (no model load).
- ✅ Notebook has exactly 12 cells.
- ✅ Phase 7 tests pass without GPU or network.
- ✅ No chunk text, vector_db, or PDFs staged (all gitignored from Phase 3).
- ✅ No `git push`. Llama-3.1-8B weights are pulled on Colab at Cell 9, not from this session.

Phase 8 next: the multimodal-context plug-in (Phase 5 disease classifier output + Phase 6 soil classifier output + user question → enriched query → same `RAGPipeline.answer(...)`). The seam is the query-construction step; nothing in `src/rag/` needs to change.

---

## Phase 3: IKS corpus pipeline + 2 books ingested (Vrikshayurveda + Brihat Samhita 12 chapters)

Phase 3 (IKS corpus, master plan §15) was deferred while Phases 5–6 (vision modules) ran first. This entry covers building the OCR → clean → chapter-split → chunk → embed → ChromaDB pipeline and ingesting the **2 books currently in hand**. The remaining 4 books drop in later via one YAML entry each — no code change needed.

**Books processed:**

| Book | Pages OCR'd | Scope | Chapters located | Chunks | Embedded |
|---|---:|---|---|---:|---:|
| Vrikshayurveda (Surapala, tr. Nalini Sadhale, AAHF 1996) | 101 | full | – | 78 | 78 |
| Brihat Samhita Part 1 (Varahamihira, tr. M. Ramakrishna Bhat, MLBD) | 593 | chapters 21–29, 40, 54, 55 | **12/12 ✓** | 207 | 207 |

**Total**: 285 vectors in ChromaDB `iks_corpus` collection. The 12 wanted Brihat chapters were all located by English-heading scan, immune to the ~8–45 page-offset drift between PDF and printed pages.

**Pipeline (`src/rag/corpus/`):**

- `ocr.py` — pdf2image + pytesseract at 300 dpi (`--oem 1 --psm 6 lang=eng`), per-page cache at `corpus/raw/<book_id>/page_NNNN.txt` so re-runs skip done pages. Windows Tesseract + Poppler binaries auto-discovered via env vars, common install paths, and a glob over `C:\poppler-*\Library\bin`.
- `cleaning.py` — drops Devanagari lines (U+0900–U+097F majority), running headers/footers, standalone page numbers, and OCR noise (<3 alpha chars); de-hyphenates line-break splits; collapses whitespace.
- `chapter_split.py` — `locate_chapters` scans cleaned page text for both the English chapter title AND the Roman-numeral marker (XXI, XXIV, LV, …); returns a `{chap_num: ChapterSpan}` dict in document order, half-open `[start, end)` page spans. Missing chapters log a WARN and are omitted, never crash.
- `chunking.py` — verse-first then paragraph-pack, 200–500 tokens, never split mid-sentence; verse markers like `"1.", "2."` detected per paragraph. `chunk_id = sha1(book_id|chapter|verse|first40chars)` so re-runs upsert idempotently.
- `embed.py` — `BAAI/bge-large-en-v1.5` (1024-dim) via sentence-transformers; upsert into ChromaDB `PersistentClient` at `corpus/vector_db/`, collection `iks_corpus`.
- `build_corpus.py` — orchestrator entry point: `python -m src.rag.corpus.build_corpus` reads `configs/corpus/books.yaml`, processes every `status: ready` book end-to-end, writes per-book JSONL + a build-wide `_manifest.json`.
- `query_smoke.py` — 3 hard-coded retrieval test queries; **two-subprocess design** to dodge a Windows-specific torch+grpc DLL conflict (`chromadb` and `sentence_transformers` cannot coexist in the same Python process here without segfaulting). Subprocess A embeds the queries, subprocess B opens Chroma and queries — the encoder process segfaults at teardown (`0xC0000005`) AFTER writing its JSON; the orchestrator treats "file exists" as the real success signal.

**Config (`configs/corpus/books.yaml`):** all 6 books listed; 2 `status: ready`, 4 `status: pending`. Each entry carries `scope: full` or `scope: chapters` (with a `chapters:` list and a `chapter_titles:` dict for the English-heading scan). Adding the remaining 4 books later is one YAML entry each.

**Tests (`tests/rag/`):** 21 new — 8 cleaning, 6 chunking, 7 chapter_split (incl. the critical page-offset-invariance test). Full `pytest tests/rag/ -q` → **29 passed** (8 pre-existing tests also still green).

**Retrieval smoke** (`python -m src.rag.corpus.query_smoke`): all 3 test queries return the right sources at top-3:

| Query | Top-1 source | Expected |
|---|---|---|
| "how to treat a diseased tree" | Vrikshayurveda v.160.3 | Vrikshayurveda / Brihat ch.55 ✓ |
| "signs that predict rainfall" | Brihat Samhita ch.28 (Signs of Immediate Rain) | Brihat ch.21–28 ✓ |
| "how to find underground water" | Brihat Samhita ch.54 (Exploration of Water Springs) | Brihat ch.54 ✓ |

**Wall time:** ~115 min total across two sessions (the build was interrupted ~halfway through by a power cut; the per-page OCR cache + idempotent sha1 chunk_ids made the resume cheap — Vrik OCR's 101 pages re-loaded from cache instantly, then Brihat OCR'd from scratch in ~40 min and 207-chunk embedding on CPU took ~75 min).

**Gitignored / not committed:** `data/corpus_pdfs/` (copyrighted PDFs), `corpus/raw/` (OCR cache), `corpus/vector_db/` (binary Chroma store). All regenerable from `configs/corpus/books.yaml` + `python -m src.rag.corpus.build_corpus`.

**Pending books** (placeholders in `books.yaml`, no PDFs yet): Krishi Parashara, Upavanavinoda, Kashyapiyakrishisukti, TBD-sixth. Each becomes a code-free addition.

---

## Phase 6 V3-tiling: patch-based texture expansion (single-stage, V2 recipe)

**NOTE:** the earlier 3-stage sequential-transfer V3 (`PHASE6_V3_SEQUENTIAL_TRANSFER_PROMPT.md`) was run on Colab and **catastrophically collapsed** all three heads to ~20% val accuracy. That multi-stage / curriculum / per-stage-head-freezing pattern is ABANDONED. V3-tiling takes a fundamentally different, safer approach: the V2 training recipe is reused **byte-for-byte**, single-stage; the only thing that changes is the texture training data (source images are tiled into a grid of patches, with image-level split integrity to prevent leakage).

### Part 1 — tiled texture dataset (executed in this session)

`src/soil/tiling.py` adds `tile_image(img, grid)`, `check_patch_resolution(images, grid)`, and `build_tiled_split(items, grid)`. Patches are 224×224 LANCZOS resamples of the per-image grid cells; every patch carries its source image's label **and** a unique `source_id` so callers can assert no source image's patches cross train/val/test boundaries.

`scripts/build_tiled_texture_dataset.py` loads `ankit-iiitdmj/iks-soil-texture-irsid-vit` (the V1 prep dataset), assigns each source image a `source_id` of the form `f"{split}_{idx:04d}"` (split prefix guarantees disjointness by construction), tiles every split independently, runs an explicit `_assert_split_disjointness` check, and pushes to the new private repo `ankit-iiitdmj/iks-soil-texture-tiled` using the proven `Dataset.push_to_hub` streaming pattern from `feedback_hf_dataset_uploads.md`. Resolution guard fires when the median patch is below 120 px pre-resize and aborts with `exit 2` unless `--force` is passed.

**Run result (this session):** the resolution guard fired at the prompt's default `grid=4` — median source min-dim was 101 px (median image ~506×407), implying a 2.2× upscale to 224. Per the prompt's working style ("stop and ask if the resolution guard fires"), Ankit was asked and chose **grid=3**. The build then finished in 38 s and pushed **2,511 patches** (train 2007 / val 252 / test 252, from 223 / 28 / 28 source images respectively) across 3 parquet shards (~43 MB total private dataset). Audit JSON written to `data/soil/tiled_texture_audit.json`. Disjointness assertion passed.

### Part 2 — V3-tiling training stack (notebook + thin wrapper)

`src/soil/train_v3_tiling.py` is a thin wrapper that re-exports V2's training functions unchanged plus two new pieces:

- `SoilCheckpointManagerV3Tiling` — subclass of V2's manager pinned to the new model repo `ankit-iiitdmj/iks-soil-multitask-v3-tiling` (created on first Cell 8 run from Colab; not from this prompt session).
- `health_check(val_metrics, threshold=0.40)` — raises `RuntimeError` if any head's val top-1 falls below 40%. Called after the FIRST epoch's val pass in the notebook's Cell 9 so a collapse aborts in minutes, not after 10+ hours.

`notebooks/phase6_soil_training_v3_tiling.ipynb` (13 cells) is identical to the V2 notebook except: Cell 6 swaps the texture dataset to `iks-soil-texture-tiled`, Cell 9 calls `health_check` after epoch 1, Cell 12 prints a V2-vs-V3-tiling table + automatic SHIP/KEEP verdict and reports BOTH patch-level and **image-level (majority vote over a source image's patches)** texture metrics. Backbone, optimizer, scheduler, augmentation, Mixup/CutMix, label smoothing, epoch count = byte-for-byte the V2 recipe.

### Untouched

V1 (`iks-soil-multitask`) and V2 (`iks-soil-multitask-v2`) model repos, original `iks-soil-texture-irsid-vit` dataset repo, and every V1/V2 source file or notebook are **unchanged in git status and on HF Hub**.

### Tests

`pytest tests/soil/test_tiling.py tests/soil/test_train_v3_tiling_smoke.py -q` → 17 passed. Covers patch counts/shapes, grid validation, resolution-guard warning + safe path, `build_tiled_split` label/source_id propagation, the **two leakage tests** (disjoint by construction + regression that detects bad construction), `health_check` boundary cases at three thresholds, and a one-step train smoke through V2's loop. Full `tests/soil/` → 51 passed.

**Ship criteria** (Cell 12 evaluates and prints "SHIP V3-TILING" / "KEEP V2"): texture top-1 strictly improves AND neither soil_type nor moisture drops more than 2 pts from V2's TTA test baseline (soil_type 89.92% / 0.851, moisture 95.76% / 0.958, texture 67.86% / 0.678). If V3-tiling doesn't clear the bar, V2 remains production and V3-tiling is preserved as a documented negative result for the paper's ablation table.

Expected Colab T4 wall-time: ~6–12 hours (same as V2). Epoch-1 health check makes the cost of a failure ~30 minutes, not 10+ hours.

---

## Phase 6 V3 prep: sequential transfer learning experiment

Notebook `notebooks/phase6_soil_training_v3.ipynb` generated (14 cells, built via `scripts/build_phase6_v3_notebook.py`). 3-stage sequential transfer learning:

- **Stage A** — Phantom-fs only (15 epochs, soil_type head; 5 frozen + 10 unfrozen)
- **Stage B** — + Sirajganj moisture (15 epochs, soil_type + moisture heads; all unfrozen)
- **Stage C** — full multi-task (20 epochs, all 3 heads; all unfrozen) → final V3 model

Hypothesis: a backbone warmed on Indian soil_type then moisture is more soil-aware than ImageNet pretraining alone, lifting the texture head. Expected texture gain +2–5 points. **This is an experiment** — if V3 doesn't clearly beat V2 it gets shelved as a documented negative result and V2 ships.

V3 reuses V1/V2 building blocks unchanged — `src/soil/train_v3.py` only composes them into the staged schedule (`SoilCheckpointManagerV3`, `build_stage_loader`, `train_stage_a/b/c` thin wrappers over V2's `train_one_epoch_v2`). Per-stage loss masking is **data-driven**: Stage A loads only Phantom-fs, so moisture/texture labels are all `-1` and their heads get zero gradient via V2's `ignore_index=-1` NaN guard; Stage B adds Sirajganj (texture still `-1`); Stage C adds texture. A fresh CosineAnnealingLR is created per stage (`T_max` = that stage's epoch count). All stages use V2's strong augmentation + Mixup/CutMix (p=0.3) + label smoothing 0.1; TTA (5 views) is used for the final test eval only.

V1 and V2 source files + notebooks are **untouched** (verified via git status). V3 pushes to NEW repo `ankit-iiitdmj/iks-soil-multitask-v3` (created on first Cell 8 run from Colab, not from this prompt session). V1 (`iks-soil-multitask`) and V2 (`iks-soil-multitask-v2`) repos stay intact for the paper's ablation.

V2 baseline (TTA) to beat: soil_type 89.92% / 0.851, moisture 95.76% / 0.958, texture 67.86% / 0.678. **Success criteria** (Cell 13 auto-checks + prints "SHIP V3" / "REVERT TO V2"): texture top-1 up ≥3 pts AND neither soil_type nor moisture drops >2 pts.

Tests: `pytest tests/soil/test_train_v3_smoke.py -q` → 5 passed (one-epoch smoke of each stage + A→B→C history chaining + V3 repo-id check). Full `tests/soil/` → 34 passed.

Expected Colab T4 wall-time: ~15–20 hours total across the 3 stages.

---

## Phase 6 V2 prep: augmentation-boosted retraining notebook ready

Notebook `notebooks/phase6_soil_training_v2.ipynb` generated (13 cells, built via `scripts/build_phase6_v2_notebook.py`). Goal: push the texture head from V1's **67.86% test top-1 / 0.670 macro F1** toward the 75–82% range without changing the locked EfficientNet-B0 backbone or 224×224 input size. soil_type (V1 89.08% / 0.818) and moisture (V1 88.98% / 0.890) may shift ±1–2 points because the backbone is shared — that's expected.

V2 adds:

- **Strong augmentation** (`src/soil/transforms_v2.py`): wider-scale RandomResizedCrop (0.7–1.0), VerticalFlip, Rotate ±30°, ShiftScaleRotate, GridDistortion/ElasticTransform (one-of, p=0.3), stronger ColorJitter, GaussianBlur/GaussNoise (one-of), CoarseDropout. Re-written against albumentations 2.x's new signatures (`RandomResizedCrop(size=...)`, `CoarseDropout(num_holes_range=...)`, `GaussNoise(std_range=...)`) with equivalent magnitudes to the spec's albumentations 1.x example.
- **Mixup + CutMix at batch level** (`src/soil/mixup.py`): `maybe_apply_mix(p=0.3, mixup_alpha=0.2, cutmix_alpha=1.0)` selects 50/50 between Mixup and CutMix when triggered. `mixed_loss()` blends per-head losses across the two label dicts.
- **Label smoothing 0.1** on cross-entropy in `compute_multitask_loss_smoothed` (`src/soil/train_v2.py`). Same `ignore_index=-1` NaN guard as V1.
- **Test-Time Augmentation** in Cell 12: `build_tta_views()` returns 5 deterministic albumentations Composes (original, HFlip, VFlip, Rot90, Rot270). `evaluate_per_task_tta()` averages logits across views before argmax.
- **40 epochs total** (V1 was 30) — heavier augmentation slows convergence. Warmup stays at 5 frozen-backbone epochs.

V1 source files are **untouched** — `src/soil/{transforms,train,model,dataset,__init__}.py` and `notebooks/phase6_soil_training.ipynb` are unchanged so the paper's ablation can compare V1 vs V2 with the exact V1 model on `ankit-iiitdmj/iks-soil-multitask`. V2 pushes to **`ankit-iiitdmj/iks-soil-multitask-v2`** (new private repo, created on first Cell 9 run from Colab — not from this prompt session).

Tests: `pytest tests/soil/test_mixup.py tests/soil/test_transforms_v2.py tests/soil/test_train_v2_smoke.py -q` → 13 passed. Covers Mixup/CutMix shapes + lam range, `maybe_apply_mix` p=0 / p=1 branches, `mixed_loss` blend math, TTA returns 5 distinct views, strong-aug is stochastic, label-smoothed loss + `train_one_epoch_v2` CPU smoke through Mixup path.

Expected Colab T4 wall-time: ~8–15 hours total (1–2 sessions), up from V1's 6–12 due to the extra 10 epochs and the small per-step overhead of the Mixup/CutMix collation. Resume-aware via HF Hub checkpoints just like V1.

---

## Phase 6 training notebook ready (B0 edition)

Notebook `notebooks/phase6_soil_training.ipynb` generated via `scripts/build_phase6_notebook.py`. Joint multi-task training on **EfficientNet-B0 at 224×224** per master plan §22 with 3 task heads: `soil_type` (7 classes), `moisture` (3 classes), `texture` (3 classes). Each head is `nn.Sequential(Dropout(0.3), Linear(1280, n_classes))`; total params 4,024,201 (4.0M backbone + 16,653 across the three heads — the headline "~5.3M B0 params" you see in the timm docs is for the original 1000-class ImageNet classifier, which we drop via `num_classes=0`).

Training code in `src/soil/` mirrors `src/disease/`:

- `src/soil/model.py` — `SoilMultiTaskClassifier` (timm B0 + 3 heads + `freeze_backbone()` / `unfreeze_backbone()`).
- `src/soil/train.py` — `TASK_WEIGHTS`, `compute_multitask_loss` (NaN-guarded ignore_index=-1), `train_one_epoch`, `evaluate_per_task`, `SoilCheckpointManager`, `auto_batch_size` (returns 64/32/16 by VRAM).
- `src/soil/transforms.py` — `build_soil_train_aug` (RandomResizedCrop + HFlip + Rotate ±15 + mild ColorJitter + Normalize) and `build_soil_eval_aug` (Resize 256 → CenterCrop 224).
- `src/soil/__init__.py` — exports the full Phase 6 API alongside the existing Phase 4 helpers.

Per-sample loss masking: each batch sample supervises exactly one head; the other two receive `-1` and are ignored by `F.cross_entropy(ignore_index=-1)`. When **all** samples in a batch carry `-1` for a head, cross-entropy normally returns NaN; the helper substitutes a graph-consistent zero so backward still produces a zero gradient on that head without poisoning the total loss.

12-cell notebook covers: setup → HF auth → GPU + auto-batch → load 3 HF datasets → transforms + ConcatDataset train loader + 3 per-task val/test loaders → model build → optimizer + scheduler + scaler + `SoilCheckpointManager` with resume → 30-epoch loop (5 frozen + 25 full unfrozen, per-task losses + per-task val metrics logged per epoch, latest + best checkpoints pushed to `ankit-iiitdmj/iks-soil-multitask` HF Hub repo) → held-out test-set evaluation (mirrors Phase 5 fix, separate `eval_metrics_test.json` file).

Expected wall-time on Colab T4: 6–12 h total. Resume-aware so 1–2 sessions work. HF Hub model repo created on first Cell 9 run from Colab (not in this prompt session).

Tests: `pytest tests/soil/` → 16 passed (3 new: `test_model.py`, `test_loss_masking.py`, `test_train_eval_smoke.py` — covering construction, forward shapes, freeze toggle, loss NaN-guard, end-to-end train+eval CPU smoke).

---

## Phase 6 prep: soil data uploaded to HF Hub

Three private dataset repos created on Hugging Face Hub:

- `ankit-iiitdmj/iks-soil-phantomfs` (soil_type, 7 classes, 1,188 images) — 6 parquet shards, 397 MB
- `ankit-iiitdmj/iks-soil-sirajganj-moisture` (moisture_appearance, 3 classes, 1,177 images) — 3 shards, 188 MB
- `ankit-iiitdmj/iks-soil-texture-irsid-vit` (texture, 3 USDA-collapsed classes, 279 images: 16 IRSID + 263 VIT) — 3 shards, 27 MB

Splits: stratified 80/10/10, seed=42 (`sklearn.train_test_split`). Phantom-fs train=951/val=119/test=119, Sirajganj train=941/val=118/test=118, Texture train=223/val=28/test=28. Each parquet row carries the image (HF `Image()` column), `label_idx`, `class_name`, and a `source` column ('phantomfs' / 'sirajganj' / 'irsid' / 'vit') for ablation.

Sirajganj and texture were pre-resized to max-dim 768 (JPEG q=90) before encoding — full-res phone photos pushed sirajganj's single parquet shard to ~450 MB which reliably crashed the LFS upload mid-transfer across three attempts. Phantom-fs stayed at native resolution because its first upload (full-res) had already succeeded by the time the resize policy was added; training-time pipeline crops to 224×224 either way.

Combined channel norm stats (`configs/data/soil_norm.yaml`) computed over the union of 2,114 train images at 224×224: `mean=[0.535, 0.459, 0.400]`, `std=[0.216, 0.200, 0.210]`.

Multi-task training will use per-sample loss masking — each row supervises exactly one head; the other two get `-1` (ignored by `CrossEntropyLoss`). Helper `src.soil.dataset.build_multitask_labels()` produces the `{soil_type, moisture_appearance, texture}` dict per sample. Class index configs at `configs/data/soil_{soil_type,moisture,texture}_classes.yaml`.

Tests: `pytest tests/data/test_soil_hf_datasets.py` → 12 passed.

---

## VIT texture dataset integration (Phase 5/6 boundary)

Added latha-soil (Reddy & Gopinath, Nature Sci. Rep. 2025, doi:10.1038/s41598-025-17384-5) as a supplementary texture-axis dataset alongside the IRSID Kaggle mirror. Local-only — no Hugging Face Hub push in this session (deferred to Phase 6 prep). Paper claims 4,000 images; the public GitHub release at `https://github.com/phd-latha/latha-soil` (commit `14a1fe2`) contains **263 images across 7 classes**: see `data/soil/vit_texture/INTEGRATION_AUDIT.json` for the full breakdown.

Class counts after canonicalisation: clayey_soils 40, loamy_sand_soil 40, loamy_soil 39, sandy_clay_soil 33, sandy_loam 36, sandy_soil 40, silt_soil 35. 0 files routed to `_review/`; 0 PIL-rejected.

Both datasets are kept on disk as separate directories (`data/soil/vit_texture/` and `data/soil/irsid/`); Phase 6 training code will combine them via PyTorch `ConcatDataset`, not by filesystem merging. The new `vit_texture:` section in `configs/data/soil_texture_label_mapping.yaml` maps each class to coarse / fine / mixed using the same USDA-soil-triangle logic as the existing IRSID block.

Decision: integrate as-is; email VIT authors in parallel asking whether the full 4,000-image release is hosted elsewhere.

---

## Phase 4 fix (post-Weeks 14–15) — Reconciliation with finalised scope

- OLID I: switched source from Zenodo (19 archives) to Kaggle `raiaone/olid-i` (single zip) and downloaded the **full 4,749 images / 23 multi-label classes** (was smoke-sample 83/3). `_labels_for()` updated to split compound symptoms like `bottle_gourd__JAS_MIT`.
- Sirajganj 2025 added as net-new (Mendeley DOI 10.17632/skcc44yvvg.2): 1,177 images / 3 classes (dry/moderate/wet) supervising the `moisture_appearance` head.
- Soil module heads pinned to 3: `soil_type` + `moisture_appearance` + `texture`. Dropped `surface` and `cover` per the supervisor-signed-off soil-parameter coverage audit. `texture` survives via the IRSID → coarse/fine/mixed mapping in `configs/data/soil_texture_label_mapping.yaml`.
- Phantom-fs 7-class verified: Alluvial / Arid / Black / Laterite / Mountain / Red / Yellow (the Phase 4 prompt's "Clay" and "Peat" do not exist upstream).
- PlantDoc 28th class (`Tomato two spotted spider mites leaf`) found to be a **vestigial 2-image folder**, documented, not modified.
- `requirements.txt` gained matplotlib / jupyter / nbformat / iterative-stratification / requests / kaggle.
- `results/dataset_stats.md` (228 lines) and `notebooks/dataset_eda.ipynb` (with a real OLID 23×23 multi-label co-occurrence heatmap) regenerated and end-to-end-executed.
- New `PHASE4_SUMMARY.md` at the repo root supersedes the previous one.
- Total disk: 30 GB (over the 25 GB envelope because Sirajganj v2 grew to 4.49 GB vs the prompt's ~500 MB estimate).

---

## Phase 4 (Weeks 14–15) — Dataset Acquisition & Preprocessing

- 6 datasets acquired: PlantVillage, PlantDoc, Paddy Doctor, Phantom-fs Soil, IRSID, OLID I (smoke sample)
- Splits generated: 5 standard 80/10/10 stratified splits + 1 cross-region soil split (Phantom-fs train, IRSID test)
- Normalisation stats computed per dataset (configs/data/*_norm.yaml)
- Augmentation pipelines defined (disease modest, soil heavier, causation geometric-only)
- Dataset classes implemented (JSONIndexedImageDataset + MultiLabelImageDataset, factory functions per dataset)
- Validation: 0 corrupt files across 185,735 images scanned
- Total disk: 5.4 GB (well under the 20 GB cap)
- TODO: full OLID I (~14 GB across 19 Zenodo archives) deferred to Phase 11 — flip `OLID_FULL_DOWNLOAD = True` in `scripts/download_olid_i.py` and re-run when C5 evaluation begins

---

## Week 2 (continued) — PDF-alignment cleanup

- Removed [ADDED] engineering hygiene (pre-commit, GitHub Actions CI, pyproject.toml + tool configs, decisions/, session reports)
- Fixed paper/thesis nesting per §41
- Added §42 references.bib, §43 BACKUP.md, §44 weekly + monthly journal templates, notebooks/00_environment_check.ipynb
- Rewrote requirements.txt to track §22 exactly
- Rewrote environment.yml to mirror requirements.txt
- All tests still passing post-cleanup

### Post-cleanup repository tree (`find . -maxdepth 2 -type d`)

```
.
./configs
./configs/disease
./configs/eval
./configs/integration
./configs/rag
./configs/soil
./corpus
./corpus/chunks
./corpus/cleaned
./corpus/raw
./corpus/vector_db
./data
./data/plant_disease
./data/soil
./data/splits
./demo
./models
./notebooks
./notes
./notes/cv
./notes/iks
./notes/rag
./notes/xai
./paper
./research_journal
./research_journal/daily
./research_journal/monthly
./research_journal/weekly
./results
./results/figures
./results/logs
./scripts
./src
./src/disease
./src/eval
./src/explain
./src/integration
./src/rag
./src/soil
./src/utils
./tests
./tests/disease
./tests/eval
./tests/explain
./tests/integration
./tests/rag
./tests/soil
./tests/utils
./thesis
```

Matches §41 exactly; extras (`notes/`, `tests/`, `research_journal/{daily,weekly,monthly}`) are all `[PDF-implied]` or `[PDF §44]`.

---

## Week 1 — Project Setup
**Dates:** May 15 - May 21, 2026

### Completed Tasks
- [x] Repository initialized on GitHub
- [x] Complete folder structure created with all subdirectories
- [x] `requirements.txt` written with all dependencies (PyTorch, transformers, RAG, evaluation tools)
- [x] `environment.yml` created for Conda environment (`iks-agri`)
- [x] `.gitignore` configured (data/, models/, vectors, pycache, etc.)
- [x] README.md written with overview, architecture, setup, and references
- [x] Configuration files created (`disease_config.yaml`, `soil_config.yaml`, `rag_config.yaml`)
- [x] Python module structure initialized (`__init__.py` in all src/ subdirectories)
- [x] Skeleton implementations: logger.py, config.py, model stubs
- [x] Streamlit app skeleton created (`demo/app.py`)
- [x] Environment check notebook created (`notebooks/00_environment_check.ipynb`)

### Blockers / Issues
- None encountered during setup

### Notes
- Seed=42 set globally for reproducibility
- All YAML configs include detailed annotations for future reference
- Soil model includes critical warnings about visual-only analysis (no NPK/pH prediction)

### Next Week Goals
- [ ] Set up development environment (create conda env, verify imports)
- [ ] Literature review: CV basics (transfer learning, ResNet, fine-tuning techniques)
- [ ] Literature review: RAG fundamentals (embeddings, vector databases, chunking strategies)
- [ ] Literature review: XAI techniques (Grad-CAM, attention mechanisms, interpretability)
- [ ] Schedule supervisor meeting with Dr. Akshay Pandey
- [ ] Identify and download PlantVillage and Soil datasets
- [ ] Sketch initial experiment plan for Phase 2 (disease module)

---

## Week 2 — Foundation Infrastructure
**Dates:** May 20 - May 26, 2026

### Completed Tasks
- [x] §1 Locked-stack `pyproject.toml`, regenerated `requirements.txt`,
      `requirements-dev.txt`, `.python-version`, `INSTALL.md`.
- [x] §2 Reproducibility utilities in `src/utils/`: `set_global_seed`,
      project paths, stdlib logging, Pydantic v2 `BaseConfig`. Unit
      tests cover seeding reproducibility, strict-extra config validation,
      directory creation, logger handler attachment.
- [x] §3 Pydantic config schemas + `configs/<module>/default.yaml` for
      disease, soil, rag, integration, eval. Soil schema enforces
      disallowed chemical outputs (guardrail #2). RAG config locks the
      Llama-3.1-8B / 4-bit / BGE choices. Integration config flags
      `require_causal_context` (contribution C5).
- [x] §4 Module skeletons for disease, soil, rag, integration, explain,
      eval — every public class, dataclass, and function has a
      NumPy-style docstring and a `NotImplementedError("Phase X — Week Y")`
      pointer. `src/rag/prompts.py` ships a working citation-enforcing
      prompt template and renderer. `src/eval/citation_verification.py`
      has a working extractor + minimal verifier.
- [x] §5 Testing infrastructure: `tests/` mirrors `src/`, shared
      `conftest.py` with `tmp_corpus_dir`, `seeded_rng`,
      `tiny_dummy_image`, `sample_retrieved_chunks` fixtures. Each
      module has an instantiation smoke test.
- [x] §6 Code quality + CI: pre-commit (ruff, ruff-format, black, mypy,
      file-hygiene hooks), GitHub Actions matrix on Python 3.11 + 3.12.
      Ruff/black/mypy/pytest config lives in `pyproject.toml`.
- [x] §7 Notes templates under `notes/{cv,rag,xai,iks}/` with the
      standard skeleton (Key concepts / Papers read / Tricky bits / Open
      questions / Code I want to try).
- [x] §8 Research ops: this Week 2 entry, `literature_tracker.csv`
      (empty), `research_journal/` with a first daily entry,
      `decisions/0001..0003.md` ADRs.

### Blockers / Issues
- None for the scaffolding. Running the full `pytest` / `pre-commit`
  loop locally was not attempted in this session — see
  `WEEK2_SUMMARY.md` for what to verify on the M.Tech workstation.

### Notes
- README, requirements.txt, and environment.yml from Week 1 referenced
  ResNet50 + LangChain. Both have been brought in line with the locked
  stack (EfficientNet-B4/B0, plain-Python RAG).
- ADR-0001 (EfficientNet backbones), ADR-0002 (Pydantic over Hydra),
  ADR-0003 (no LangChain in RAG) capture the reasoning.

### Next Week Goals
- [ ] Run `pip install -e ".[dev]"` on the workstation and verify
      `pytest -q`, `ruff check .`, `black --check .`, `mypy src/`,
      `pre-commit run --all-files` are all green.
- [ ] Start filling `notes/cv/transfer_learning.md` and `notes/cv/efficientnet.md`.
- [ ] Begin literature_tracker.csv (target: 20 rows by end of Week 3).
- [ ] Supervisor meeting with Dr. Pandey to walk through ADR-0001..0003.

---

## Phase Milestones

### Phase 1: Foundation (Weeks 1-3)
- **Goal:** Establish development environment and foundational knowledge
- **Key Activities:**
  - Environment setup and dependency validation ✅ (Week 1)
  - Literature review on CV, RAG, and XAI
  - Supervisor alignment on approach and timeline
  - Dataset acquisition and initial exploration
- **Deliverable:** Full working development environment + knowledge summary document

### Phase 2: Disease Detection Module (Weeks 4-7)
- **Goal:** Implement and validate plant disease classification
- **Key Activities:**
  - PlantVillage dataset exploration and preprocessing
  - ResNet50 fine-tuning on disease dataset
  - Grad-CAM integration for model explainability
  - Validation metrics and baseline performance
- **Deliverable:** Trained disease model (>90% accuracy on test set) + Grad-CAM visualization

### Phase 3: Soil Analysis Module (Weeks 8-10)
- **Goal:** Implement multi-task soil visual analysis
- **Key Activities:**
  - Soil dataset preparation and balancing
  - Multi-task architecture design (soil type + texture + surface + moisture)
  - Training pipeline with multi-task loss
  - Performance baseline on each task
- **Important:** Model predicts ONLY visual attributes; does NOT claim NPK/pH/fertility
- **Deliverable:** Trained multi-task soil model + per-task evaluation metrics

### Phase 4: RAG Pipeline (Weeks 11-14)
- **Goal:** Build hybrid retrieval system over classical agricultural texts
- **Key Activities:**
  - Digitized text preprocessing (Vrikshayurveda, Krishi Parashara, Upavanavinoda)
  - Sentence-window chunking with semantic tagging
  - Embedding model selection and fine-tuning
  - Dense + BM25 hybrid retrieval implementation
  - LLM integration (Llama-3.1-8B) with prompt engineering
- **Deliverable:** Functional RAG pipeline + retrieval baseline evaluation

### Phase 5: Integration & Optimization (Weeks 15-17)
- **Goal:** Unify all components and optimize for real-time inference
- **Key Activities:**
  - End-to-end system integration
  - Latency profiling and optimization
  - Streamlit web interface refinement
  - Error handling and edge case management
- **Deliverable:** Deployable web app with <5s total inference time

### Phase 6: Evaluation & Ablation (Weeks 18-20)
- **Goal:** Rigorous evaluation using established metrics
- **Key Activities:**
  - RAGAS evaluation framework (Faithfulness, Relevance, Context Recall)
  - Expert annotation of recommendations for groundtruth
  - Ablation studies (disease only vs. soil+disease vs. full system)
  - Comparison with template-based baselines
- **Deliverable:** Comprehensive evaluation report with tables/plots

### Phase 7: Thesis Writing & Defense (Weeks 21-24)
- **Goal:** Document research and prepare for defense
- **Key Activities:**
  - Literature review chapter
  - Methodology chapter (architecture, datasets, training procedures)
  - Results chapter (with tables, confusion matrices, case studies)
  - Discussion and conclusions
  - Final revisions and formatting
- **Deliverable:** Complete thesis manuscript

---

## Known Constraints & Notes

1. **Soil Module Limitation:** The multi-task soil classifier predicts ONLY visually observable attributes (soil type from color/texture, surface condition, moisture appearance). It CANNOT predict NPK, pH, or soil fertility—these require lab testing. This is documented in README.md and all relevant code files.

2. **Classical Texts:** Vrikshayurveda, Krishi Parashara, and Upavanavinoda form the grounding corpus. Digitization and cleaning are prerequisites for RAG.

3. **Reproducibility:** All random seeds set to 42; dependency versions pinned in requirements.txt for reproducibility across systems.

4. **Supervisor:** Dr. Akshay Pandey, CSE Department, IIITDM Jabalpur

5. **Disclaimer:** This is a research prototype. Not for production use without expert validation.

---

## Communication Log

| Date | Contact | Topic | Outcome |
|------|---------|-------|---------|
| (TBD) | Dr. Akshay Pandey | Project kickoff meeting | - |

---

**Last Updated:** May 15, 2026
**Status:** Week 1 — Setup Phase Complete ✅
