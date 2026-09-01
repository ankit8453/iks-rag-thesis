# Claude Code Prompt — Phase 9: Explainability Layer (Grad-CAM + retrieved-chunk highlighting)

> Paste below the horizontal rule into Claude Code in a fresh session on the laptop. It builds `src/explain/` and a Colab notebook that produces visual explanations for the vision models and the retrieval step. Agent time ~35 min. Colab run ~20–30 min.

---

## CONTEXT

Phase 9 (master plan §18) adds the explainability layer: *where* each vision model looked
(Grad-CAM) and *why* each chunk was retrieved (chunk highlighting). These visuals are both a
paper selling point (the honest interpretability claim in §35: "interpretability via Grad-CAM
and chunk highlighting") and the components the Phase 10 Streamlit UI will display.

**What already exists (reuse, do NOT rebuild):**
- Disease model: EfficientNet-B4 @ 380×380, HF `ankit-iiitdmj/iks-disease-plantdoc`. Inference
  in `src/disease/infer.py` (`DiseaseInferenceEngine`, `InferenceResult`). NOT quantized → Grad-CAM works.
- Soil model: EfficientNet-B0 @ 224×224, multi-task (3 heads: soil_type, moisture, texture),
  HF `ankit-iiitdmj/iks-soil-multitask-v2`. Inference in `src/soil/infer.py`
  (`predict_image(...) -> SoilPrediction`). NOT quantized → Grad-CAM works.
- RAG pipeline: `src/rag/pipeline.py` `RAGPipeline` (HybridRetriever + grounded generator).
  Corpus = 206 chunks / 4 books in private HF dataset `ankit-iiitdmj/iks-corpus-chunks`.
- Phase 8 integration: `src/integration/` (TemplateStrategy etc.) to build the query whose
  retrieved chunks we explain.

**Platform: Colab/Linux** (same as Phases 7–8: corpus rebuild + models on T4). Grad-CAM itself
is light, but keeping it on Colab matches the working setup and avoids the Windows chromadb/torch issue.

**Hard rules:**
- All paths via `src.utils.paths`. Logging via `src.utils.logging_setup.get_logger`.
- **Local commits only — never `git push`.**
- Reuse Phase 5/6/7/8 code by import; do NOT reimplement vision inference, the RAG pipeline, or query construction.
- Models/corpus are READ-ONLY here.

---

# Mission

Build `src/explain/` producing: (1) Grad-CAM heatmap overlays for the disease model and for
ALL THREE soil heads, and (2) a retrieved-chunk explanation that shows each chunk's source +
verse + similarity score and highlights the query terms that matched. Demonstrate both in a
Colab notebook on sample (leaf, soil) pairs.

## Locked Decisions

| # | Decision |
|---|---|
| 1 | Grad-CAM on: disease (predicted class) + soil_type + moisture + texture (4 heatmaps per input pair). |
| 2 | Chunk explainability: similarity score + source_text/chapter/verse + highlight overlapping query↔chunk terms; plus a small top-k score bar chart. |
| 3 | Use `pytorch-grad-cam`. Target layer = last conv of each timm EfficientNet (inspect the model to find it; for timm EfficientNet usually `model.conv_head` or `blocks[-1]`). |
| 4 | Soil model is multi-task → wrap each head so Grad-CAM can target a single head's logits. |
| 5 | Colab notebook + `src/explain/` code. Single local commit. No push. |

---

## Deliverables

### `src/explain/gradcam.py`
- Helper to locate the Grad-CAM target layer of a timm EfficientNet backbone (inspect
  `named_modules()`; prefer the final conv before global pool — `conv_head` if present, else
  the last block's last conv). Provide a small function `find_target_layer(backbone) -> nn.Module`
  with a clear fallback + a logged warning if it has to guess.
- `disease_gradcam(image_path, engine) -> dict{overlay_rgb, heatmap, pred_label, pred_conf}`:
  - Preprocess exactly as the disease engine does (380×380, same normalization).
  - Grad-CAM w.r.t. the predicted class. Return the heatmap overlaid on the original image (RGB array).
- `class SoilHeadWrapper(nn.Module)`: wraps the multi-task soil model and a head name; `forward(x)`
  returns ONLY that head's logits (so pytorch-grad-cam's ClassifierOutputTarget works on it).
- `soil_gradcam(image_path, soil_model, head) -> dict{overlay_rgb, heatmap, pred_label, pred_conf}`
  for head in {soil_type, moisture, texture}: preprocess at 224×224 (same norm as training),
  wrap the head, Grad-CAM w.r.t. that head's predicted class.
- All Grad-CAM runs: model in eval mode, gradients enabled (NOT inside torch.no_grad), input on
  the model's device. These models are full-precision (only the LLM is 4-bit), so gradients flow.

### `src/explain/chunk_highlight.py`
- `tokenize(text) -> set[str]`: lowercase, strip punctuation, drop English stopwords, optional
  light stemming (no heavy NLP deps — a small stopword set + simple normalization is fine).
- `explain_chunks(query, retrieved_chunks) -> list[dict]`: for each chunk return
  {rank, score, source_text, chapter, verse_or_section, matched_terms (query∩chunk),
   text_with_markers (chunk text with matched terms wrapped in **…** markers)}.
- Pure-Python, no model needed → unit-testable without GPU/network.

### `src/explain/visualize.py`
- `render_vision_panel(disease_cam, soil_cams: dict) -> matplotlib Figure`: a row showing the
  original leaf + disease heatmap, and the original soil + the 3 soil-head heatmaps, each
  captioned with predicted label + confidence.
- `render_retrieval_panel(explained_chunks) -> Figure`: a panel listing the top-k chunks with
  source/verse, the matched terms, and a horizontal bar chart of the similarity scores.
- `save_explanation(sample_name, vision_fig, retrieval_fig, out_dir)`: save PNGs to
  `results/explainability/<sample_name>/`. (results/ is fine to commit; it's figures, not data.)

### `notebooks/phase9_explainability.ipynb` (12 cells, nbformat)
1. **Markdown** — Phase 9 goal (§18), what Grad-CAM + chunk highlighting show, paper relevance.
2. **Setup** — clone, pip install (add `grad-cam`, `matplotlib`; rest already present).
3. **HF auth** — token (models + private chunks dataset).
4. **GPU check.**
5. **Load vision models** — disease B4 + soil B0 multi-task from HF.
6. **Load RAG pipeline** — Phase 7 `RAGPipeline` (rebuild ChromaDB from `iks-corpus-chunks`,
   206 chunks / 4 books; hybrid retriever). Print chunk count.
7. **Demo inputs** — 3 (leaf image, soil image, crop) sample pairs. IMPORTANT: use REAL,
   DISTINCT per-crop leaf images from the PlantDoc test set (rice/cereal, tomato, mango or other),
   NOT a single placeholder — otherwise the Grad-CAM heatmaps explain a wrong disease label and
   the figures are misleading. Print each sample's source image path + predicted disease name.
8. **Disease Grad-CAM** — `disease_gradcam` on each sample; show original + heatmap + label/conf.
9. **Soil Grad-CAM ×3** — `soil_gradcam` for soil_type, moisture, texture on each sample's soil image.
10. **Retrieval explanation** — build the query (Phase 8 TemplateStrategy or B), retrieve top-k,
    run `explain_chunks`; print matched terms + scores; render the retrieval panel.
11. **Combined figure** — `render_vision_panel` + `render_retrieval_panel` per sample, saved to
    results/explainability/. These are the figures the Phase 10 UI will surface and the paper can use.
12. **Markdown** — notes: Grad-CAM confirms the model attends to lesion/soil regions (sanity of
    the vision claim); chunk highlighting makes retrieval auditable; reminder that demo images
    must be real (not placeholder) for honest figures; next = Phase 10 (Streamlit UI wraps these).

### Tests (`tests/explain/`)
- `test_chunk_highlight.py`: tokenize drops stopwords + punctuation; explain_chunks returns
  matched_terms = query∩chunk and wraps them with markers; handles a chunk with zero overlap.
- `test_gradcam_targetlayer.py`: `find_target_layer` returns a conv layer for a tiny stub timm
  EfficientNet (or a monkeypatched module tree) and logs a warning on fallback. No GPU/network.
- Run `pytest tests/explain/ -q`.

### progress.md + commit
- Append Phase 9 entry: Grad-CAM for disease + 3 soil heads; chunk highlighting (matched terms +
  scores); figures saved to results/explainability/; demo should use real per-crop images.
- Stage: `src/explain/*`, `notebooks/phase9_explainability.ipynb`, `tests/explain/*`,
  `results/explainability/.gitkeep`, `progress.md`. (Do NOT stage chunk text / vector_db / PDFs / weights.)
- Commit message: `"Phase 9: explainability layer (Grad-CAM disease+3 soil heads, retrieved-chunk highlighting)"`
- **No git push.**

---

## End Checks (must all pass)
- [ ] `python -c "from src.explain.gradcam import disease_gradcam, soil_gradcam, SoilHeadWrapper; from src.explain.chunk_highlight import explain_chunks; print('ok')"` imports (no model load).
- [ ] `python -c "import nbformat; print(len(nbformat.read('notebooks/phase9_explainability.ipynb',as_version=4).cells))"` == 12.
- [ ] `pytest tests/explain/ -q` passes (no GPU, no network).
- [ ] Notebook Cell 7 prints DISTINCT source image paths + DISTINCT predicted disease names across the 3 samples (guards against the placeholder-image bug from Phase 8).
- [ ] `git status`: no chunk text / vector_db / PDF / weights staged.
- [ ] `git log --oneline -1` starts with "Phase 9:".
- [ ] No `git push`.

## Working Style
- Plan first (numbered). No push.
- Reuse Phase 5/6/7/8 code by import — do not duplicate inference or retrieval.
- For the multi-task soil model, Grad-CAM MUST target a single head via SoilHeadWrapper — a raw
  multi-output forward will break ClassifierOutputTarget.
- Use REAL distinct per-crop leaf images in the demo; a Grad-CAM heatmap over a wrong-disease
  placeholder is a misleading figure — flag loudly if only a placeholder is available.
- Stop and ask if: the Grad-CAM target layer can't be confidently located in the wrapped models;
  the vision models appear quantized (they should NOT be — only the LLM is 4-bit); the chunks
  dataset shows ≠ 206.

## What success looks like
1. For each (leaf, soil) sample: a disease heatmap + 3 soil-head heatmaps, each captioned with
   the predicted label/confidence, over REAL distinct images.
2. A retrieval panel showing top-k chunks with source/verse, matched query terms highlighted, and a score bar chart.
3. Combined figures saved to results/explainability/ — ready for the Phase 10 UI and paper figures.
4. Tests pass without GPU/network; one unpushed commit.
5. Ready for Phase 10 (Streamlit UI surfaces these explanations live).

Begin.
