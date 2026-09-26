# Master Experiment Log — IKS-Grounded Multimodal Agricultural Advisory System

**M.Tech Thesis · IIITDM Jabalpur · Author: Ankit Pawar · Supervisor: Dr. Akshay Pandey**

> **What this document is.** The single results-first ledger for the whole project. Every experiment we run goes here with: what we tried, the hypothesis, the method, the numbers we got, and a clear **PASS / FAIL / verdict**. Read the *Current State* section first to know where we stand; read the *Component Results* tables for the headline numbers; read the *Disease-model diagnosis* section for the deepest piece of analysis (it is the most paper-critical).
>
> **Companion docs.** `progress.md` = narrative weekly log (engineering detail per phase). `research_journal/daily/*` = day-by-day "ran / worked / didn't work". This file = the structured results ledger that ties it together. Keep all three updated.
>
> **Last updated:** 2026-09-15 (see §6m — first clean-corpus Phase 11 baseline).

---

## 1. Current State (read this first)

**System goal.** Upload a leaf photo + soil photo + crop + suspected cause → get a recommendation grounded in classical Indian agricultural texts (Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda), with disease/soil predictions, Grad-CAM heatmaps, and cited source chunks.

**What works end-to-end (as of 2026-06-12):** The full pipeline runs live (Phase 10 Streamlit UI on Colab+cloudflared). Disease model → soil model → Strategy-B query rewrite → grounded RAG answer with citations → highlighted source chunks. Demonstrated working with real images.

**The open problem we are actively fixing (2026-09):** the RAG side's **over-refusal on disease queries** — 81.8% of the 22 answerable silver queries are refused, and the grounded answer rate has been frozen at 13.64% (3/22) across three corpus versions. Corpus expansion (§6j), displacement (§6k) and the chapter-span bug (§6l) have all been ruled out as causes; the live hypothesis is that the classical texts genuinely lack remedies for these fungal/bacterial diseases and the silver labels are wrong (coverage check in progress, §6m).

**Earlier open problem (resolved into a position, §6c):** the **disease classifier's PlantDoc (in-the-wild) stage**. It predicts correctly but with only 72.3% accuracy and — critically — Grad-CAM shows it attends to **background, not the leaf**. We have diagnosed the cause (see §5) and are testing the fix (LP-FT, see §6).

**Component status snapshot:**

| Component | Status | Headline number |
|---|---|---|
| IKS corpus (6 books, 233 chunks) | ✅ Clean (§6l) | brihat over-capture fixed; 327 → 233 |
| IKS RAG retrieval | ✅ Working | P@5 0.673, nDCG@5 0.878, Hit@5 1.00 (§6m) |
| Grounded answering | ⚠️ Under investigation | grounded 13.6%, over-refusal 81.8% (§6m) |
| Strategy-B query rewrite | ✅ Working | retrieval score 0.59–0.96 vs 0.01–0.04 |
| Grounded generation (Llama-3.1-8B) | ✅ Working | §17 prompt, cites source+chapter+verse |
| Soil multi-task classifier (B0) | ✅ Production (v2) | soil_type 89.9% / moisture 95.8% / texture 67.9% |
| Disease classifier (B4) | ⚠️ Works but background-biased | PlantDoc 72.3% (frontier) but off-leaf attention |
| Disease fix: LP-FT | 🔬 Running now | Target: recover accuracy + leaf attention |
| Full-system UI (Phase 10) | ✅ Working | Colab + cloudflared tunnel |

---

## 2. System Architecture

Three model families, wired by the integration layer:

1. **Disease classifier** — EfficientNet-B4 @ 380×380, 3-stage transfer cascade (PlantVillage → Paddy Doctor → PlantDoc). `src/disease/`.
2. **Soil classifier** — EfficientNet-B0 @ 224×224, multi-task (3 heads: soil_type / moisture / texture). `src/soil/`.
3. **RAG advisory** — hybrid retrieval (BM25 + bge-large dense + cross-encoder rerank) over a ChromaDB of IKS corpus chunks, grounded generation by Llama-3.1-8B (4-bit). `src/rag/`.
4. **Integration** — turns vision predictions into a retrieval query. Strategy A (template) vs **Strategy B (LLM-mediated rewrite, the winner)**. Plus a user-supplied causal pathway (contribution C5). `src/integration/`.
5. **Explainability** — Grad-CAM for disease + 3 soil heads; retrieved-chunk term highlighting. `src/explain/`.
6. **UI** — Streamlit full-system demo. `app/`.

**Key infra decisions (locked):**
- All training data + checkpoints live on HuggingFace Hub (Colab-friendly; survives free-tier session timeouts via per-epoch checkpoint push/resume).
- `GIT_LFS_SKIP_SMUDGE=1` on clone to dodge LFS bandwidth quota.
- Embedder + reranker on CPU, only Llama + vision on GPU (avoids the Phase 8 OOM).

---

## 3. Component Results

### 3.1 Disease classifier — cascade stage-by-stage (CONFIRMED by audit 2026-06-12)

| Stage | Dataset | Classes | OLD model acc | R model acc | Grad-CAM attention |
|---|---|---|---|---|---|
| 1 | PlantVillage (clean lab) | 38 | **99.8%** | 90.7% | **leaf** ✅ |
| 2 | Paddy Doctor (field canopy) | 10 | **97.0%** | 95.7% | **lesion** ✅ |
| 3 | PlantDoc (in-the-wild) | 27 | **72.3%** | 66.8% | **background** ❌ |

- OLD = standard cascade. R = background-randomization retrain (Phase 5-R) — **a failed experiment**, see §7.1.
- Accuracies measured on a 600-image random sample of each dataset's test split (2026-06-12 audit), except PlantDoc top-1 which is the 256-image audit figure.
- HF repos: `iks-disease-{plantvillage,paddy-doctor,plantdoc}` (OLD), `iks-disease-r-{...}` (R).
- **Published context:** PlantDoc SOTA across the literature is ~73–78% (Singh 2020 ~70.5%; ViT/hybrid 2025–26 ~74–77%). **Our 72.3% is at the frontier — not behind.** The problem is *how* it gets there (background), not the number.

### 3.2 Soil multi-task classifier (production: `iks-soil-multitask-v2`, TTA test)

| Head | Top-1 | Macro F1 |
|---|---|---|
| soil_type | 89.92% | 0.851 |
| moisture | 95.76% | 0.958 |
| texture | 67.86% | 0.678 |

- Texture is the weak head (V1 67.86%); V2 augmentation + a V3-tiling experiment targeted it. Texture remains the hardest head — candidate for future work.

### 3.3 RAG retrieval + integration

- **Strategy B (LLM-mediated query rewrite)** is the Phase 8 winner: retrieval score **0.59–0.96** vs Strategy A (template) **0.01–0.04**. B rewrites modern vision labels ("Apple Scab Leaf") into classical-text vocabulary ("scorched leaves with whitish spots") that the corpus actually uses.
- Corpus: ~206 chunks across 4 books, ChromaDB, bge-large-en-v1.5 embeddings.
- Generation follows the §17 grounded-advisor prompt: answer only from retrieved passages, cite `[Source Text, ch.X, v.Y]`, refuse if evidence insufficient.

---

## 4. Phase-by-Phase Journey (complete)

Quick index, then full detail per phase. Engineering minutiae: `progress.md`.

| Phase | What | Outcome |
|---|---|---|
| Week 1 | Project setup, repo scaffolding | ✅ |
| Week 2 | PDF-alignment cleanup (match thesis spec §41) | ✅ |
| 4 | Dataset acquisition + preprocessing (6 datasets) | ✅ |
| 4-fix | Reconcile datasets with finalised scope | ✅ |
| 5 | Disease cascade training (B4, 3 stages) | ✅ PlantDoc ~71–72% |
| 5-R | Background-randomization retrain + no-leaf + Grad-CAM audit | ⚠️ **FAILED as a fix** (§7.1) |
| 6-prep | Soil data → HF Hub (3 repos) + VIT texture integration | ✅ |
| 6-V1 | Soil multi-task B0 (baseline) | ✅ texture 67.86% |
| 6-V2 | Strong aug + Mixup/CutMix + label smoothing + TTA | ✅ **PRODUCTION** |
| 6-V3-seq | 3-stage sequential transfer learning | ❌ **collapsed to ~20%** (§7.2) |
| 6-V3-tiling | Patch-based texture expansion | ⚠️ experiment (§7.3) |
| 3 | IKS corpus pipeline + 2 books (Vrik + Brihat) | ✅ 285 chunks |
| 3b | Register Krishi Parashara + Upavanavinoda (Gemini OCR) | ✅ |
| 3b.2 | Gemini re-OCR → final corpus | ✅ 206 chunks / 4 books |
| 7 | Grounded RAG (hybrid retrieval + Llama-3.1-8B) | ✅ |
| 8 | Multimodal integration (Strategy A/B/C + C5) | ✅ **B wins** |
| 9 | Explainability (Grad-CAM + chunk highlighting) | ✅ surfaced bias |
| 10 | Full-system Streamlit UI (Colab + tunnel) | ✅ live demo |
| Disease fix | Step-wise diagnosis + LP-FT | 🔬 running |

### Week 1 — Project setup
Repo on GitHub; full folder structure; `requirements.txt` + `environment.yml` (conda env `iks-agri`); configs for disease/soil/rag; `src/` module skeletons; Streamlit skeleton; seed=42 global. No blockers.

### Week 2 — PDF-alignment cleanup
Aligned the repo to the thesis spec (§41): fixed paper/thesis nesting, added `references.bib`, `BACKUP.md`, journal templates, environment-check notebook. Rewrote `requirements.txt`/`environment.yml` to track §22 exactly. Tests still green.

### Phase 4 — Dataset acquisition & preprocessing
6 datasets acquired: PlantVillage, PlantDoc, Paddy Doctor, Phantom-fs Soil, IRSID, OLID I. Generated 5 stratified 80/10/10 splits + 1 cross-region soil split. Per-dataset norm stats. Dataset classes (`JSONIndexedImageDataset`, `MultiLabelImageDataset`). **Validation: 0 corrupt files across 185,735 images.**

**Phase 4 fix** (reconcile with finalised scope):
- OLID I: full **4,749 images / 23 multi-label classes** (Kaggle source).
- Sirajganj 2025 added: **1,177 images / 3 classes** (dry/moderate/wet) → moisture head.
- Soil heads pinned to **3**: soil_type + moisture_appearance + texture (dropped surface + cover per supervisor).
- Phantom-fs verified 7-class: Alluvial/Arid/Black/Laterite/Mountain/Red/Yellow.

### Phase 5 — Disease cascade training (EfficientNet-B4)
3-stage transfer cascade, 380×380, mixed precision, HF-Hub checkpoints (resume-after-timeout). PlantVillage (38) → Paddy Doctor (10) → PlantDoc (27). Original PlantDoc top-1 **~71–72%**. This is the model the whole disease-diagnosis work (§5) later dissected. Repos `iks-disease-{plantvillage,paddy-doctor,plantdoc}`.

### Phase 5-R — Background-randomization retrain ⚠️ FAILED (see §7.1)
Hypothesis: composite leaves onto random backgrounds each epoch so background can't be a label cue; add a `no_leaf` reject class (28th). Built segmentation+mask-cache + randomized dataset + cascade-R trainer + Grad-CAM audit (keep/revert rule: central-attn +5pp AND top-1 drop ≤3pp). **Result: failed the bar** — see §7.1. Repos `iks-disease-r-*`.

### Phase 6 — Soil multi-task classifier (EfficientNet-B0, 224×224, 3 heads)

**6-prep — data to HF Hub.** 3 private repos: `iks-soil-phantomfs` (soil_type, 7-class, 1,188 img), `iks-soil-sirajganj-moisture` (moisture, 3-class, 1,177 img), `iks-soil-texture-irsid-vit` (texture, 3 USDA-collapsed classes, 279 img = 16 IRSID + 263 VIT). Per-sample loss masking (each row supervises one head; others `-1`, ignored). Also integrated the latha-soil/VIT texture dataset (263 images, 7 classes).

**6-V1 — baseline.** B0 + 3 heads (`nn.Sequential(Dropout, Linear)` each), 30 epochs (5 frozen + 25 unfrozen), per-task loss masking with NaN-guard. Results (test):
- soil_type **89.08%** / 0.818 F1
- moisture **88.98%** / 0.890 F1
- texture **67.86%** / 0.670 F1 ← weak head

**6-V2 — augmentation boost → PRODUCTION.** Added strong aug (wider RandomResizedCrop, rotate ±30, GridDistortion/Elastic, stronger ColorJitter, CoarseDropout), **Mixup + CutMix** (p=0.3), **label smoothing 0.1**, **TTA** (5 views), 40 epochs. Goal: lift texture toward 75–82%. Results (TTA test):
- soil_type **89.92%** / 0.851
- moisture **95.76%** / 0.958 (big jump from V1)
- texture **67.86%** / 0.678 (texture did NOT improve — still the weak head)
- **Shipped as production** (`iks-soil-multitask-v2`) — moisture gain + no regression on the others. Texture remains the open soil problem.

**6-V3-sequential — 3-stage transfer ❌ COLLAPSED (see §7.2).** Hypothesis: warm the backbone on soil_type then moisture before full multi-task → more soil-aware → lift texture. On Colab it **catastrophically collapsed all 3 heads to ~20% val**. Abandoned; documented negative result.

**6-V3-tiling — patch-based texture expansion ⚠️ (see §7.3).** Safer retry: reuse V2 recipe byte-for-byte, only change texture data — tile source images into a 3×3 grid (2,511 patches, leakage-safe by source_id). Health-check aborts if any head <40% after epoch 1. Ship criterion: texture up AND soil_type/moisture not down >2pp. _Outcome: V2 remained production unless it cleared the bar — confirm final number from the Colab run._

### Phase 3 — IKS corpus pipeline + 2 books
OCR → clean → chapter-split → chunk → embed → ChromaDB. Ingested **Vrikshayurveda (78 chunks)** + **Brihat Samhita 12 chapters (207 chunks)** = **285 vectors** in `iks_corpus`. Pipeline (`src/rag/corpus/`): Tesseract OCR with per-page cache, Devanagari-line drop cleaning, English-heading chapter location (page-offset-invariant), verse-first chunking with sha1 idempotent chunk_ids, bge-large-en-v1.5 embeddings. Retrieval smoke: all 3 test queries returned correct sources at top-3. 29 tests pass.

### Phase 3b / 3b.2 — add 2 more books via Gemini OCR
Tesseract's Devanagari-confused output was derailing the generator on some queries. Registered **Krishi Parashara** + **Upavanavinoda** for higher-quality Gemini OCR (config + one loader branch; cost-gated ~$0.03). After 3b.2 re-OCR, the live corpus is **206 chunks across 4 books** (the number asserted by Phase 8/9/10).

### Phase 7 — Grounded RAG pipeline
Hybrid retrieval: **dense (bge-large) + sparse (BM25Okapi) → Reciprocal-Rank-Fusion (k=60) → cross-encoder rerank (bge-reranker-base)**, all toggleable. **Llama-3.1-8B-Instruct 4-bit** (nf4 + double-quant) generator under the locked §17 grounded-advisor prompt (answer only from passages, cite `[Source Text, ch.X, v.Y]`, refuse if insufficient). Chunks transported via private `iks-corpus-chunks` HF dataset (§38-compliant — no copyrighted text in the public repo). Runs single-process on Colab/Linux (Windows has a chromadb+torch DLL conflict). 41 tests pass.

### Phase 8 — Multimodal integration (C2 + C5)
Three query-construction strategies on one `MultimodalContext`:
- **A — Template** (deterministic fill-in).
- **B — LLM-mediated** (Llama rewrites vision labels → classical vocabulary). **WINNER.**
- **C — Multimodal embedding projection** (honest ablation showing the modality gap; B≥A>C as expected).
- **C5 causal hook**: user-supplied pathway (soil_driven/pest_vector/contagion/unknown), never inferred from images, threaded through A and B.
- Also finished Phase 6 soil single-image inference (`SoilInferenceEngine`) here. 22 tests pass.
- **Retrieval result: Strategy B 0.59–0.96 vs A 0.01–0.04.**

### Phase 9 — Explainability
Grad-CAM for disease + 3 soil heads (`SoilHeadWrapper` forwards one head's logits so `ClassifierOutputTarget` works on the multi-task model). Retrieved-chunk term highlighting (query↔chunk lexical overlap, audit-grade). Matplotlib panels reused in notebook + Streamlit. **This phase surfaced the disease background-bias finding** (only 3/256 PlantDoc test images had central CAM). 17 tests pass.

### Phase 10 — Full-system Streamlit UI
Live demo: upload leaf+soil → predictions → Grad-CAM → grounded cited answer → highlighted chunks. Colab + cloudflared tunnel (localtunnel dropped Streamlit's JS chunks). T4 memory plan: embedder+reranker on CPU. 13 app tests pass.

**Phase 10 FINAL (2026-06-13)** — rebuilt for the validated pipeline + modern UI:
- **Crop-first C-PD**: `iks-disease-plantdoc-crop` + YOLO `LeafCropper` (conf 0.10).
- **Heatmap only for disease**: eigen Grad-CAM (`disease_gradcam_eigen`, blocks[-2]) shown only when diseased; healthy → no heatmap.
- **Advisory gated on disease**: healthy → "no treatment needed"; diseased → Strategy B (LLM-mediated) query → grounded IKS answer + chunks.
- **Modern/futuristic CSS**: dark gradient, glassmorphism cards, gradient hero, green/red status pills.
- Files: `app/config.py` (C-PD repo, HEALTHY_CLASSES, `is_healthy`, YOLO cfg), `app/loaders.py` (`load_cropper` + `EngineBundle.cropper`), `app/streamlit_app.py` (rewrite), `src/explain/gradcam.py` (`disease_gradcam_eigen` + `eigen_smooth`), `notebooks/phase10_launch_ui.ipynb` (+ultralytics).
- **Corpus coverage caveat**: corpus = tree science (Vrikshayurveda) → covers tree/fruit diseases (apple scab → real treatment); grains (corn) → honest refusal. Demo both (integrity + novelty).

### Disease fix (active) — see §5, §6.

---

## 5. Disease-Model Diagnosis (THE key analysis — 2026-06-12)

This is the most paper-critical piece of work. A step-wise Grad-CAM + accuracy audit localized *exactly* where and why the disease model fails.

### 5.1 The symptom
Phase 9 Grad-CAM showed the PlantDoc-final disease model attends to **image corners / background**, not leaves (only ~1.2% of 256 PlantDoc test images had central attention). The model is *correct but for the wrong reasons*.

### 5.2 Ruling out a visualization artifact
We feared the Grad-CAM target layer (`conv_head`) was producing corner artifacts. Tested 3 target layers (`conv_head`, `blocks[-1]`, `blocks[-2]`) on the same images. **`blocks[-2]` picked up slightly more leaf but background corners persisted across all layers.** → The bias is **real**, not a target-layer artifact. (We did learn `conv_head` exaggerates it, so the exact "1.2%" stat is layer-sensitive — report the qualitative finding, soften the precise number.)

### 5.3 Localizing the failure to a stage (the breakthrough)
Tested each cascade stage on its own clean data:
- **PlantVillage (clean):** both OLD and R models attend to the **leaf**. 99.8% / 90.7%.
- **Paddy Doctor (canopy):** both attend to **lesions**. 97.0% / 95.7%. On images with a visible brown lesion, the heat lands *on the lesion*.
- **PlantDoc (cluttered):** attention drifts to **background**. 72.3% / 66.8%.

**Conclusion: stages 1–2 are healthy. The bias is introduced specifically at the PlantDoc fine-tuning stage.** Full fine-tuning of B4 on the small (~2k), cluttered, in-the-wild PlantDoc set distorts the good leaf-features built in stages 1–2.

### 5.4 Cross-test (clutter vs stage)
Ran the healthy Paddy-stage model on cluttered PlantDoc images. Result: even the "before" model is faint/unfocused on cluttered backgrounds (not cleanly leaf-locked). → The good features hold on clean backgrounds but **don't transfer robustly to clutter**; full fine-tuning then makes it worse. This is why background randomization alone (R) didn't fix it.

### 5.5 Literature check (done before choosing a fix — 2026-06-12)
Deep review of what actually works for background-shortcut learning + PlantDoc:
- **Background randomization** (what we tried in R): consistent with literature that it often *hurts* in-distribution accuracy. Our failure is expected, not a bug.
- **Segmentation masking** (object × mask): fixes *attention* but **not accuracy**; mask noise sabotages it; PlantVillage bias is "capture bias" that masking doesn't remove. → Not an accuracy fix.
- **LP-FT / backbone freezing** (Kumar et al., ICLR 2022; Frontiers 2026): freezing good pretrained features and training only the head beats full fine-tuning under domain shift by **11–15pp** on field PlantDoc. **Best fit for our exact diagnosis.**
- **Detect-then-crop → classify**: strongest historical PlantDoc lever (uncropped ~30% → cropped 70.5% in Singh 2020). Strong second option.
- **Realistic ceiling:** ~74–78%. We won't blow past it, but should recover ~5–6pp lost to feature distortion.

---

## 6. Active Experiment: LP-FT (Linear-Probe Fine-Tuning)

**Hypothesis.** Freezing the healthy OLD Paddy-stage backbone and training only a fresh 27-class PlantDoc head will preserve leaf-attention and recover accuracy.

**Method.** `src/disease/train_lpft.py` — freeze backbone (BatchNorm in eval to keep inherited running stats), train head-only with AdamW + cosine schedule, push to NEW repo `iks-disease-plantdoc-lpft` (old model untouched for comparison). Notebook: `notebooks/phase5_lpft_plantdoc.ipynb`. Unit tests: `tests/disease/test_train_lpft.py` (2 passing — backbone byte-identical after a step, head changes).

**Decision rule (built into the notebook's test cell).**
- LP-FT accuracy **up** (toward ~77%) AND heatmap **on the leaf** → ✅ this is the fix; build the paper on it.
- Otherwise → **Step 2: detect-then-crop**.

**Result (2026-06-12). ❌ FAILED.** LP-FT (pure linear probe, frozen Paddy backbone) scored **61%** on PlantDoc test vs OLD full-FT **72%** — an **11-point DROP**. Grad-CAM showed **no attention improvement** (still leaf+background mix, some images still corner-biased). ~25 min training.

**Why it failed:**
1. **Froze the wrong backbone** — the Paddy-stage backbone is rice-specialized (narrowed to 10 rice classes); a frozen rice-tuned backbone + linear head can't represent PlantDoc's 27 multi-crop classes. That capacity ceiling explains most of the drop.
2. **A frozen backbone + linear head cannot change *where* the model looks** — attention is set by the backbone, so the heatmap could never move (and didn't).
3. Technical note: full "LP-FT" is 2-phase (probe THEN low-LR fine-tune); we did only the probe. But phase 2 would re-introduce full-FT feature distortion on the same rice-narrow backbone — not worth it.

**Decision:** record as negative result (§7.4). Move to **Step 2 — detect-then-crop** (strongest evidence; PlantDoc ~30%→70.5% historically from cropping). Both LP-FT and background randomization failed for the same reason: they don't remove the clutter. Cropping does.

---

## 6b. Step 2: Detect-then-crop (decided 2026-06-12, after LP-FT failed)

**Research verdict (deep review):**
- **PlantDoc ships ground-truth bounding boxes** (~9,216 boxes, VOC XML, `github.com/pratikkayal/PlantDoc-Object-Detection-Dataset`).
- **The "cropped PlantDoc 70.5%" is an ORACLE** — built from ground-truth boxes (uncropped 29.7% → GT-crop 70.5%), NOT an automatic detector. No published paper hits 70.5% end-to-end. Report our automatic number honestly against this ceiling.
- **rembg/SAM are the wrong tools.** rembg = salient-object seg (grabs wrong/merged region on multi-leaf clutter; maintainer confirms you can't target an object). SAM needs a detector in front anyway. **YOLO is the class-aware localizer that wins.**
- **Pretrained YOLO exists, no training needed:** `foduucom/plant-leaf-detection-and-classification` (YOLOv8s, mAP@0.5 0.946, downloadable `best.pt`).

**Plan (one notebook, full ablation):**
1. **Prove** — crop PlantDoc test by GT boxes, run OLD B4, measure. Isolates "does cropping help" from detector quality. Headline result, ~70% ceiling.
2. **Deploy** — pretrained YOLOv8 leaf detector → crop highest-confidence box → classify. Gap to oracle = detector-quality story.
3. Report: uncropped 72% / YOLO-crop X% / GT-oracle Y%.

**Built (2026-06-12):** `src/disease/detect_crop.py` (VOC parse + crop + class-name match; 7 tests pass), `notebooks/phase5_detect_crop_plantdoc.ipynb`. **Test-only — no retraining**: uses the existing OLD model, clones the PlantDoc detection repo for GT boxes, loads pretrained YOLOv8 (`foduucom/...`), prints the no-crop / YOLO-crop / GT-oracle table + previews + Grad-CAM on crops.

**Result (2026-06-12). ❌ FAILED (inference-time cropping).**
- Detection-repo test set gave a fake **91.9%** → **data leakage** (those images were in the classifier's training set). Discarded.
- On the honest held-out HF test split (256 imgs): **no-crop 72.3%** (confirms baseline), **YOLO-crop 58.2%** — cropping makes it **14pp WORSE**.
- **Why:** the model leans on background; cropping removes that crutch and exposes its true (lower) leaf-only ability. Inference-time cropping on a background-trained model can't fix it.
- **The only way cropping helps = retrain on cropped images** (C-PD style, train+test on crops) → ~70%, i.e. SAME accuracy as now but honest leaf attention.

---

## 6c. Disease model — FINAL POSITION (2026-06-12)

**Three fixes tried, all failed for the same reason** — none retrains the model to look at leaves:
| Fix | Held-out result |
|---|---|
| Background randomization (R) | worse everywhere |
| LP-FT (freeze backbone) | 61% (−11pp) |
| Inference-time YOLO crop | 58.2% (−14pp) |

**Established facts:**
- Honest PlantDoc accuracy = **72.3%**, at the **published frontier (~74–78%)**. No method reliably beats it.
- The 72.3% is **background-driven** (rigorously shown via stage-wise Grad-CAM: stages 1–2 healthy, stage 3 breaks).
- Retraining on GT-crops would give ~70% with honest attention — same accuracy, better interpretability.

**Decision (2026-06-12, Ankit):** the goal is to FIX the damaged model (it looks at background, fails even on clean PlantVillage images), NOT to chase accuracy above 72%. A *correct* model at ~70% beats a *broken* one at 72%. → **Retrain on cropped (leaf-only) images (C-PD).** Built `src/disease/train_crop.py` (cropped-dataset builders, warm-start from PlantVillage backbone; reuses `train_one_stage`) + `notebooks/phase5_crop_retrain_plantdoc.ipynb`. Pushes to NEW repo `iks-disease-plantdoc-crop`.

**RESULT (2026-06-13). ✅ FIX WORKS (the honest tradeoff).**
- Held-out leaf-crop test (452 crops): **66.6% top-1, 0.652 macro-F1** — matches the PlantDoc paper's C-PD (70.5%) ballpark.
- Grad-CAM: **attention moved onto the leaf** (clear on several images; partial on busy multi-leaf scenes).
- Cell 3 confirmed the damage first: OLD model lights up background/edges even on CLEAN PlantVillage leaves.
- Honest-vs-cheating: OLD 72.3% via **background** ❌ vs NEW 66.6% via **leaf** ✅.
- Caveats: 66.6% (crops) vs 72.3% (full images) are different test sets/inputs — not directly comparable; deployment = crop-first then classify; the Cell-3/5 Grad-CAM is on full images (OOD for the crop-model).

**DISEASE WORK LOCKED.** Full arc: diagnosis → 3 failed fixes (R, LP-FT, inference-crop) → C-PD fix (honest ~67% leaf attention).

**KEY FOLLOW-UP FINDINGS (2026-06-13):**
- **Healthy-vs-diseased is near-perfect.** On a 16-sample check the model got health-status 16/16 right; the 27-class top-1 "66.6%" is mostly **within-crop disease-SUBTYPE confusion** (corn rust↔corn blight, potato early↔late blight), NOT healthy/diseased errors. Report top-1 66.6% AND the much-higher healthy-vs-diseased accuracy (Phase 9 Cell 5b measures it on the full test set). This reframes 66.6% from "weak" to "works where it matters."
- **Grad-CAM corner hotspots were a visualization artifact**, not the model — fixed with `eigen_smooth=True` (plain GradCAM was noisiest; GradCAM++/EigenCAM also noisy; eigen-smoothed GradCAM is cleanest).
- **YOLO leaf detector** (`foduucom/plant-leaf-detection-and-classification`) works at **conf=0.10** (0.25 was too strict). `src/disease/leaf_detect.py::LeafCropper` (shared by Phase 9 & 10).
- **"Heatmap only for disease" rule** (design decision): healthy leaves show NO heatmap (a heatmap on a healthy leaf is meaningless); diseased leaves show the eigen Grad-CAM on the lesion. Implemented as `disease_explain()` in Phase 9.
- Clean demo images: diseased leaves **idx 248, 106, 22** (PlantDoc test) give clean lesion attention. Healthy leaves give diffuse maps (correct, but we suppress the heatmap).

**Phase 9 FINAL (2026-06-13):** rebuilt the explainability notebook around C-PD + YOLO crop (conf 0.10) + eigen Grad-CAM + heatmap-only-for-disease + the healthy-vs-diseased metric + the 248/106/22 showcase. New shared helper `src/disease/leaf_detect.py`.

**Thesis framing:** present BOTH models — full-FT 72.3% (background, Grad-CAM-shown) vs C-PD 66.6% (honest leaf attention). Contribution = the **IKS-grounded advisory system + the rigorous disease-bias diagnosis + the C-PD fix**, NOT a record-breaking classifier. PlantDoc ceiling ~70–78%; honest-leaf ~67–70%.

---

## 6d. Retrieval realignment + honest scope + adaptive collection (2026-06-28)

Three changes that together fix a conceptual mismatch between how we queried the
corpus and how the corpus is actually organised, and close the system's
"unknown input" hole. All approved by the supervisor before implementation.

### 6d.1 Symptom-driven retrieval (was: crop-name-driven)
- **The mismatch.** We treated the corpus as a *crop-indexed database*: the query led with the plant name and, when that name was absent from the texts, the generator refused ("not enough information"). Our agronomy expert **Dr. Sunita T. Pandey** flagged that the classical texts are **general and symptom-based, not crop-specific** — a remedy given for a symptom applies to any plant showing it (her kunapajala work was validated on potato among many crops).
- **Verification before changing anything.** Deep literature research confirmed it (24/25 claims verified under 3-vote adversarial checking): Vrikshayurveda classifies affliction by **symptom + dosha**, not by crop, and a peer-reviewed Vrikshayurveda **expert system diagnoses purely from symptom questions and never asks the crop** (Rananavare & Chitnis, *J Ayurveda Integr Med* 2024). Nuance kept: Surapala does contain a minority of crop-named recipes (mango, pomegranate, coconut) — so the claim is **"predominantly symptom-general", never absolute.**
- **Fix (two places).** (a) Strategy B now states the texts are symptom-based, enforces **"LEAD WITH THE SYMPTOM"**, bans crop-led query patterns as poor-retrieval, and demotes the crop to *background context only*. (b) The §17 grounded prompt gains **rule 1a** scoping the refusal: a passage addressing the **observed condition** is sufficient evidence **even if it never names the crop**; refuse only when no passage addresses the condition. The faithfulness guardrail itself is unchanged — we corrected its *misapplication*.
- **Why it matters:** we were *under-reporting* real coverage (refusing cases the texts do cover), which undercut the very bridge that is the contribution.

### 6d.2 Scope gate — the farmer declares the plant
- **The flaw.** The classifier has a **fixed 27-class head and cannot say "I don't know"** — given an untrained plant it returns its closest class *confidently*. Deciding "is this plant supported?" from the model asks it a question it structurally cannot answer.
- **Fix (Ankit's design).** The **farmer selects the plant**; the **model names the disease**. Clean division of labour — the person reliably knows their own crop. Scope is settled **before any model runs**. `supported_crops()` derives the list from the model's **own `class_names`** (13 crops from the 27 classes), so the displayed scope can never drift from the checkpoint. Out-of-scope → honest *"it will not guess"* + the supported list.
- The disease label's implied crop is retained as a **cross-check only**: disagreement raises a warning (*"you selected tomato but this looks like potato"*) **without blocking** — the farmer stays the authority.

### 6d.3 Adaptive learning — collection, not live self-learning
- Supervisor asked whether the system can learn from new inputs. Adopted form: **collect → expert validates → offline retrain**. Learning directly from unverified user labels would poison the model; naive retraining also risks catastrophic forgetting.
- **Storage.** Samples go to a **private HF dataset** (`iks-feedback-samples`). Colab's disk is wiped at session end, so local storage would silently lose everything, and the end goal is a farmer-facing app rather than a notebook. Deliberately **no multi-backend abstraction** — a real DB can be swapped in when the app is built.
- **Design detail:** one **self-contained record per sample** (`images/<id>.jpg` + `records/<id>.json`) rather than appending to a shared index, so concurrent submissions cannot clash. Every sample arrives `status="pending_expert_review"`, `expert_label=None`. Upload is **best-effort and never raises** — a missing token must not break a farmer's advisory.
- **Collected:** out-of-scope plants (the farmer's typed name is the only label we have — and what makes the sample useful), low-confidence predictions, and crop mismatches.

**Validated:** `test_symptom_driven.py` (7) + `test_scope_gate.py` (5) + `test_feedback.py` (6). Full suite **325 passed**.

**Phase 11 (evaluation) remains deliberately deferred to the end** — RAGAS context_precision/recall needs the expert-curated gold-query set; Dr. Pandey's newer asks add Precision@k / Recall@k / nDCG and the no-bridge / no-grounding **baselines** (the baselines are what will actually *prove* the contribution).

> **Doc debt:** `README.md`'s timeline still uses the older 10-phase numbering (Phase 9 = Evaluation) while `progress.md` and all real work use 9 = explainability, 10 = UI, 11 = evaluation. Align before the write-up.

---

## 6e. Untrained-plant handling: calibrated confidence, not hard refusal (2026-07)

Dr. Pandey's refinement of the scope gate. Diseases are visually similar across
species, so the classifier's *disease* knowledge transfers even to plants it was
never trained on. Rather than flatly refuse an untrained plant, we run the model,
show a **calibrated confidence**, and advise via symptom-RAG with a caution —
refusing only when the model is genuinely unsure.

- **Why calibration first.** A softmax classifier is over-confident, worse so on
  out-of-distribution inputs (exactly the untrained-plant case). Showing the raw
  softmax to a farmer would mislead. **Temperature scaling** (Guo et al. 2017) —
  one scalar `T`, `softmax(z/T)` — fixes the *number* without changing *which*
  class wins, so accuracy is untouched. `src/disease/calibration.py` fits `T` by
  a deterministic golden-section NLL minimisation (pure NumPy, dependency-free,
  unit-tested); `scripts/fit_disease_temperature.py` fits it once on the PlantDoc
  test set on Colab, reports ECE before/after, and suggests the advise floor from
  the confidence distribution of the model's *correct* predictions.
- **Routing.** `confidence >= CONFIDENCE_ADVISE_MIN` -> advise (with a caution if
  the plant is untrained); below it -> do NOT guess, collect the image to the
  feedback DB (`reason="low_confidence"`) and ask for more photos to retrain on.
- **Show the disease, not the wrong plant.** Labels are plant+disease coupled;
  `disease_type_from_class()` drops the crop token so an untrained plant is shown
  its *disease* ("Septoria leaf spot"), and the crop is soft context for RAG.
- **Honest wording.** "confidence", never "accuracy"; the untrained caution says
  the disease was recognised at X% and to confirm with an expert.

This replaces the scope gate's hard pre-inference refusal (§6d.2). It is the same
philosophy as symptom-based retrieval, applied to the vision side: the *disease*
generalises; the plant is context. Defaults `DISEASE_TEMPERATURE=1.0` (identity
until fitted) and `CONFIDENCE_ADVISE_MIN=0.50` (until the fit script sets it).
Thesis angle: cross-plant disease generalization with calibrated confidence + a
human-in-the-loop safety net. Tests: `test_calibration.py` (7) + scope-gate
additions; app + config + scope + feedback + calibration suites green.

---

## 6f. Phase 11 — first evaluation pass (PRELIMINARY, silver query set)

First real run of the harness on Colab (Llama-3.1-8B, 22 answerable + 2 negative
queries, book-level silver labels). **Preliminary until the expert gold-set.**

**Retrieval (book-level Precision@k etc.; Recall = n/a until passage labels):**
| variant | P@5 | nDCG@5 | MRR | Hit@5 |
|---|---|---|---|---|
| full (dense+BM25+rerank) | 0.736 | 0.942 | 0.909 | 1.00 |
| **keyword_only (baseline)** | **0.555** | **0.697** | **0.615** | 0.909 |
| dense_only | 0.800 | 0.942 | 0.943 | 1.00 |
| hybrid_no_rerank | 0.709 | 0.863 | 0.814 | 1.00 |

**Generation / grounding:** grounded-answer rate 13.6%, valid-citation rate
14.2%, **honest refusal 100%** (both negatives), **over-refusal 54.5%** (answerable
queries refused); ungrounded control 0% unfounded citations.

**RAGAS (gpt-4o-mini judge, local bge embeddings):** faithfulness **0.559**,
answer_relevancy **0.163**. Independent cross-check (direct gpt-4o-mini, refusals
excluded): faithfulness **0.86** (n=10), answer_relevancy 0.48.

**Findings.**
1. **The bridge is proven** — full beats the keyword-only baseline decisively
   (nDCG 0.94 vs 0.70, MRR 0.91 vs 0.62). Semantic retrieval matters because the
   corpus is classical translated text and queries are modern symptom descriptions.
2. **Ablation surprise:** `dense_only` (P@5 0.80) edges out `full` (0.74) — BM25
   injects lexically-matched but off-topic passages that the reranker only partly
   cleans up. Honest finding; reinforces "semantic > lexical for this corpus".
   Caveat: n=22 + book-level labels, so the dense-vs-full gap may be noise.
3. **When it answers it is faithful (0.56–0.86) and never fabricates** (100% honest
   refusal, 0% unfounded citations). The low relevancy (0.16) is dominated by the
   refusals (each is the same sentence, scores ~0), not by bad answers.
4. **The limiter is corpus coverage**, not retrieval or the bridge: 206
   tree-focused passages don't cover every symptom, so the generator honestly
   refuses ~55% of the answerable queries. → motivates the 2 AAHF texts.

**Infra notes for reproducibility:** T4 OOM if `load_all()` is used (loads unused
vision models) or the chroma embedder is left on GPU — build only
retriever+generator, embedder/reranker on CPU. RAGAS needs a pinned
ragas 0.1.21 + langchain 0.2.x stack (Colab's langchain 0.3 breaks unpinned ragas).

---

## 6g. Crop-agnostic disease-TYPE classifier — ISOLATED experiment (2026-08-13)

Research-only experiment (never overwrites the deployed C-PD model; new HF repos,
new `data/disease_type/`). **Hypothesis:** pooling the ~194 crop×disease labels
into ~13 crop-agnostic disease *types* (rust, blight, leaf_spot, …) is trainable,
aligns with the symptom-based IKS direction, and supports the untrained-plant
claim (§6e). Backbone = PlantVillage warm-start; EfficientNet-B4. Sources unified
by `scripts/build_disease_type_dataset.py` (PlantVillage + PlantDoc + Dr. Pandey's
Brazilian multi-crop set), deepest-disease-folder wins, content-hash dedup.

**Two runs — the leaf-crop trade-off (same shape as C-PD):**

| Run | TEST acc | macro F1 | Grad-CAM attention on FIELD images |
|---|---|---|---|
| uncropped | **0.782** | — | **background** on field/scene images (palm-blight bias) ❌ |
| **YOLO leaf-crop** | **0.719** | 0.635 | **on the leaf/lesions**, background cold ✅ |

- **The −6.3 accuracy is the honest tax; the leaf attention is what it buys.**
  Grad-CAM (6-image diverse panel) on the cropped model: corn-in-field *blight*
  and *leaf_spot* now heat the **lesions**, soil/background cold — the exact
  failure the crop targeted. Same call as C-PD (66.6% leaf-focused > 72.3%
  background): **keep the cropped version.**
- **Per-class (cropped):** healthy 0.82, rot 0.87, blight 0.79, powdery 0.77,
  rust 0.76, leaf_spot 0.73; weak = **downy_mildew 0.000** (8 samples — too small
  to be honest), **early_blight 0.364** (early↔late confusion), leaf_mold recall
  0.42 (12 samples).
- **The one Grad-CAM miss** (bacterial→leaf_spot) is a disease↔disease confusion
  on plant tissue, **not** background — a "good" error to report.
- **Caveat:** disease-type accuracy is a *coarser* label space than the 27-class
  C-PD, so 0.72 here is **not** directly comparable to C-PD's 66.6% — different
  task. It is a proof-of-concept for the crop-agnostic direction, not a drop-in.

**Next (paper-critical):** merge early+late→`blight`, drop/grow downy_mildew, then
a **held-out-crop test** (train excluding one crop, test on it) — the strongest
direct evidence for cross-plant generalization behind §6e.

---

## 6h. Corpus re-OCR + Kashyapiya ingest — coverage expansion (2026-08-31)

Motivated by the Phase 11 finding that **corpus coverage, not method, is the
limiter** (§6f: faithful but ~55% over-refusal). Two workstreams:

**(A) Re-OCR of the existing books with Gemini Flash 3.6 (no output-token cap).**
Ankit's hypothesis was that the original 3.5 OCR truncated dense pages. Confirmed:
the original run had a hard output cap that (i) truncated multi-verse pages and
(ii) once saved an API-error string as page text
(`brihat_samhita/page_0332.txt` = "An error occurred..."). Re-OCR fix = Flash 3.6,
**no `max_output_tokens`**, a completeness instruction, a tightened SKIP rule
(keep Introduction/preface/commentary prose; skip only blank/Devanagari-only/
index), and error-string detection (never cache a refusal). Ran **one book at a
time, verified old-vs-new, then swapped** (safety: separate `raw_reocr/` folder,
compared, copied into live `corpus/raw/`, temp deleted).
- **Vrikshayurveda:** 15 pages fixed (9 wrongly-blanked incl. the Introduction;
  6 truncated e.g. p71 622->3031, p72 464->3068 chars). 0 regressions.
- **Brihat Samhita:** the error page 332 fixed (56->1418 chars) + 18 truncated
  pages restored. 0 regressions. (scope=chapters, so the build still keeps only
  the 12 wanted chapters regardless of extra OCR'd pages.)
- Krishi Parashara + Upavanavinoda: **left as-is** (already Gemini-OCR'd, clean).
- Cost: Vrik ~₹7.7, Brihat ~₹29.9 (Flash 3.6). Script: `scripts/reocr_one_book.py`.

**(B) Kashyapiya Krishisukti ingested as the 5th book.** The 64-page Sanskrit-only
scan Dr. Pandey supplied was OCR'd + translated verse-by-verse (Gemini Flash 3.6,
857 verses, ~₹17; see `sanskrit_ocr_demo/`). Registered as `ready_external` with
the **English translation only** as `text_source`
(`corpus/ocr_external/kashyapiyakrishisukti.md`, 859 lines) so it matches the
other books' English-embedded format; Sanskrit kept in `sanskrit_ocr_demo/`.
**Honest caveat (verified against the AAHF scholarly edition + our own retrieval
test):** Kashyapiya is a **cultivation** text (soil/water/seed/sowing retrieve
0.62-0.69) — **light on disease-treatment** (disease queries ~0.58, one generic
pest tip only). So it strengthens soil/water/cultivation coverage but is not
expected to fix the disease-advice refusals; **Vishvavallabha** (Chakrapani
Mishra, ~1577; explicitly covers plant disease/pest management) is the higher-
value text to obtain next.

**Rebuild note (Windows):** the old `iks_corpus` was deleted + recreated empty in
a chromadb-only process (never with torch in-process — the DLL-crash rule), then
`build_corpus` re-embeds all 5 books fresh. Embedding forced to **CPU**
(`CUDA_VISIBLE_DEVICES=""`) — bge-large thrashes the 2 GB MX550 GPU (293 s/batch),
but CPU is ~0.4 s/chunk after a one-time ~116 s model load.

**Result (build done 2026-08-31):** corpus grew **206 -> 259 chunks (+53, +26%)**,
4 books -> 5. Per book: Brihat 136->140, Vrikshayurveda 42->**52** (+10, the
re-OCR recovery), Krishi Parashara 13 (same), Upavanavinoda 15 (same), Kashyapiya
**39** (new). Verified: live `iks_corpus` = 259 vectors (matches manifest).
Phase 11 re-run over-refusal: **72.7% (WORSE than 54.5%)** — see §6j for the
result and the diagnosis.

---

## 6i. NITI Aayog natural-farming manual ingested (2026-09-09)

Dr. Pandey's point that the corpus need not be classical books alone -- IKS-derived
practical knowledge counts too -- turned out to target our actual bottleneck. Phase 11
showed coverage, not retrieval, is the limiter (~55% over-refusal); the classical texts
are thin on symptom -> remedy.

**Source.** "Empowering Farmers: Natural Farming Training Toolkit and Best Practices
Guide", NITI Aayog, Feb 2026, ISBN 978-81-991080-0-4. 192p with a **clean text layer
-- no OCR, zero API spend**. Ch5 (Pest & Disease Management) and Ch6.1-6.2 (bio-inputs)
give structured **Purpose / Ingredients / Preparation / Application** entries for
Jeevamrit, Beejamrit, Neemastra, Brahmastra, Agniastra, Dashaparni -- exactly the
missing layer, and IKS-rooted (cow dung, urine, neem = the kunapajala tradition).

**Ingested** (`scripts/extract_niti_manual.py`): Ch2 seed (pp.36-43), Ch4 soil (60-65),
Ch5 pest+disease (66-77), Ch6.1-6.2 bio-inputs (78-92), Ch8 22 key crops (112-165).
**Skipped:** Ch1 theory, Ch3 water, Ch6.3-6.4 BRC infrastructure+schemes, Ch7
certification, Ch9 carbon credits, Ch10-11 frameworks -- none give plant-level advice.
64 sections, 23,819 words. Sections are emitted one per block so a formulation is not
split from its ingredients (verified: Neemastra intact in one 277-word chunk).

**Honesty tier.** Registered with `source_tier: modern_iks` -- a 2026 government
manual is not a classical treatise, so citations must say which tier they came from
and the "grounded in classical texts" claim stays true.

**Corpus: 259 -> 327 chunks** (NITI 68). Per book: brihat 140, niti **68**,
vrikshayurveda 52, kashyapiya 39, upavanavinoda 15, krishi_parashara 13.

**Deferred:** Upavanavinoda's English Introduction (PDF pp.9-42, Majumdar's essay,
contains translated verses) -- registered as `upavanavinoda_introduction`
(`source_tier: scholarly_commentary`) but the Gemini key hit free-tier quota
(429 RESOURCE_EXHAUSTED) mid-OCR, so it skips cleanly until its text_source exists.

**Infra fix.** `IKS_EMBED_DEVICE=cpu` override added to `embed.py`: bge-large (1.3GB)
thrashes the 2GB MX550 at **~454 s per 16-chunk batch**, and `CUDA_VISIBLE_DEVICES=""`
does NOT force CPU on Windows (PyTorch treats the empty string as unset -- the rebuild
log said "on cuda"). With the override the same batches ran ~34 s. Note the earlier
259-chunk build was also silently on GPU.

**Result:** the re-run made over-refusal **worse** (54.5% -> 72.7%). Full result and
root-cause analysis in **§6j** — the book covers insect pests, not the fungal leaf
diseases our query set asks about.

---

## 6j. Phase 11 re-run on 327 chunks — corpus expansion made it WORSE (2026-09-09)

The number §6h/§6i were waiting for. **Adding the NITI manual increased over-refusal
from 54.5% to 72.7%.** Honest negative result; kept in full.

**Generation / grounding (22 answerable + 2 negatives, Colab, same harness as §6f):**

| metric | 206 chunks (§6f) | 327 chunks | |
|---|---|---|---|
| over-refusal (answerable refused) | 54.5% | **72.7%** | WORSE by 18pp |
| grounded answer rate | 13.6% | 13.64% | unchanged |
| valid citation rate | 14.2% | **41.7%** | ~3x BETTER |
| honest refusal (negatives) | 100% | 100% | held |
| unfounded citations (ungrounded control) | 0% | 0% | held |

Raw counts: **10/22 answered -> 6/22 answered**; refusals 12 -> 16. The four lost
answers were *ungrounded* ones, so the system became **safer but less useful** — it
stopped guessing and refused instead. That also explains the citation-rate jump.

**Retrieval (same run):** full nDCG@5 0.94 -> **0.87**, P@5 0.74 -> **0.65**, MRR 0.91
-> 0.86; dense_only 0.94 -> 0.88; keyword_only collapsed 0.70 -> **0.49** (Hit@5 0.91
-> 0.64). **Hit@5 stayed 1.00 for full and dense_only.**

**Two things to separate here.**
1. *Measurement artifact.* The silver labels are **book-level** and were written when
   the corpus was 4 books, so the 107 new chunks (Kashyapiya 39 + NITI 68) appear in
   **no** query's `relevant_books` and score as noise even when useful. P@5/nDCG must
   fall mechanically. **Hit@5 = 1.00 proves nothing was lost.** The query set must be
   re-labelled before any future expansion is judged, or every addition will look like
   a regression.
2. *A real effect.* Over-refusal is measured on **answers**, not on book labels, so the
   artifact does not touch it. It genuinely got worse.

**Root cause (evidence).** Our 22 answerable queries are **100% fungal / bacterial /
viral leaf diseases** (scab, rust, blight, gray leaf spot, early/late blight, Septoria,
bacterial spot, leaf mould, mosaic, yellow virus). Word counts in the ingested NITI text:

| term | count |
|---|---|
| insect-pest terms (insect 44, borer 15, caterpillar 14, sucking pest 14, aphid 7, whitefly 6, jassid 5, mite 6, thrips 3, larvae 4, mealybug 1) | **119** |
| leaf spot / septoria / bacterial spot / leaf mould / gray leaf | **0 each** |
| scab / mildew / mosaic | 1 each |
| rust | 2 |
| blight | 4 |

So **NITI is an insect-pest manual, not a fungal-disease manual.** Neemastra treats
aphids, jassids and whiteflies — not apple scab or Septoria. (`rot: 66` in a first pass
was a false positive matching "rotation"; true standalone count is 17.)

**Mechanism.** NITI chunks are *semantically adjacent* ("pest and disease management",
"control", "spray") so they rank into the top-5, **displacing classical passages**
(P@5 0.74 -> 0.65). The generator then has too little usable classical evidence for the
§17 prompt's sufficiency test and refuses. Near-miss content is **actively harmful**,
not merely neutral.

**The generalisable finding (paper-worthy):** *expanding a RAG corpus with
topically-adjacent but non-matching material degrades performance by displacing
relevant passages — corpus growth must be matched to the query distribution, not to
volume.* This is the honest, evidence-backed answer to "what is the benefit of adding
a book": **not all books help; this one measurably hurt on disease queries.**

**Process failure to remember.** The book was recommended on the strength of "pest 197 /
disease 91" keyword counts and the well-structured Neemastra recipe, **without checking
overlap against the specific diseases in the query set**. Rule going forward: before
ingesting a source, measure its term overlap with the evaluation queries.

**Agreed plan (2026-09-14), in order:**
1. **Displacement check** — confirm NITI chunks actually occupy top-5 slots for the 22
   queries (local, no Colab, no LLM). Turns the hypothesis into evidence.
2. **Tier-aware retrieval** — route by problem type: fungal/bacterial disease -> classical
   tier; insect/pest -> NITI (`source_tier: modern_iks`). Keeps the book for what it is
   genuinely good at instead of reverting it.
3. **Extend the query set with insect-pest queries** — NITI is currently being tested on
   the one thing it cannot do; a fair evaluation needs queries in its scope. Re-label
   `relevant_books` at the same time (see artifact above).
4. Then consider extending the pipeline so the vision side can also *signal pest damage*,
   letting the combiner generate a pest-side query — a design change to evaluate only
   after step 1-3 show the routing works.

**Not done / deferred:** Upavanavinoda English Introduction (Gemini free-tier quota hit
mid-OCR); Vishvavallabha (the classical text that *does* cover plant disease) remains the
highest-value content fix.

---

## 6k. Displacement check — CONFIRMED, and it uncovered a much older corpus bug (2026-09-14)

Step 1 of the §6j plan. `scripts/check_displacement.py` retrieves the top-5 for each of
the 22 answerable queries **twice** — over the full 327-chunk corpus, then over
classical-only — and diffs them. Local, no Colab, no LLM. (Doc embeddings are cached to
`results/_docemb_*.npy`; the first run costs ~16 min on CPU, re-runs are seconds.)

**Result — the displacement hypothesis is CONFIRMED:**

| | |
|---|---|
| top-5 slots taken by NITI (modern tier) | **25 / 110 (22.7%)** |
| queries with >=1 NITI passage in top-5 | **17 / 22 (77%)** |
| classical passages pushed out | **28** |

Slot share: vrikshayurveda 56 (50.9%), **niti 25 (22.7%)**, brihat 18 (16.4%),
kashyapiya 6, upavanavinoda 4, krishi_parashara 1.

**But the per-query detail does NOT support the simple story.** Inspecting what was
actually displaced, most of it was *not* useful content:
- `vrikshayurveda 1.1` = **the publisher's address block** ("Chairman, Asian Agri-History
  Foundation, 47 ICRISAT Colony-I ... Secunderabad") — displaced in q04, q08, q15
- `brihat_samhita section_11` = "meteors are ... those who fall down after having enjoyed
  the fruits of their meritorious deeds" (astrology)
- `brihat_samhita section_1` = "Chapter XXVII—The Wind Circle [According to Utpala this
  chapter is **spurious**...]"
- `brihat_samhita section_7` = "[For an explanation of the Karaṇas see the author's
  *Fundamentals of Astrology* p.185...]"
- `vrikshayurveda 1.2` = editorial commentary about the text, not a remedy

Only **three** genuinely valuable displacements were found: `vrikshayurveda 207-222`
(remedy: sugar/sesame/milk for heat-dried trees, q04), `vrikshayurveda 173-189` (the key
symptom passage — insects at the roots, yellowing leaves, q06) and `vrikshayurveda
110-125` (watering guidance, q07).

So NITI is mostly **replacing junk with off-topic-but-clean text**. Displacement is real,
but it is *not* a sufficient explanation for the 18pp over-refusal jump — the generator
had roughly the same amount of usable classical evidence either way. Flagged as an open
question rather than papered over.

### The bigger finding: Brihat Samhita chapter spans over-capture (bug since Phase 3)

Chasing the junk led to a genuine defect in `src/rag/corpus/chapter_split.py`
(`locate_chapters`, lines ~161-166):

```python
if i + 1 < len(sorted_chapters):
    end_idx = sorted_chapters[i + 1][1]   # start of the next WANTED chapter
```

Each wanted chapter's span ends at the next **wanted** chapter, not the next **actual**
chapter. Every gap between wanted chapters is therefore silently ingested. With
`chapters: [21..29, 40, 54, 55]` that means **chapters 30-39 are swallowed into 29, and
41-53 into 40**.

Evidence in the data — chunks tagged `chapter: 40` ("Growth of Crops", a short chapter)
number **75 of Brihat's 140 chunks** and contain:
- "If at the time of the Sun's entry into Scorpio Jupiter be in Aquarius..." (astrology)
- "Sign Aries is considered to rule over cloths, sheep's wool..." (commodity divination)
- "The Gods with Indra as their leader ... went to the Milky Ocean" (mythology)
- house-dimension calculations in cubits for Brahmana dwellings (architecture)

Span check from the build log: chapter 29 -> pages 327-376 (50pp), **chapter 40 -> pages
377-543 (167pp)** for a chapter that is a few pages long.

**Scale:** roughly **~83 of 140 Brihat chunks (~25% of the whole 327-chunk corpus)** is
content we never intended to index. It has polluted retrieval **since Phase 3**, so the
original 206-chunk baseline in §6f is affected too — every evaluation number in this log
was measured against a partly-unintended corpus.

**Revised fix order (supersedes the §6j plan):**
1. **Fix `locate_chapters`** to end a span at the next *detected* chapter heading (or a
   page cap), not the next wanted one; rebuild; confirm Brihat drops from 140 to roughly
   40-55 chunks of genuinely wanted material.
2. **Drop front-matter / editorial-apparatus chunks** (publisher block, "[Cf. ...]",
   "see the author's ...", "this chapter is spurious") with a cleaning rule + test.
3. **Then** re-run Phase 11. Only after a clean corpus does tier-aware routing get
   evaluated — otherwise we would be tuning routing against polluted retrieval.
4. Tier-aware routing + pest queries (§6j steps 2-3) follow, evaluated as **A** (pest
   queries vs NITI alone) **and B** (full set re-run) — Ankit's refinement, so a bad book
   can be told apart from bad routing.

**Lesson for the write-up:** the displacement test was designed to confirm a hypothesis
and instead surfaced a deeper data-quality defect. Worth reporting as-is — corpus quality
was the confound underneath the coverage story.

---

## 6l. Chapter-span fix applied and rebuilt — corpus 327 -> 233 (2026-09-15)

Step 1 of the §6k plan, done. `locate_chapters` now ends a wanted chapter at the next
**detected chapter heading** rather than the next **wanted** chapter
(`find_chapter_starts`, tightened with `min()` so a span can only shrink — a
previously-correct span cannot break).

**Heading detection needed three guards**, each earned from a real false positive in this
OCR, and each verified against the actual pages:
1. require the literal word "chapter" — running headers are `Treatment of Trees LV 533`;
2. require a separator + capitalised title after the numeral — this rejects
   `[Cf. chapter IX]` and "as stated in chapter XXI". **Without it the first version of
   the fix silently truncated ch.23 Rainfall to 1 page and ch.24 Rohini to 10** — caught
   only by checking the gap pages against the printed book before rebuilding;
3. require the heading at the top of the page.
Verified: real headings at p.330 (XXX), p.381 (XLI), p.581 (LVI) detect; cross-references
at p.291, p.304, p.314 do not.

**Brihat spans now match the printed book** (96 pages ingested, was 319):

| ch | title | pages | len |
|---|---|---|---|
| 21-29 | rain / cloud / prognostics | 275-329 (contiguous) | 42 |
| 40 | Growth of Crops | 377-380 | **4** (was 167) |
| 54 | Exploration of Water Springs | 544-571 | 28 |
| 55 | Treatment of Trees | 572-580 | 9 |

**Rebuild result:** brihat **140 -> 46 chunks**; corpus **327 -> 233**. Others unchanged
(vrikshayurveda 52, niti 68, kashyapiya 39, upavanavinoda 15, krishi_parashara 13).
Build took **74 min** on CPU (`IKS_EMBED_DEVICE=cpu`); no OCR, no API spend — the raw
text was already cached, which is what made the fix free after the Gemini quota ran out.
Tests: 4 new regression tests; **414 passed** across the whole suite.

**Verification that the junk is gone.** Commodity divination ("Fluctuation of Prices",
"sheep's wool"), mythology ("Milky Ocean", "Indra's Banner") and "Signs of Swords /
Crowns / Pimples" are now **0 matches**. Remaining astrology terms (Jupiter, Scorpio,
meteor) and "cubits" are **legitimate**: ch.21/28/40 predict rain and crop yield from
planetary positions — *"If at the Sun's entry into Scorpio Jupiter be in Aquarius…"* IS
the text of "Growth of Crops" — and ch.54 measures well depth in cubits. A first pass
flagged these as junk; inspecting them in context showed the pattern, not the corpus,
was wrong.

### Correction to §6k

§6k said NITI was mostly "displacing junk", citing `vrikshayurveda 1.1` (the
publisher-address block) as an example. **That was partly wrong.** Chunk `1.1` begins
with the ICRISAT address but **ends with Table 1 — the disorder / cause / symptom /
remedy table** ("Broken trees should be smeared with a paste of the bark of plaksa and
udumbara mixed with ghee, honey, wine, and milk"). So one of the displaced passages was
genuinely useful, and the displacement harm in §6k was **understated**.

The other §6k examples (Brihat meteors, "Wind Circle … spurious", Karana notes) came from
the over-captured chapters and are **now removed by this fix**.

**Front matter still present, deliberately NOT deleted:** 4 vrikshayurveda chunks
(`1.1` x2, `306.1`, `section_2`) open with author affiliations / "About the Translator"
but continue into Table 1, Table 2 (materials + properties) and land-suitability
indicators. Deleting them would destroy real remedy content; the correct fix is to strip
the front-matter *prefix* during cleaning, which needs another 74-min rebuild and is not
worth blocking the re-baseline. Logged as open.

**Next:** push the 233 chunks to HF and re-run Phase 11. This will be the **first clean
baseline** — every earlier number in this log (including §6f's 206-chunk run) was
measured against a corpus that was ~25% content we never asked for.

---

## 6m. Phase 11 on the clean 233-chunk corpus — the first honest baseline (2026-09-15)

Re-run after the §6l chapter fix. This is the first Phase 11 measured on a corpus that
contains only what we asked for.

**Retrieval (22 answerable queries, book-level silver labels):**

| variant | P@5 | nDCG@5 | MRR | Hit@5 |
|---|---|---|---|---|
| full (hybrid + rerank) | 0.6727 | 0.8779 | 0.8561 | **1.0000** |
| dense_only | 0.6636 | 0.8818 | — | 1.0000 |
| hybrid_no_rerank | 0.5364 | 0.7250 | — | — |
| keyword_only | 0.3273 | 0.5151 | — | 0.6818 |

vs the 327-chunk run: P@5 0.65 -> 0.67, nDCG 0.87 -> 0.88, keyword_only nDCG 0.49 ->
0.52. Removing 94 chapters of astrology and commodity divination **did** clean up
retrieval, as predicted. The rerank is still carrying the pipeline (+0.14 P@5 over
hybrid_no_rerank), and keyword_only remains the weak leg — expected with Sanskrit
transliterations and a symptom-phrase query style.

**Generation / grounding:**

| metric | 206 (§6f) | 327 (§6j) | **233 (clean)** |
|---|---|---|---|
| grounded answer rate | 13.6% | 13.64% | **13.64%** |
| valid citation rate | 14.2% | 41.7% | **55.00%** |
| over-refusal | 54.5% | 72.7% | **81.82%** |
| honest refusal (negatives) | 100% | 100% | 100% |
| unfounded citations | 0% | 0% | 0% |

**The finding that reframes §6j.** Across three corpora of 206, 327 and 233 chunks —
different books, different sizes, a 94-chapter bug fixed in between — the grounded
answer rate is **frozen at exactly 13.64%, i.e. 3 of 22 queries, every single time.**
Nothing we changed moved it.

What *did* move is the weakly-grounded answers: **7 -> 3 -> 1**. Those were answers the
generator produced without solid support. As the corpus got cleaner, it stopped
producing them and refused instead. That single mechanism explains all three trends at
once: over-refusal rises (54.5 -> 72.7 -> 81.8) because refusals replace weak answers,
and valid-citation rate rises (14.2 -> 41.7 -> 55.0) because the surviving answers are
the well-supported ones. The system is behaving **more honestly at every step**.

**So the §6j root cause was incomplete.** NITI displacement is real (§6k measured it:
25/110 top-5 slots), and the chapter bug was real, but **neither was the cause of the
refusals** — fixing both left the grounded rate untouched. The remaining explanation is
the simplest one: **the classical texts may genuinely not contain remedies for most of
these 22 diseases.** The query set is *silver* — we asserted `expect_answerable: true`
for all 22; we never verified the corpus can answer them.

If that is right, "over-refusal" has been measuring the system against queries that have
no answer in the corpus, and **refusing is the correct behaviour** — the metric, not the
system, is wrong.

**Next: coverage check.** *(Done — see below; the result was ~4, the first branch.)* `scripts/check_coverage.py` retrieves the top-5
for each of the 22 queries locally (same hybrid+rerank pipeline, no LLM, no API cost) and
prints the passages with a remedy-language triage, so the question is settled from the
text itself rather than from metrics:
- ~3-5 queries with real treatment content -> the system is correct, the labels are
  wrong, and the story is **coverage** (-> Vishvavallabha, the classical text that does
  cover plant disease);
- ~12-15 -> the generator's sufficiency rule is too strict, and the fix is the §17
  prompt, not the corpus.

**Tier-aware routing (§6j step 2) is ON HOLD** until this resolves. On current evidence
it would be solving a problem that is not there: if the classical tier has no remedy for
these diseases, routing queries to it changes nothing.

**Also open:** the 233 chunks are not yet pushed to HF; the silver set still needs
re-labelling (new books score as noise, §6j); 4 vrikshayurveda front-matter prefixes
remain (§6l).

---

### Coverage check — RESOLVED: the corpus genuinely cannot answer 13 of the 22 queries

`scripts/check_coverage.py`, local, no API cost. **The cross-encoder's own top-1 score is
the measurement** — it is a relevance judge, trained to score a query/passage pair, and it
needs no heuristic on top of it.

| top-1 rerank score | queries | which |
|---|---|---|
| **>= 0.35** strong match | **4** | q22 rain signs (0.72), q20 boring insects (0.70), q21 soil preparation (0.63), q14 yellow/stunted leaves (0.58) |
| 0.15-0.35 marginal | 5 | q17 powdery mildew, q13 mosaic, q16 pepper spot, q02 apple rust, q04 corn blight |
| **< 0.15** no match | **13** | scab, Septoria, gray leaf spot, early/late blight, bacterial spot, leaf mould, black rot, corn rust, mites, weak tree |

**4 strong matches vs a grounded answer rate frozen at 3/22.** The generator has been
producing an answer almost exactly when the corpus actually contains one. The refusals are
correct.

**Why the mismatch is structural, not a retrieval failure.** The chunk retrieved rank-1 for
10 of the 22 queries is Vrikshayurveda **Table 1 — the disorder / cause / symptom / remedy
table**, which is real remedy content (`0a189afb`, `177254da`). But it is indexed **by
cause**: vata, pitta, kafa, fire, lightning, axe wound, ants, faulty seed — humoral
imbalance and physical injury. Our queries are indexed **by visual pathology**: "numerous
small dark spots with pale centres". Table 1 has no row for that, and no row for scab,
Septoria or mildew. The reranker returns it anyway because it is the most disease-like text
in the corpus, then scores it 0.06-0.22 — *"the best I have, and it does not match."*

So the classical texts classify plant disease by **aetiology in Ayurvedic terms**; the
PlantDoc label set classifies by **lesion appearance**. For 13 of 22 queries no mapping
exists, because the target concept is absent from the source tradition.

**Consequence for the metric.** `expect_answerable: true` on all 22 was an assumption we
never verified. Over-refusal of 81.8% is measured against 13 queries that have no answer;
the honest denominator is the 4-9 that do. The system's real behaviour is: **answers ~3 of
the 4 it can, refuses the rest, and fabricates nothing** (unfounded citations 0% in every
run). That is the desired behaviour of a grounded system, and the metric was hiding it.

**This also kills the §6j NITI verdict a second time, in the other direction.** NITI
`section_10` is rank-1 for q20 (0.70) and q14 (0.58) — the two highest-scoring disease-side
queries in the whole set. The book is not harmful; it is the *only* source that matches the
pest-damage and chlorosis queries. §6j's "corpus growth must match the query distribution"
finding stands, but the fair statement is that **NITI was tested almost entirely on
fungal-lesion queries that nothing in the corpus can answer.**

### Second defect found: citation labels do not identify passages

4 labels are shared by **17 chunks**: `vrikshayurveda v.1.1` x5, `v.1.2` x5, `v.1.3` x4,
plus one more. Five different passages all cite as `[Vrikshayurveda, ch. full, v. 1.2]`.
A reader cannot tell which was used, and the **valid-citation rate (55%) is scored against
labels that are not unique**. Separately, `chapter: "full"` and `verse_or_section: "1.2"`
for a 3,382-character chunk is not a verse reference at all — these come from the
whole-book ingest path, not the chapter-split path. Independent of the refusal question and
needs fixing before any citation number is reported in the thesis.

### Revised plan

1. **Re-label the silver query set** — mark the 13 unanswerable queries
   `expect_answerable: false` with the rerank evidence recorded, and report over-refusal on
   the answerable subset. This is the honest denominator, and it is a *finding*, not a
   patch: it quantifies the coverage gap between classical IKS aetiology and modern disease
   labels.
2. **Add pest/soil/season queries** (§6j step 3) — the four strong matches show what this
   corpus is genuinely good at. Test A (NITI alone) and B (full set) as agreed.
3. **Fix citation granularity** — real verse ranges instead of `ch.full v.1.2`.
4. **Tier-aware routing (§6j step 2): DROPPED for now.** Routing fungal-disease queries to
   the classical tier cannot help when the classical tier has no such content. Revisit only
   after a source that covers plant disease is ingested.
5. **Vishvavallabha becomes the critical path** — it is the classical text that treats
   plant disease directly. It is now the only route to raising the grounded rate.

**The reframed thesis contribution.** Not "we built a RAG system that answers disease
queries from Sanskrit texts" — the texts cannot answer most of them, and claiming otherwise
would require the system to fabricate. It is: *a grounded multimodal system that maps modern
vision-model disease labels onto classical IKS treatment knowledge, and that refuses rather
than fabricates where the traditions do not overlap — with the overlap measured, at 4-9 of
22 modern leaf-disease categories.* The refusal behaviour is the safety result, and the
coverage gap is a quantified finding about IKS digitisation, not a failure of the pipeline.

---

## 6n. Citation uniqueness + extending the query set into what the corpus covers (2026-09-26)

Two of the three open items from §6m. Vishvavallabha (the third) stays blocked — the book
has not been obtained. **No paid API anywhere in this work**: the OCR text is already
cached, embedding and reranking run on CPU, generation on Colab's free tier.

### Safety net first

The citation fix forces a corpus rebuild, so the change was made provably reversible
before any code was touched: git tag `thesis-safe-2026-09-26-pre-citationfix`, full copies
of `corpus/chunks/`, `corpus/vector_db/` and the query set, and a content fingerprint
(`corpus/_fingerprint_pre_citationfix.json`). Restore commands in `corpus/RESTORE.md`.

`scripts/verify_corpus.py` turns "did we lose anything?" into a checkable claim. The fix
renames labels and never edits text, so the **sorted set of 233 chunk texts must stay
byte-identical**; the script compares a SHA-256 of that set. PASS proves no content moved.
This is precisely the check that was missing when the chapter over-capture bug (§6l) went
unnoticed through two builds.

### Defect: a citation could not identify a passage

220 distinct labels for 233 chunks — **4 labels shared by 17 chunks**, all Vrikshayurveda
front matter:

| label | chunks |
|---|---|
| `vrikshayurveda ch.full v.1.1` | 5 |
| `vrikshayurveda ch.full v.1.2` | 5 |
| `vrikshayurveda ch.full v.1.3` | 4 |
| `vrikshayurveda ch.full v.1.4` | 3 |

So `[Vrikshayurveda, ch.full, v.1.2]` pointed at five different passages, and the
**valid-citation rate (55.0%, §6m) was being scored against labels that cannot be
checked.** The other 35 Vrikshayurveda chunks carry proper verse ranges (`v.1-17`,
`v.18-37`, …) and were never affected; no other book collides.

**Root cause.** The front matter contains numbered lists — the abbreviations list opens
*"1. Upavana. = Upavanavinoda. 2. ch. = Chapter no. 3. …"*. `_split_verses` reads each
`1.` as verse 1, and every oversized verse is then sub-split with `sub_marker =
f"{marker}.{sub_idx}"` where `sub_idx` **restarts for each oversized verse**. Five separate
"verse 1"s therefore each produced `1.1, 1.2, …`.

**Fix:** `uniquify_labels()` in `chunking.py`, applied per book in `build_corpus.py` at
both the external-OCR and Tesseract branches. Colliding labels gain a letter suffix in
document order (`1.2` → `1.2a`, `1.2b`, …) and `chunk_id` is recomputed, since the label is
part of its hash.

Deliberately a **post-hoc repair rather than a change to marker detection**: it only ever
touches labels that already collide, so a label that was correct cannot be broken. That is
the same narrowing discipline used for the chapter fix, and for the same reason — the first
version of that fix silently deleted good content (§6l).

7 regression tests: only-colliding-labels-change, all-labels-unique, text-never-changes,
chunk_id-recomputed-for-renamed-only, no-op-when-already-unique, same-label-in-different-
chapters-left-alone, >26-collisions. Suite **415 → 422 passing**.

### Extending the query set — and why the method matters

§6m established that 13 of the 22 queries have no answer in the corpus, because the texts
index disease by Ayurvedic cause and the queries index by lesion appearance. The set is
therefore testing the system almost exclusively on the one thing this corpus cannot do.

16 candidate queries drafted across the subject areas the treatises **do** cover: insect
and pest damage, seed treatment, soil suitability and enrichment, liquid manure, sowing
season, rain signs, irrigation, well-water divining, planting and transplanting, physical
injury, unproductiveness, grain storage.

**Method, stated because it is the part that can go wrong.** The candidates were written
from the **domain** — what a farmer would plausibly ask — **not** by reading the corpus and
picking passages. Choosing queries to match text we had already read would make the
evaluation circular: we would be measuring retrieval on passages selected for matching.
Labels are then assigned *from* measurement, never before it: `check_coverage.py` scores
each candidate with the pipeline's own cross-encoder and `merge_new_queries.py` writes the
label using the §6m thresholds (≥0.35 strong, 0.15–0.35 marginal, <0.15 none).

**Candidates scoring "none" are kept, not deleted.** Deleting them would bias the set
toward whatever the corpus happens to contain and inflate every number afterwards. A query
set built only from what the corpus can answer is not an evaluation.

`check_coverage.py` now accepts any query file and tags its outputs, so a candidate run
cannot overwrite the silver-set run.

### Results — Stage 1 and Stage 2 (26 Sep)

**Stage 1, citation uniqueness: done and verified.** Rebuild took 623 s, not the 74 min
feared — the OCR cache carried most of it. Corpus **233 chunks, 233 distinct citation
labels**, text byte-identical to the pre-fix fingerprint. Every passage is now uniquely
citable, so a citation number finally means something.

**A second defect surfaced during the rebuild, and it is worth recording.** The build log
read *"ChromaDB collection now holds 250 vectors"* for a 233-chunk corpus. `embed_chunks`
upserts by id, so renaming 17 chunks **added** their new ids and left the old ones
orphaned: every renamed passage was present twice, the second copy still carrying the
ambiguous label the fix existed to remove. Retrieval would have returned duplicates inside
a single top-5 and reintroduced un-checkable citations — silently undoing the fix.

`verify_corpus.py` had printed **PASS** while this was true, because it only compared the
chunk *files*. **Retrieval reads the vector store.** The verifier now compares both and
refuses to PASS unless they agree. This is the same class of gap that let the chapter
over-capture bug through two builds: a check that did not cover the thing that mattered.
`prune_stale_vectors.py` deletes an orphan only after confirming its text survives under a
new id, and refuses entirely if any does not. After pruning: 233 chunks, 233 labels, 233
vectors, PASS. Tests 422.

**Stage 2, the new queries — and a methodological correction.**

The coverage script scored **16 / 16 "strong"**, several above 0.95. That result is not
trustworthy, and reading the passages showed why:

| graded by reading | n | examples |
|---|---|---|
| **answers** — a retrieved passage directly answers | **8** | n05 seed treatment finds the Beejamrit recipe; n10 rain signs finds Brihat ch.28 on ants shifting eggs, snakes mating, chameleons gazing up; n11 watering finds Vrik v.110-125 with an exact schedule by soil and season; n14 broken branch finds Table 1, dress the spot with honey and ghee; n15 no flowers finds Upavanavinoda 177 |
| **partial** — right subject, top-1 off-topic or identification only | **7** | n13 "how to plant a sapling" scored **0.985** but returned text about *where* to cultivate vegetables; n02 "caterpillars chewing holes" scored **0.915** but top-1 was about raising plants from seed |
| **no** — nothing addresses it | **1** | n16 "protecting stored grain" scored **0.625** and returned a materials table about roots and branches |

**The finding: the cross-encoder score is sound evidence of ABSENCE but not of PRESENCE.**
It measures vocabulary overlap. A score of 0.06 (§6m) genuinely means nothing in the corpus
matches. A score of 0.98 means the passage shares the query's vocabulary, which is not the
same as containing an answer. §6m used it in the safe direction; extending it to confirm
coverage would have inflated this result by roughly 2x.

Score and reading disagree badly in the upper range: n07 scores **0.462** and answers
fully, while n13 scores **0.985** and does not. **Labels are therefore set from reading,**
and the score-only tier is retained alongside for comparison.

One sharper illustration, taken from the two query sets:

| query | score |
|---|---|
| q19 "a tree that is weak and poorly nourished with pale drooping foliage in exhausted soil" | **0.056** |
| n15 "a tree that does not flower or bear fruit even though it looks healthy" | **0.935** |

Nearly the same question about an unproductive tree; a 17x difference in score, because
n15's wording tracks Upavanavinoda 177 almost verbatim. **Phrasing, not content
availability, dominates the score** — which makes Stage 0 (does Llama phrase well?) the
decisive open question rather than a side check.

**Honesty note on method.** These candidates were drafted from the domain rather than from
the corpus, to avoid circularity. That was only partly achieved: passages from
Vrikshayurveda Table 1, v.173-189, NITI section_10 and Kashyapiya section_26 had been read
earlier in the same working session, so some corpus vocabulary very likely leaked into the
phrasing. The 16/16 score is partly an artefact of that. The reading-based grades are the
defensible numbers; they are the author's own and await expert ratification like the rest of
this silver set.

**Query set now:** 24 to **40 queries**, answerable 9 to **24**; 17 disease-label queries
and 23 domain queries. Reported separately from here on, because they measure different
things — the disease queries exercise the whole deployed pipeline (vision to bridge to
retrieval), the domain queries exercise only corpus and retrieval.

**What the two sets say together.** Disease-label queries: 1 of 17 reaches a genuine match.
Domain queries: 8 of 16 are directly answered, 15 of 16 at least on-topic. The books are not
the limitation — **the question vocabulary is.** The corpus was being examined on a syllabus
it was never taught.

**Still open:** Stage 0 (capture and score Llama's real queries — needs Colab), HF push,
Phase 11 A/B.

---


## 6o. Stage 0 — the bridge measured, not assumed. It is the bottleneck. (2026-09-26)

Every retrieval number in this project had been measured on **hand-written** queries. The
deployed system never sends those: Strategy B (Llama-3.1-8B) writes the query from the
vision label. Nobody had checked that the two match, so the entire retrieval evaluation
rested on an untested assumption. Ankit's call was to test both before fixing anything.

`scripts/capture_llama_queries.py` ran Strategy B over the 17 disease labels on Colab
(vision models not run — the label is exactly what the vision model hands over; neutral soil
held constant so any difference comes from the label alone; temperature 0.2, seed 42). The
generated queries were then scored by the same cross-encoder as the hand-written ones.

### Result: the hand-written queries win 14–0

| | mean top-1 | strong | marginal | none |
|---|---|---|---|---|
| hand-written | **0.1691** | 1 | 5 | 11 |
| **Llama (deployed bridge)** | **0.0445** | 1 | 0 | **16** |

Per query: Llama better on **0**, hand-written better on **14**, tied on 3 (tie = within
0.05). Llama's mean is **3.8x lower**. Sixteen of 17 score below 0.15.

**So the published numbers overstate the deployed system.** nDCG@5 0.878, P@5 0.673 and
Hit@5 1.00 (§6m) were all measured on queries the system does not produce. The real disease-
side performance is materially worse, and §6m's headline — "the corpus cannot answer these" —
was only half the story. There are **two independent problems**, not one:

1. **corpus coverage** — real, established in §6m, affects 13 queries;
2. **bridge quality** — newly measured here, affects all 17.

### Why Llama's queries fail — three faults, in every single one

The prompt is explicit ("LEAD WITH THE SYMPTOM"; crop as "light background context at
most"). Llama violates it consistently:

> *"**In loam soil with moderate moisture**, where apple plants exhibit dark, rough, corky
> lesions on their leaves, **what are the observable symptoms and underlying causes described
> in the classical Sanskrit treatises?**"*

against the hand-written equivalent:

> *"dark rough corky lesions and scabby patches spreading over the leaves of a tree"*

1. **It leads with soil.** All 17 open with "In loam soil with moderate moisture" — words
   that match nothing in the corpus and dilute the symptom that does.
2. **It asks a question *about the books*.** "…described in the classical Sanskrit
   treatises?" The corpus contains remedies, not discussion of treatises, so that clause is
   pure noise. The corpus is written in **statements**; a query phrased as a question about
   sources cannot match it.
3. **It keeps modern pathology vocabulary** — "necrotic patches", "plant vigor", "lesions" —
   the very terms the bridge exists to remove.

The one thing it does right is the core translation: *Apple Scab → "dark, rough, corky
lesions"*. Rule 3's symptom-family mapping is working. The failure is everything Llama wraps
around it.

**This is prompt non-compliance, not a model limitation** — which is the good news, because
it is fixable by rewriting the prompt rather than by retraining anything.

### Consequences

- **Strategy B's headline claim needs restating.** "0.59–0.96 vs 0.01–0.04" (Strategy B vs
  the template baseline) came from a small qualitative Phase 8 comparison. Measured properly
  across 17 queries, the deployed bridge averages **0.0445**. Strategy B still beats a bare
  template, but the gap is far smaller than reported, and the honest framing is that **the
  bridge is necessary but currently under-performing its own design**.
- **The 11 provisional `no_coverage` labels stand.** They were flagged for re-measurement in
  case Llama phrased better; it phrased worse. The hand-written wording is the *generous*
  ceiling, so those labels are if anything conservative. No re-labelling needed.
- **The evaluation must switch to Llama's queries** for the disease half. Hand-written
  queries remain useful as a *ceiling* — "what a well-worded query could have achieved" —
  but they are a diagnostic, not a measurement of the system.
- **Reporting both is the contribution.** The gap between the two (0.169 vs 0.045) is a
  quantified measure of how much the query-generation step costs, which is exactly the kind
  of number a multimodal-bridge paper should carry and almost never does.

### Stage 3 — the prompt fix, cheapest first

1. **Forbid the soil-first opening and the meta-question.** Require the query to be a
   *statement describing what is seen*, never a question, and never a reference to the texts
   or treatises. Soil only if a soil cause is supplied, and never in first position.
2. **Few-shot examples drawn from the corpus** — show Llama four real lines
   (*"drying, yellowness, and excessive paleness of leaves"*, *"the trees ooze out even
   without wounds"*) plus one good query, so it copies the register instead of inferring it.
3. **A corpus word list** in the prompt: prefer *yellowness, paleness, drying, withering,
   oozing, scorched, eaten away, spots*; avoid *necrotic, vigor, lesion, pathogen*.
4. Only if 1–3 are insufficient: **two-round retrieval** — rough query, show Llama the real
   passages retrieved, let it rewrite in their vocabulary, retrieve again.
5. **No fine-tuning.** It needs data we do not have and would lock the bridge to today's
   corpus.

Re-run `capture_llama_queries.py` after each step and compare — the target is to close the
0.1246 mean gap to the hand-written ceiling.

---

### Stage 3 round 1 — the prompt rewrite, measured

Same 17 labels, same corpus, same scorer. The prompt rewrite alone:

| | before | after | hand-written ceiling |
|---|---|---|---|
| mean top-1 | 0.0445 | **0.0977** | 0.1691 |
| strong | 1 | **2** | 1 |
| marginal | 0 | **1** | 5 |
| none | 16 | **14** | 11 |
| per-query wins vs hand-written | 0 | **1** | — |
| ties (within 0.05) | 3 | **5** | — |

**The bridge more than doubled (2.2x) and closed 43% of the gap to the hand-written
ceiling**, with no retraining and no cost. Compliance with the prompt went from near-zero
to near-total:

| fault | before | after |
|---|---|---|
| opens with a soil/framing clause | 17/17 | **0** |
| phrased as a question | 14/17 | **0** |
| mentions the treatises | 13/17 | **0** |
| modern pathology vocabulary | 7/17 | 5/17 |

Queries now read like the corpus: *"dark rough corky patches spreading over the leaves of
an apple tree"*, *"Leaves with small, circular, dark spots with pale centres on a corn
plant"*.

**One fault barely moved, and it was a contradiction I introduced.** Rule 3 hands Llama a
ready-made phrase per disease family, and one of them read *"blight -> spreading brown
NECROTIC patches"* while the new WORD CHOICE section bans "necrotic". The prompt told Llama
to use a word it also told it to avoid, and Llama followed the more specific instruction.
Rule 3's phrases are now written in the corpus's own vocabulary, and a test asserts rule 3
can never contain a word WORD CHOICE forbids, so the two sections cannot drift apart again.
The measurement above does **not** include that fix.

**What the remaining failures are made of.** The two queries that now succeed are both
*virus* cases — yellowing and mottling — which is precisely the vocabulary the corpus uses
(*"yellowness", "excessive paleness"*). Everything still failing is a fungal spot, blight or
scab: the categories §6m established the texts do not contain. So once the bridge stopped
adding noise, **the residue is largely genuine coverage gap rather than phrasing** — which
is the separation this whole exercise was for.

**Honest caveat on the ceiling.** The hand-written queries were drafted after reading corpus
passages in the same session, so part of their 0.1691 is vocabulary leakage rather than
craft. The true gap between the bridge and a fair human baseline is therefore smaller than
0.0714, and 0.1691 should be treated as an optimistic ceiling, not a target.

**Next:** re-run after the rule-3 fix, then decide whether two-round retrieval (show Llama
the retrieved passages, let it rewrite in their words) is still needed.

---

### Stage 3 round 2 — the bridge now matches the hand-written ceiling

Rule 3's phrases rewritten into the corpus's vocabulary. Same 17 labels, same corpus, same
scorer.

| | baseline (§6o) | round 1 | **round 2** | hand-written ceiling |
|---|---|---|---|---|
| mean top-1 | 0.0445 | 0.0977 | **0.1368** | 0.1691 |
| strong | 1 | 2 | **2** | 1 |
| marginal | 0 | 1 | **4** | 5 |
| none | 16 | 14 | **11** | 11 |
| wins vs hand-written | 0 | 1 | **4** | — |
| gap to ceiling | −0.1246 | −0.0714 | **−0.0323** | — |

**74% of the gap closed, and the mean difference is now inside the 0.05 noise floor** — the
script's verdict flipped to *"no meaningful difference"*. Llama's "none" count (11) now
equals the hand-written count exactly, and it beats the hand-written wording outright on
4 queries, including mosaic virus at **0.7634** against 0.2706.

Compliance is total: 0/17 open with soil, 0/17 are questions, 0/17 mention the treatises,
0/17 use banned vocabulary.

**Read the verdict carefully.** The *mean* is within noise, but per query the hand-written
set still wins 10 to 4. So the honest statement is that the bridge has caught up on
average, not that it is equal everywhere. And the ceiling itself is soft: those hand-written
queries were drafted after reading corpus passages in the same session, so part of 0.1691
is vocabulary leakage. Measured against a fair human baseline the remaining gap would be
smaller still.

**Conclusion: two-round retrieval is not needed.** Step 4 of the Stage 3 plan is dropped.
Prompt engineering alone took the bridge from 3.8x worse than a human to statistically
level, at zero cost and with no retraining.

**A behaviour change worth recording.** Forcing corpus vocabulary made the queries less
discriminating. Septoria leaf spot, bacterial spot and bell-pepper leaf spot now generate
*the same query* apart from the crop name — "small dark spots with pale or ringed centres" —
and late blight collapsed to "leaves drying and withering". Three diseases producing one
query means the system cannot return three different answers.

Whether that is a defect is genuinely arguable, and the honest answer is that it is mostly
not: the treatises do not distinguish those conditions either. They prescribe by observed
symptom, and all three really do present as dark spots with pale centres. The bridge has
stopped asserting distinctions the target tradition does not make. It should be reported,
not hidden — it is a direct consequence of mapping a fine-grained modern taxonomy onto a
coarse-grained classical one, which is the same boundary §6m measured from the other side.

Minor artefact: one query says "a corn tree", from the tree-centric worked examples.

**What the residue is.** The 11 remaining failures are scab, rust, gray leaf spot, Septoria,
bacterial spot, leaf mould and black rot — exactly the set §6m established the texts do not
contain. The two successes are both virus cases, whose yellowing-and-mottling language the
corpus does use. With the bridge no longer adding noise, **what is left is the coverage
boundary, not a query-wording problem.** The two failures are now cleanly separated and can
be reported independently.

---



## 6p. Phase 11 on the deployed system — and two measurement bugs it exposed (2026-09-26)

The first end-to-end run of the system as it actually behaves: `QUERY_SOURCE="generated"`,
so the 17 disease queries carry the wording Strategy B really produces rather than a
hand-written stand-in. Corpus 233 chunks with unique citation labels, pushed to HF before
the run. Query set 40 (24 answerable, 16 unanswerable).

Two defects surfaced, both in the *measurement* rather than the system. Both are recorded
in full because each made a headline number look better or worse than the truth.

### Bug 1 — the answer key was copied from the answers (retrieval metrics)

`merge_new_queries.py` set each new query's `relevant_books` to
`sorted({p["book"] for p in row["top5"][:3]})` — the books retrieval had just returned.
Precision@5, nDCG and MRR then graded retrieval against its own output.

Visible symptom: `keyword_only` P@5 rose **0.33 → 0.53** purely from adding queries, which
no change to a retriever can explain. The first retrieval table of this run
(full P@5 **0.7083**, nDCG 0.888) was inflated throughout.

The original 24 queries were never affected — their labels were authored before any
retrieval ran. The 16 domain queries were re-labelled from **each treatise's subject
matter**, the same basis, with the reason recorded per query in `relevant_books_basis`.
13 of 16 changed, and several now *disagree* with what retrieval returned — the sowing
calendar goes to Krishi Parashara rather than NITI's banana schedules, watering to
Vrikshayurveda rather than Brihat Samhita, kunapajala to Vrikshayurveda rather than
Kashyapiya. Those disagreements are the evidence the labels are no longer derived from the
ranking.

Generation metrics were never affected: they use `expect_answerable` and citation
resolution, never `relevant_books`.

### Bug 2 — the prompt collided with itself (citation rate)

Measured valid-citation rate: **20.74%**, against 55.00% in §6m. That reads as a collapse in
grounding. It is not.

The context block headed each passage `[Source 2] Vrikshayurveda, ch.full, v.1.2d` while
rule 2 asked for `[Source Text, ch.<chapter>, v.<verse>]`. The word "Source" appears in
both, so the model merged them and cited `[Source 2, ch.full, v.1.2d]` — **chapter and verse
correct, the book's name replaced by its position in the retrieved list.** Such a citation
resolves to nothing even though the right passage was used.

Per-query output settles it. q08 cited `v.1.2d`, a genuinely retrieved chunk, and scored
invalid purely on the name. Every one of the 6 failures used the `Source N` form. The single
answer that did resolve cited `[Kashyapiya Krishisukti, ch.pages_1_64, v.section_3]` — by
name.

Each passage header is now the citation itself, character for character, so the model copies
rather than assembles; the translator credit sits outside the brackets so it cannot leak
into the verse field; and rule 2 states that `"Source 2"` is not a citation. Four regression
tests, including one asserting every header parses as a valid citation — a perfectly
obedient model must not be able to produce an unresolvable one.

**Note also that §6m's 55% was itself inflated**, for the opposite reason: labels were then
duplicated, so `v.1.2` matched any of five passages and a vague citation got credit. Neither
20.74% nor 55% is the real figure; the re-run after this fix is the first trustworthy one.

### Retrieval, with honest labels

| variant | P@5 | nDCG@5 | MRR | Hit@5 |
|---|---|---|---|---|
| full (hybrid + rerank) | 0.6417 | 0.8423 | 0.8472 | 0.9167 |
| **dense only** | 0.6250 | **0.8875** | **0.8764** | **1.0000** |
| hybrid, no rerank | 0.6250 | 0.8115 | 0.7722 | 0.9583 |
| keyword only (baseline) | 0.5667 | 0.7834 | 0.7562 | 0.9167 |

**Finding A — BM25 is a net negative, and this is the third time it has shown.**
`dense_only` beats the full hybrid on nDCG, MRR and Hit@5, and finds a relevant book for
**every** query (Hit@5 1.00) where the full hybrid misses two. §6f saw this at n=22 and it
was set aside as possible noise; it has now held across three runs, a larger query set and
independently authored labels. Adding BM25 to dense retrieval pushes relevant passages out
of the top 5. The "hybrid retrieval" component should be re-examined — dense-only, or a
reweighted fusion — rather than defended.

**Finding B — the keyword baseline has nearly caught up, and that sharpens the
contribution claim.** In §6m the gap was nDCG 0.878 vs 0.515; here it is 0.842 vs 0.783.
The cause is the query set: domain queries such as *"the proper season for sowing"* already
use the corpus's own vocabulary, so BM25 handles them well. Disease labels such as *"Apple
Scab"* must be translated first, and that is where semantic retrieval earns its place.

So **the bridge's advantage is specific to modern diagnostic labels, not to retrieval in
general.** §6f's headline "nDCG 0.94 vs 0.70" was measured only on disease queries and
therefore overstated the general case. Retrieval must be reported split by query type from
here on, never pooled.

### Generation — first run, and why it must be repeated

| metric | value |
|---|---|
| answerable queries | 24 |
| grounded answer rate | 16.67% |
| valid citation rate | 20.74% (*Bug 2 — not a real figure*) |
| honest refusal | 93.75% (15/16) |
| over-refusal | 83.33% (20/24) |
| unfounded citations | **0%** |

**Unfounded citations remain 0%.** With no corpus at all, the model still invents no
citations. That guarantee has now held across every run and every corpus version.

Grounded answer rate 13.64% (§6m) → 16.67%: 4 of 24 rather than 3 of 22. Still low, and it
will move once Bug 2 is fixed, because an answer only counts as grounded when at least one
citation resolves.

**Honest refusal fell below 100% for the first time** — q15 (*"fine pale stippling and
bronzing of leaves with tiny mites and webbing beneath"*) was answered rather than refused.
Reading the answer, it opens *"is not directly addressed. However, the texts prescribe
treatments for similar symptoms"* and cites nothing. So it is not fabrication — the refusal
detector simply does not recognise a hedge that declines in substance while continuing to
talk. That is a **detector** gap, not a safety failure, but it means the honest-refusal
metric is softer than it looks and the refusal classifier needs its own test.

**Deferred:** re-run cells 2 and 3 after the citation fix; split retrieval reporting by
query type; investigate dense-only against the hybrid; harden the refusal detector against
hedged non-answers.

---

## 7. Negative Results (paper ammunition — keep these honest)

A thesis is stronger for documenting what *didn't* work and why.

### 7.0b Brihat Samhita chapter over-capture (found 2026-09-14) — CORPUS BUG, see §6k
- **Defect:** `locate_chapters` ends each wanted chapter at the next *wanted* chapter, so chapters 30-39 and 41-53 were silently ingested (astrology, commodity divination, mythology, architecture).
- **Scale:** ~83 of 140 Brihat chunks (~25% of the 327-chunk corpus). Present since Phase 3 — every evaluation number in this log was measured against a partly-unintended corpus.
- **Verdict:** fix before any further retrieval tuning; re-baseline afterwards.

### 7.0 Corpus expansion with near-miss content (2026-09-09) — FAILED, see §6j
- **Idea:** add the NITI Aayog natural-farming manual (68 chunks) to close the symptom->remedy gap behind the ~55% over-refusal.
- **Result:** over-refusal **54.5% -> 72.7%**; answered 10/22 -> 6/22. Valid-citation rate improved 14.2% -> 41.7% (it stopped guessing).
- **Why:** the manual is insect-pest content (119 pest terms) while every evaluation query is a fungal/bacterial/viral leaf disease (leaf spot/septoria/bacterial spot/leaf mould = 0 mentions). Its chunks are semantically adjacent, so they displace classical passages in the top-5.
- **Verdict:** near-miss corpus material is actively harmful. Match corpus growth to the query distribution. Fix = tier-aware routing (§6j plan), not reverting the book.

### 7.1 Background randomization (Phase 5-R) — FAILED as a fix
- **Idea:** segment the leaf, composite onto random backgrounds each epoch so background can't be a label cue. Added a `no_leaf` reject class.
- **Result:** central-attention rate 1.2% → 5.9% (marginal), but accuracy DROPPED everywhere: PlantVillage 99.8→90.7, PlantDoc 72.3→66.8.
- **Verdict:** net negative. Background randomization cost accuracy for almost no attention gain. **Consistent with the literature.** Keep as a documented ablation; do not build on the R model.

### 7.2 Soil V3-sequential transfer — FAILED (catastrophic collapse)
- **Idea:** warm the B0 backbone on soil_type, then moisture, then full multi-task — hoping a "soil-aware" backbone lifts the weak texture head.
- **Result:** all three heads **collapsed to ~20% val accuracy** on Colab. The multi-stage / curriculum / per-stage head-freezing pattern destabilized training.
- **Verdict:** abandoned. Documented negative result. Led directly to the safer V3-tiling design (reuse V2 recipe unchanged, only change data).

### 7.3 Soil V3-tiling — texture head did not clearly improve
- **Idea:** tile texture source images into a 3×3 patch grid (2,511 leakage-safe patches) to give the weak texture head more training signal, reusing the V2 recipe byte-for-byte.
- **Ship criterion:** texture top-1 up AND soil_type/moisture not down >2pp vs V2.
- **Result:** _V2 remained production (texture stayed ~67.9%); V3-tiling preserved as an ablation. Confirm the exact Colab number and record here._
- **Takeaway:** texture (67.9%) is a genuinely hard head — a real open problem, candidate for future work, not a quick win.

### 7.4 Disease LP-FT (linear probe, frozen Paddy backbone) — FAILED
- **Idea:** freeze the healthy Paddy backbone, train only a fresh 27-class PlantDoc head (preserve good features per Kumar et al.).
- **Result (2026-06-12):** 61% vs OLD 72% — an 11pp DROP. No Grad-CAM improvement.
- **Why:** (a) Paddy backbone is rice-specialized — wrong features for PlantDoc's multi-crop diversity; (b) a frozen backbone + linear head can't change attention.
- **Verdict:** negative result. Confirms (with the cross-test) that **feature-preservation tricks don't fix clutter.** → Step 2.

### 7.5 Disease detect-then-crop (inference-time) — FAILED
- **Idea:** crop the leaf (YOLO / GT box) before classifying, no retraining.
- **Result (2026-06-12):** held-out no-crop 72.3% → YOLO-crop **58.2%** (−14pp). (A leaky detection-set run showed a fake 91.9% — discarded.)
- **Why:** the model depends on background; cropping it away removes the crutch and exposes its true lower leaf-only ability. Inference cropping can't fix a background-trained model.
- **Verdict:** third failed fix. Cropping only helps if you RETRAIN on crops (~70%, same accuracy, honest attention). Confirms 72% is the PlantDoc ceiling for this model.

---

## 8. Decision Log / Open Questions

- **Which disease model do we build on?** → The **OLD** cascade (healthier than R at every stage). R is a negative-result ablation only. (Decided 2026-06-12.)
- **Disease fix order:** LP-FT first (cheap, best-fit), then detect-then-crop, then (only for the *attention* story, not accuracy) segmentation masking. (Decided 2026-06-12.)
- **Open:** does LP-FT actually recover accuracy + leaf attention? (running)
- **Open:** texture head (67.9%) is the soil weak point — revisit if time permits.
- **Open:** Grad-CAM target layer — standardize on `blocks[-2]` for reporting (better localization than `conv_head`).

---

## 8b. Decision Rationale — WHY we chose each path (the reasoning, not just the what)

This section captures the reasoning chain from our working sessions that `progress.md` never recorded. For each fork, *why* we went the way we did — so it can be defended in a seminar.

**Why we audited the disease model stage-by-stage instead of just retraining.**
The demo showed the disease model attending to corners. The naive reaction is "retrain it." But we'd *already* retrained once (Phase 5-R background randomization) and it failed. So before spending another week, we forced a **step-wise diagnosis**: test each cascade stage on its own data with accuracy + Grad-CAM. The logic: if we don't know *which* stage breaks, any fix is a guess. This discipline paid off — it localized the failure to stage 3 (PlantDoc) precisely.

**Why we suspected the Grad-CAM code before blaming the model.**
The corner attention looked identical across the OLD and R models and across every image. An artifact that doesn't vary with model or input is usually the *visualization*, not the model. So we tested 3 target layers. We were partly right (`conv_head` exaggerates corners) and partly wrong (the bias is real — it persists at `blocks[-2]`). Worth doing: it stopped us from over-claiming and corrected the exact statistic we report.

**Why we tested PlantVillage even though it "already had 99%".**
The user's instinct: prove the foundation before chasing the hard dataset. If the model can't even attend to leaves on *clean* images, there's no point fixing PlantDoc. Result: it CAN (99.8%, leaf attention). That ruled out "the architecture is fundamentally broken" and pointed the finger at fine-tuning. Critical: 99% on PlantVillage is real but *in-distribution* — high accuracy there doesn't prove generalization, only the Grad-CAM (leaf vs background) tells us if it's earned. It was.

**Why we rejected our own R model.**
Loyalty to prior work is a trap. R cost accuracy at *every* stage (PlantVillage 99.8→90.7, PlantDoc 72.3→66.8) for a marginal attention gain. The honest call: demote R to a negative-result ablation, build on the healthier OLD cascade. The literature review confirmed background randomization commonly hurts — our failure wasn't a bug, it was expected.

**Why we did a literature review BEFORE picking the next fix.**
The user explicitly asked: "research deeply what others did, rather than randomly doing anything." We were about to commit a week to segmentation masking. The review found masking fixes *attention but not accuracy* (capture bias, mask noise) — i.e. it would have burned a week for the wrong metric. Same review surfaced LP-FT (freeze backbone) with 11–15pp field evidence, matching our diagnosis (full fine-tuning distorts good features). This single step reordered our priorities and saved the week.

**Why LP-FT first, not detect-then-crop (which has bigger historical numbers).**
Ranked by (probability of working) × (low effort). LP-FT is a one-day experiment that directly matches the diagnosis (preserve the stage-1/2 features full FT destroyed) and needs no new component. Detect-then-crop has the strongest historical PlantDoc number but requires adding a leaf detector/cropper — more effort. So: cheapest-best-fit first, escalate only if it fails. We kept a clear decision rule so we don't move the goalposts.

**Why we keep failures in the record.**
A thesis/paper is stronger for an honest ablation table. R's failure and (if it happens) masking's flatness are contributions: they tell the next researcher what *not* to do, with evidence.

---

## 9. For the Progress Seminar (what to say)

1. **Built a complete IKS-grounded multimodal advisory system** — disease + soil vision models feeding a RAG pipeline over classical Sanskrit agricultural texts, with a live UI. End-to-end working.
2. **Strategy B (LLM-mediated query rewrite) is the core integration contribution** — bridges modern vision labels to classical-text vocabulary; 0.59–0.96 vs 0.01–0.04 retrieval.
3. **Rigorous interpretability finding on the disease model** — via step-wise Grad-CAM I localized a background-shortcut bias to the PlantDoc fine-tuning stage (stages 1–2 healthy at 99.8%/97.0%, stage 3 drops to 72.3% with off-leaf attention). Confirmed it's a real model property, not a visualization artifact.
4. **Honest negative result** — background randomization (a hypothesized fix) failed, consistent with the literature.
5. **Principled fix in progress** — LP-FT (freeze the healthy backbone, retrain the head), chosen from a literature review, with a built-in accuracy+attention test.
6. **Our 72.3% on PlantDoc is at the published frontier (~74–78%)** — the contribution is the diagnosis + fix + the grounded-advisory system, not chasing SOTA accuracy.

---

## 10. How to maintain this doc

After every experiment, append to the relevant section with: **date, what, hypothesis, method, numbers, verdict.** Update the *Current State* snapshot (§1) and *Component Results* tables (§3) when a number changes. Add failures to §7 — they are as valuable as successes. Mirror the day's work into `research_journal/daily/YYYY-MM-DD.md`.
