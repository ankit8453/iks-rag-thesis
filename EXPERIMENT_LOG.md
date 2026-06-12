# Master Experiment Log — IKS-Grounded Multimodal Agricultural Advisory System

**M.Tech Thesis · IIITDM Jabalpur · Author: Ankit Pawar · Supervisor: Dr. Akshay Pandey**

> **What this document is.** The single results-first ledger for the whole project. Every experiment we run goes here with: what we tried, the hypothesis, the method, the numbers we got, and a clear **PASS / FAIL / verdict**. Read the *Current State* section first to know where we stand; read the *Component Results* tables for the headline numbers; read the *Disease-model diagnosis* section for the deepest piece of analysis (it is the most paper-critical).
>
> **Companion docs.** `progress.md` = narrative weekly log (engineering detail per phase). `research_journal/daily/*` = day-by-day "ran / worked / didn't work". This file = the structured results ledger that ties it together. Keep all three updated.
>
> **Last updated:** 2026-06-12.

---

## 1. Current State (read this first)

**System goal.** Upload a leaf photo + soil photo + crop + suspected cause → get a recommendation grounded in classical Indian agricultural texts (Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda), with disease/soil predictions, Grad-CAM heatmaps, and cited source chunks.

**What works end-to-end (as of 2026-06-12):** The full pipeline runs live (Phase 10 Streamlit UI on Colab+cloudflared). Disease model → soil model → Strategy-B query rewrite → grounded RAG answer with citations → highlighted source chunks. Demonstrated working with real images.

**The open problem we are actively fixing:** the **disease classifier's PlantDoc (in-the-wild) stage**. It predicts correctly but with only 72.3% accuracy and — critically — Grad-CAM shows it attends to **background, not the leaf**. We have diagnosed the cause (see §5) and are testing the fix (LP-FT, see §6).

**Component status snapshot:**

| Component | Status | Headline number |
|---|---|---|
| IKS corpus + RAG retrieval | ✅ Working | Strategy B retrieval score 0.59–0.96 |
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
Live demo: upload leaf+soil → predictions + 4 Grad-CAM heatmaps → Strategy-B query → grounded cited answer → highlighted chunks. Colab + cloudflared tunnel (localtunnel dropped Streamlit's JS chunks). Disease model behind a config switch; no-leaf guardrail (native R class OR segmentation fallback). T4 memory plan: embedder+reranker on CPU. 13 app tests pass.

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

**Result.** _⏳ Pending — fill in accuracy delta (LP-FT vs OLD) + Grad-CAM verdict when the run finishes._

---

## 7. Negative Results (paper ammunition — keep these honest)

A thesis is stronger for documenting what *didn't* work and why.

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
