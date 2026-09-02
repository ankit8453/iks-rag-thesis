# Project Sync — Catch-up for Planner-Claude (chat)

> **Purpose of this file.** Ankit works in a loop: *chat-Claude (you)* writes the prompts → Ankit pastes them into *Claude Code* → Code implements on the repo. Since the **Phase 10 prompt**, Code and Ankit have done a lot of work that chat never saw. This document brings you fully up to date: what you already know, what changed since, why, what broke, and what's pending. Read it top to bottom and you'll be in sync with the actual repo state as of **2026-06-15**.

---

## 0. Project one-liner (so we're grounded)

**Thesis:** *IKS-Grounded Multimodal Agricultural Advisory System* — IIITDM Jabalpur, M.Tech, Ankit Pawar, supervisor **Dr. Akshay Pandey**.

**The system:** a farmer uploads a **leaf photo** + a **soil photo** → vision models predict disease + soil properties → those labels are bridged into a query against a corpus of **classical Indian agricultural treatises** (Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda) → an LLM (Llama-3.1-8B, **frozen**, RAG only) returns **grounded, cited** advice — or an **honest refusal** when the corpus doesn't cover it.

**Phases:** 3 (corpus) · 5 (disease) · 6 (soil) · 7 (RAG) · 8 (integration) · 9 (explainability) · 10 (Streamlit UI).

---

## 1. What you (chat) ALREADY know

- The overall architecture and all phases up to and including the **Phase 10 prompt** you wrote (build the Streamlit UI wiring disease + soil + RAG + integration + explainability).
- That the disease model had a **background-attention problem** flagged by Dr. Pandey (model looked at background, not the leaf).
- The general RAG design (hybrid retrieval + Llama generator).
- Strategy A (template baseline) vs Strategy B (LLM-mediated query construction) existed as a concept.

**Everything below is what happened AFTER that — most of which you have not seen.**

---

## 2. THE BIG ONE — disease model background-bias: diagnosed and FIXED

This was the main scientific work since you last looked. Dr. Pandey's concern ("not getting leaf attention properly") was taken seriously and run down step by step.

**Diagnosis:**
- The disease cascade is EfficientNet-B4: PlantVillage (38) → Paddy Doctor (10) → **PlantDoc (27)**. The background-bias is introduced at the **PlantDoc fine-tuning stage** (PlantDoc images have cluttered, real-world backgrounds; the model learned background shortcuts).

**Things tried that did NOT work (all documented):**
- **LP-FT** (linear-probe-then-fine-tune): failed (~61%) — it froze the rice-narrow Paddy backbone.
- **Detect-then-crop with YOLO at inference only**: held-out 58.2%. The "91.9%" we first saw was **data leakage** (detection test set ⊂ classifier train set).

**The fix that WORKED — "C-PD" (Cropped-PlantDoc retrain):**
- Retrain on **leaf CROPS** (GT-box leaf regions), warm-started from the **PlantVillage** backbone (NOT the rice-narrow Paddy one). Approach follows Singh et al. 2020.
- Repo: **`ankit-iiitdmj/iks-disease-plantdoc-crop`**.
- Result: **66.6% top-1, but HONEST leaf attention** (Grad-CAM now on the leaf, not background). Decision rule we adopted: *"a correct model at 66.6% beats a broken one at 72.3%."*

**Crucial reframe of the 66.6%:**
- **Healthy-vs-diseased is near-perfect.** The model has 10 healthy + 17 disease classes. Most top-1 errors are **within-crop disease-SUBTYPE confusion** (corn rust ↔ corn blight, potato early ↔ late blight), NOT healthy/diseased mistakes. So "66.6%" undersells it — it *works where it matters* (does this plant need attention or not).

**Pipeline is now CROP-FIRST everywhere:** a pretrained YOLO leaf detector (`foduucom/plant-leaf-detection-and-classification`, **conf=0.10** — 0.25 was too strict) crops the leaf BEFORE the C-PD classifier sees it.

---

## 3. Explainability (Phase 9) — finalized

- **Grad-CAM corner-hotspot artifact was a VISUALIZATION bug, not the model.** Fix: `eigen_smooth=True` + target layer `blocks[-2]` (cleaner than `conv_head`). New shared function `disease_gradcam_eigen()` in `src/explain/gradcam.py`.
- **Rule: heatmap ONLY for disease.** Healthy leaf → no heatmap (a heatmap on a healthy leaf is meaningless / confusing). Diseased → eigen Grad-CAM on the lesion.
- **Rule: advisory ONLY for disease** (gating — see §4).
- Clean demo indices for diseased lesion attention: **idx 248, 106, 22** (PlantDoc test).
- Decision: **keep Grad-CAM** (it's the interpretability evidence that answers Dr. Pandey) but **stop perfecting it** — it's locked.
- A **diagnosis summary notebook** (`notebooks/disease_diagnosis_summary.ipynb`) was built **specifically for Dr. Pandey**: stage-wise hero figure (PV ✅ → Paddy ✅ → PlantDoc-OLD background ❌ → C-PD fix ✅) + "techniques tried" table + proven-ceiling note.

---

## 4. The advisory gating + Strategy B (this is the NOVELTY — important for you)

A real logical flaw was caught and fixed: the advisory was generating treatment queries **even for healthy leaves** ("organic treatment for healthy peach leaf" — nonsense). Now:

- **Healthy leaf → NO query, NO retrieval.** Just "✓ no treatment needed, keep monitoring."
- **Diseased leaf → Strategy B query → retrieval → grounded cited advice.**

**Strategy B = the contribution.** It uses the frozen Llama to **rewrite** the modern vision labels into the *descriptive/symptomatic vocabulary the classical corpus actually uses*. Example: "rice blast" / "sandy_loam" → "scorched leaves with whitish lesions in fertile riverine soil." Classical texts don't contain modern disease names, but they DO describe the underlying phenomena — Strategy B bridges that gap. (Strategy A = blunt template baseline, retrieval scores 0.001–0.04; Strategy B = 0.59–0.96.)

**The LLM is NOT fine-tuned.** Llama-3.1-8B is frozen, steered by prompts (Strategy B rewrite prompt + §17 grounding prompt). All domain knowledge comes from the **retrieved corpus** (RAG), not the weights. (Ankit asked this explicitly — confirm it if he asks you again.)

---

## 5. Corpus coverage — a HONEST limitation you must know

- The corpus is **tree/fruit science** (Vrikshayurveda = "science of trees"). It covers **tree and fruit diseases** (e.g. apple scab → real, grounded treatment) but **NOT grains** (corn, rice, wheat → honest refusal).
- This is **by design and is part of the integrity story**, not a bug. The demo deliberately pairs:
  - **corn** → honest refusal ("corpus doesn't cover this") = *integrity*
  - **apple scab** → real grounded treatment with citations = *the novelty delivers*
- Retrieval gotcha: the **CONTAGION** causal pathway over-narrows the query ("spreading from neighbouring plants"). Use **UNKNOWN** or **SOIL_DRIVEN** for a richer answer.

---

## 6. Phase 10 UI — BUILT (you asked for this; here's how it actually turned out)

The Streamlit UI (`app/`) was rewritten for the updated scenario + made "modern, creative, futuristic" (Ankit's ask):

- **Crop-first pipeline:** YOLO `LeafCropper` (conf 0.10) → C-PD classify → `is_healthy()` gate.
- **Heatmap only for disease** (eigen Grad-CAM); healthy shows "no region."
- **Advisory gated** (healthy → "no treatment"; diseased → Strategy B → grounded answer + cited chunks).
- **Futuristic look:** dark gradient background, glassmorphism cards, gradient hero title (🌿 VṚKṢA), green/red status pills, gradient buttons.
- Files: `app/config.py`, `app/loaders.py`, `app/streamlit_app.py`, `src/explain/gradcam.py`, launcher `notebooks/phase10_launch_ui.ipynb`.
- Launches on Colab T4 via cloudflared tunnel (cloudflared > localtunnel — handles Streamlit's lazy JS chunks).

**Phase 10 BUG found on the live UI (2026-06-15) and fixed:**
- A **corn** leaf (correctly detected "Corn rust leaf" 98%) produced a query about **rice**, because the query pulled `crop_type` from the **sidebar dropdown** (left on its default "rice") instead of the detected label.
- **Fix:** new `app_config.crop_from_disease()` derives the crop from the disease label ("Corn rust leaf" → corn). The crop dropdown now defaults to **"auto"** (use detection); explicit picks override. The query, the `MultimodalContext`, and the status banner all use the detected crop now.
- (The corn refusal itself is still correct — corpus gap. The fix only makes the query *consistent*.)

---

## 7. RAG + embedding facts (Ankit asked; capture for the paper)

- **Embedder:** `BAAI/bge-large-en-v1.5` (1024-dim), sentence-transformers, normalized → cosine. Chosen because: top open English retriever at selection time; corpus is English-translated; free/local (budget); small corpus (~206 chunks) so accuracy > speed; pairs with the bge reranker. Multilingual fallback `bge-m3` configured but English is primary.
- **Retrieval is NOT Naive RAG.** It's **Advanced/Modular RAG**: hybrid (dense bge-large + sparse BM25, `hybrid_alpha=0.5`) → cross-encoder rerank (`bge-reranker-base`) → grounded generation with citations + §17 refusal. Plus **query transformation** (Strategy B). The multimodal-label→classical-vocabulary bridge is the genuine novelty.
- Verdict on switching RAG types (GraphRAG / Self-RAG / Agentic): **not recommended** — the bottleneck is corpus *coverage*, not architecture. GraphRAG is overkill for 206 chunks. The one free framing win: call the existing refusal **CRAG-style faithfulness gating**.

---

## 8. Current repo state (as of 2026-06-15)

- **Branch:** `cleanup/pdf-alignment`. **All work is LOCAL commits only — never pushed.** Ankit does the push himself. (Hard rule.)
- **Build on the OLD cascade, not the "R" retrain** (another hard rule from Ankit).
- Disease model: **C-PD** (`iks-disease-plantdoc-crop`) — final, leaf-attention.
- Phase 9 notebook: FINAL, clean, 14 cells.
- Phase 10 UI: built + crop-derivation bug fixed.
- Latest commits: `71e05f6` (crop fix) ← `6ae0661` (Phase 10 UI FINAL) ← `cb13500` (Phase 9 Cell 10).
- Docs kept current: `EXPERIMENT_LOG.md` (master ledger), `research_journal/daily/2026-06-13.md`.

---

## 9. What's PENDING (Ankit's to-do, not done yet)

1. `git push origin cleanup/pdf-alignment` (Ankit pushes; Code never does).
2. Run final **Phase 9** end-to-end on Colab (Run all) — capture the healthy-vs-diseased number + showcase + gated Cell 10 output.
3. Run the new **Phase 10 UI** on Colab and confirm the futuristic look + the corn/apple-scab demo pair.

---

## 10. Working constraints you (chat) should bake into every future prompt

- **Local commits only — never instruct a push.** Ankit pushes manually.
- **Build on the OLD cascade, not R.**
- **Budget-sensitive:** Ankit is on a tight paid budget for any paid API (e.g. Gemini). Default to free/local; ask before spending.
- **Documentation discipline:** keep `EXPERIMENT_LOG.md` + the daily research journal updated with the **WHY** behind pivots, not just the what. (Ankit has flagged this twice.)
- **Honesty over vanity metrics:** a correct 66.6% beats a broken 72.3%; honest refusal is a feature.
- **Disease work is LOCKED** — don't reopen it; the contribution is the *system + the diagnosis*, not chasing accuracy.

---

## 11. TL;DR for you in three sentences

We diagnosed and fixed the disease background-bias with a crop-retrained model (C-PD, 66.6% but honest leaf attention; healthy-vs-diseased near-perfect), finalized Phase 9 explainability (eigen Grad-CAM, heatmap+advisory gated to disease only), and built + bug-fixed the futuristic Phase 10 Streamlit UI (crop-first, crop now derived from the detected disease, Strategy-B grounded advisory). The corpus is tree/fruit science so grains get an honest refusal (by design), and the RAG is Advanced/Modular (hybrid + rerank + query-bridging), with the multimodal→IKS query bridge as the novelty. Everything is local commits on `cleanup/pdf-alignment`; pending items are Ankit pushing and running Phases 9 + 10 on Colab.
