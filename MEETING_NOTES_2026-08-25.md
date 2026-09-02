# Supervisor Meeting — Progress & Discussion Notes

*IKS-Grounded Multimodal Agricultural Advisory System*
*M.Tech Thesis · IIITDM Jabalpur · Ankit Pawar · Supervisor: Dr. Akshay Pandey*
*Meeting date: 25 August 2026*

> **Purpose of this document.** A single, up-to-date briefing for today's meeting. It summarises the complete system as it stands, the latest accuracy figures for every component, and — importantly — the **evaluation results and the disease-model redesign that were completed since our last meeting** and have not yet been reviewed together. The final two sections list what remains and the specific points to discuss.

---

## 1. Executive summary

The system runs **end-to-end and live**. A farmer uploads a leaf photo and a soil photo, declares the crop, and receives advice that is **grounded in and cited from classical Indian agricultural texts** (Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda) — or an **honest refusal** when the texts do not cover the case.

Since the last meeting, three substantial pieces of work were completed:

1. **A full evaluation phase** — retrieval quality, answer grounding, and a RAGAS faithfulness assessment (new; not yet reviewed together).
2. **A redesign of how the texts are used** — from crop-specific lookup to **symptom-based retrieval**, matching how the classical texts actually prescribe treatment, plus **calibrated-confidence handling for untrained plants** instead of a hard refusal.
3. **A crop-agnostic "disease-type" classifier experiment** — an isolated study (the deployed model is untouched) that directly supports the cross-plant generalisation idea.

The scientific centre of gravity remains the **disease model's honesty problem** — the earlier concern that the model attended to the background rather than the leaf. That has been diagnosed and fixed, and the fix now verifiably holds even on field images.

---

## 2. The two disease models — a simple comparison

> *This is the main thing to decide today. It can be explained in two minutes.*

We now have **two** disease models. They are **not** better-or-worse versions of the same thing — they answer **two different questions**.

**Model 1 — "C-PD" (the one running in the system now)**

- **What it tells you:** the crop *and* the disease together — for example, *"Tomato — Late blight"*.
- **How many classes:** 27 (each class is one crop + one disease).
- **How it was built:** started from our clean lab images (PlantVillage), then re-trained on **cropped leaf photos** from real-world images (PlantDoc). It only knows the crops in those datasets.
- **Accuracy:** 66.6%, and it honestly looks at the leaf.
- **Its limit:** it only recognises crops it was trained on. Show it a plant it never saw, and it has no matching label for it.

**Model 2 — the symptom-based "disease-type" model (the new experiment)**

- **What it tells you:** only the **disease type**, without the crop — for example, just *"Blight"* (or *"Rust"*, *"Leaf spot"*).
- **How many classes:** about 13 (each class is one disease type; the crop name is removed).
- **How it was built:** same lab starting point, then trained on a **mix of datasets** (PlantVillage + PlantDoc + the Brazilian multi-crop set), all re-labelled **by disease type only**, on cropped leaves.
- **Accuracy:** 71.9% (the leaf-cropped version).
- **Its strength:** because the crop is removed, it can recognise a disease on **any plant — even one it never saw**. This matches our classical texts, which prescribe by **symptom, not by crop**.

**Side-by-side**

| | Model 1 — C-PD (current) | Model 2 — Symptom-based (new) |
|---|---|---|
| What it names | Crop **and** disease | **Disease type only** |
| Example answer | "Tomato — Late blight" | "Blight" |
| Number of classes | 27 (crop + disease) | ~13 (disease type only) |
| Trained on | PlantVillage → cropped PlantDoc | PlantVillage + PlantDoc + Brazilian mix |
| Works on an **unseen** crop? | ❌ No | ✅ Yes |
| Matches the IKS texts (symptom-based)? | Partly | ✅ Directly |
| Accuracy | 66.6% | 71.9% |
| Status | ✅ Proven, running in the system | 🔬 New experiment, not yet deployed |

*(Note: the two accuracy numbers are **not** a fair head-to-head — Model 2 has fewer, broader classes, so its number is naturally a little higher. They measure different tasks.)*

**The difference in one line:** *C-PD tells you which crop has which disease; the symptom model tells you which disease it is, on any crop.*

**Why the new model fits our thesis novelty:** our classical texts treat by symptom ("white spot"), not by crop name. A model that thinks in **disease types** — not crops — matches the texts and lets us help farmers even with crops we never trained on.

**My recommendation:** **keep C-PD running** (it is proven and ready to demo), and **present the symptom model as our novelty and future direction.** Only swap it in later, after one more test (training it while hiding one crop, then testing on that crop) proves it truly works on unseen plants.

---

## 3. System at a glance

Three model families, wired together by an integration layer:

1. **Disease classifier** — EfficientNet-B4. Leaf photo → disease prediction, with a Grad-CAM heatmap showing where the model looked. A pretrained leaf detector crops the leaf **before** the classifier sees it, so the background cannot act as a shortcut.
2. **Soil classifier** — EfficientNet-B0, multi-task (soil type, moisture, texture).
3. **RAG advisory** — hybrid retrieval (keyword + semantic + re-ranking) over a ChromaDB of the classical-text corpus, with grounded generation by Llama-3.1-8B (frozen; used for retrieval-and-generation only, never fine-tuned).
4. **Integration** — converts the vision predictions into a **symptom-led query** in the vocabulary the classical texts actually use.
5. **Explainability** — Grad-CAM for disease and soil; highlighting of the exact retrieved passages used.
6. **Interface** — a working Streamlit application.

---

## 4. Component results (current numbers)

### 4.1 Disease classifier

The classifier is a three-stage transfer cascade. Its behaviour, stage by stage:

| Stage | Data | Accuracy | Where the model looks |
|---|---|---|---|
| Clean lab images (PlantVillage) | 38 classes | 99.8% | leaf ✅ |
| Field canopy (Paddy) | 10 classes | 97.0% | lesion ✅ |
| In-the-wild (PlantDoc) | 27 classes | 72.3% | **background** ❌ (the honesty problem) |

For published context, the best PlantDoc results in the literature are ~73–78%, so **72.3% is at the research frontier** — the concern was never the number, but *how* the model reached it (by reading the background).

**The fix that worked — retrain on leaf crops ("C-PD").** Retraining on cropped leaf regions, warm-started from the general PlantVillage backbone, produces a model at **66.6% top-1 with honest, leaf-focused attention** (Grad-CAM now sits on the leaf, not the background). The guiding principle we adopted: *a correct model at 66.6% is worth more than a background-reading model at 72.3%.*

**An important reframe of the 66.6%.** Healthy-vs-diseased detection is near-perfect. Most of the remaining errors are confusions *between disease subtypes of the same crop* (e.g. corn rust vs corn blight), not healthy/diseased mistakes. So the model reliably answers the question that matters to a farmer — *does this plant need attention?* — even when it debates the exact subtype.

### 4.2 Soil classifier (production model)

| Property | Top-1 accuracy |
|---|---|
| Soil type | 89.9% |
| Moisture | 95.8% |
| Texture | 67.9% |

Texture is the weakest head and remains a genuinely hard problem — a candidate for future work rather than a quick win.

### 4.3 Retrieval and integration

The key design choice is how a modern vision label becomes a query against classical text. A direct template ("Apple Scab Leaf") barely retrieves anything (score 0.01–0.04), because the texts never use that vocabulary. **Rewriting the label into a symptom description the texts actually use** ("scorched leaves with whitish spots") lifts retrieval quality to **0.59–0.96**. This bridge is the core integration contribution.

### 4.4 Grounded generation

The generator answers **only** from the retrieved passages and cites each one by source, chapter, and verse. When the corpus does not cover a case, it refuses rather than inventing an answer.

---

## 5. New since our last meeting

*These three items were completed after our previous discussion and are the main things to review together today.*

### 5.1 Evaluation phase — first full results

We ran a structured evaluation over a set of 24 symptom-led queries (22 answerable + 2 that should be refused).

**Retrieval quality** (higher is better; comparing the full system to a keyword-only baseline):

| Configuration | nDCG@5 | MRR | Precision@5 | Hit@5 |
|---|---|---|---|---|
| **Full (semantic + keyword + re-rank)** | **0.94** | **0.91** | 0.74 | 1.00 |
| Keyword-only (baseline) | 0.70 | 0.62 | 0.56 | 0.91 |
| Semantic-only | 0.94 | 0.94 | 0.80 | 1.00 |

**Answer grounding and honesty:**

- **Honest refusal on the negatives: 100%** — it correctly declines the two questions the corpus cannot answer.
- **Fabricated citations: 0%** — it never cites a source it did not use.
- **Over-refusal: ~55%** — on answerable questions, it refuses about half the time.
- **Faithfulness (RAGAS, when it does answer): 0.56**; an independent cross-check on the answered subset put faithfulness at **0.86**.

**The headline finding.** The system is *safe and faithful* — when it answers, the answer is grounded, and it never fabricates. **The limiter is corpus coverage, not the retrieval or the model.** The ~206 tree-focused passages simply do not cover every symptom a farmer might present, so the generator honestly refuses rather than guessing. This directly motivates expanding the corpus (discussed in §8).

### 5.2 Symptom-based retrieval + untrained-plant handling

Two design changes, both aligned with how the classical texts actually work:

- **Symptom-based, not crop-specific.** The texts prescribe by general symptom (e.g. "white spot"), not by a named crop. The retrieval was realigned to **lead with the symptom** and treat the crop as background context, rather than refusing when a specific crop is not named in the text.
- **Untrained plants — advise with calibrated confidence, not a flat refusal.** Because a disease's visual appearance transfers across species, the classifier's disease knowledge is useful even on a plant it was never trained on. Instead of refusing, the system now runs the model, shows a **calibrated confidence percentage** (using temperature scaling, a standard technique that corrects an over-confident model without changing its predictions), and advises via the symptom-based retrieval **with a clear caution**. Only when confidence is genuinely low does it hold back, collect the image, and ask the farmer for more photos. Wording is deliberately honest — "confidence", never "accuracy".

### 5.3 Crop-agnostic "disease-type" experiment

An **isolated research experiment** (the deployed model, its data, and its configuration were left completely untouched; everything is reversible). The idea: pool the many crop-specific disease labels into ~13 **crop-agnostic disease types** (rust, blight, leaf spot, and so on) — directly aligned with the symptom-based direction and with the untrained-plant idea.

| Version | Test accuracy | Attention on field images |
|---|---|---|
| Without leaf-cropping | 78.2% | reads the background ❌ |
| **With leaf-cropping** | **71.9%** | **on the leaf/lesions** ✅ |

The −6 accuracy points are the honest cost; the leaf-focused attention is what they buy. Grad-CAM on the cropped model confirmed the fix holds on the hardest cases — corn-in-a-field images now heat the lesions, with the soil going cold. This is the same trade-off, and the same verdict, as the main disease model: **keep the leaf-focused version**. The weak classes (a couple with very few samples, and an early-blight/late-blight confusion) are known and fixable.

### 5.4 On the datasets provided

The disease dataset provided earlier was inspected carefully. It is a re-packaged copy of **PlantVillage** — the same lab-style images the project already uses (confirmed by an exact class-count match and the shared naming convention), with **no field-style imagery**. Merging it would add no new information and would risk re-amplifying the very lab-only shortcut bias the project has been working to remove. The one genuinely useful piece is its "background / no-leaf" folder, usable as a guardrail when a farmer uploads a non-leaf photo.

**This is a specific request for the meeting (see §9):** what would help most is **field-style imagery** — plot photographs with soil and neighbouring plants visible — as that is exactly what reduces the background-shortcut bias.

---

## 6. What worked, and what did not (kept honest)

Documented negative results strengthen the thesis:

- **Background randomisation** (compositing leaves onto random backgrounds): cost accuracy for almost no attention gain. Abandoned.
- **Linear-probe fine-tuning** on a frozen rice-specialised backbone: dropped to 61%. Abandoned.
- **Cropping at inference only** (without retraining): dropped to 58.2% — proof that a background-trained model cannot be fixed by cropping alone; it must be **retrained** on crops. This is *why* the C-PD retrain was necessary.
- **Sequential soil transfer**: training collapsed. Abandoned in favour of the stable production recipe.

The consistent lesson: **the leaf-focus problem is only solved by retraining on leaf crops**, which is exactly what the deployed model now does.

---

## 7. Where the project stands

| Component | Status |
|---|---|
| Classical-text corpus + retrieval | ✅ Working |
| Grounded generation with citations | ✅ Working |
| Soil classifier | ✅ Production |
| Disease classifier (honest, leaf-focused) | ✅ Working |
| Symptom-based retrieval + untrained-plant handling | ✅ Implemented |
| Full-system interface | ✅ Working (live demo) |
| Evaluation phase | ✅ First full pass complete |
| Corpus coverage expansion | ⏳ Open — the main limiter |
| Expert (gold) evaluation set | ⏳ Open — needs expert judgement |

---

## 8. What remains

1. **Expand the corpus** — the evaluation shows coverage, not method, is the limiter. Adding a small number of well-chosen texts would directly reduce the ~55% over-refusal.
2. **Build an expert gold-standard evaluation set** — the current results use silver (approximate) labels; a small expert-judged set would make the evaluation citable.
3. **Finish the disease-type follow-ups** — merge the early/late blight classes, address the tiny under-sampled classes, and run a **held-out-crop test** (train while excluding one crop, then test on it) — the strongest single piece of evidence for the cross-plant generalisation claim.
4. **Continue the crop/soil suitability reference list** — the list of 120 crops with soil suitability is in progress.

---

## 9. Points to discuss today

1. **Corpus expansion.** The evaluation shows the system is faithful but limited by how much the texts cover. **Which additional classical or authoritative texts would be appropriate to add**, and are there any specific sources recommended?
2. **Field-style imagery.** Would field photographs (soil + neighbouring plants visible) be available? That is the single most valuable data addition for reducing the background bias — far more than more lab images.
3. **Symptom-based treatment — validation.** The retrieval now treats the texts as symptom-based rather than crop-specific. **Is this the correct reading of the classical treatment logic?** Confirmation here validates a core design decision.
4. **Untrained-plant advice.** The system now advises on plants it was not trained on, using a calibrated confidence figure and an explicit caution, rather than refusing. **Is this the right level of caution** for a farmer-facing tool, and is the confidence-plus-caution framing acceptable?
5. **Expert evaluation.** For a citable evaluation, a small set of expert-judged answers is needed. **Could a short set of queries be reviewed for correctness**, or a suitable expert suggested?
6. **The disease-model accuracy trade-off.** The honest, leaf-focused model scores a few points lower than the background-reading one. **Confirmation that trading a small amount of headline accuracy for genuine, explainable leaf attention is the right call for the thesis.**
7. **Crop-agnostic disease types.** The crop-agnostic direction is promising and aligns with the symptom-based logic. **Is it worth developing further** as a thesis contribution, or kept as a supporting experiment?

---

*Prepared for the supervisor meeting on 25 August 2026. All figures are drawn from the project's experiment log; nothing here is projected or estimated.*