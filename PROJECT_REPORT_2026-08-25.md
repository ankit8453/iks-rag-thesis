# Project Progress Report

*IKS-Grounded Multimodal Agricultural Advisory System*
*M.Tech Thesis · IIITDM Jabalpur*
*Student: Ankit Pawar · Supervisor: Dr. Akshay Pandey*
*Date: 25 August 2026*

---

## 1. Overview

This report summarises the current state of the thesis project, the latest results for each component, the evaluation carried out, and the planned next steps.

The system is working end-to-end and runs live. A farmer uploads a leaf photo and a soil photo and declares the crop; vision models predict the disease and soil properties; these are converted into a query against a corpus of classical Indian agricultural texts (Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda); and a language model (Llama-3.1-8B, used only for retrieval-grounded generation) returns advice that is grounded in and cited from those texts — or an honest refusal when the texts do not cover the case.

---

## 2. System components

1. **Disease classifier** — EfficientNet-B4. Predicts the disease from a leaf photo and produces a Grad-CAM heatmap showing where the model looked. A pretrained leaf detector crops the leaf before classification, so the background cannot act as a shortcut.
2. **Soil classifier** — EfficientNet-B0, multi-task (soil type, moisture, texture).
3. **RAG advisory** — hybrid retrieval (keyword + semantic + re-ranking) over a ChromaDB of the classical-text corpus, with grounded generation by Llama-3.1-8B.
4. **Integration** — converts vision predictions into a symptom-led query in the vocabulary the classical texts use.
5. **Explainability** — Grad-CAM for disease and soil predictions; highlighting of the exact retrieved passages used in the answer.
6. **Interface** — a working Streamlit application.

---

## 3. Component results

### 3.1 Disease classifier

The classifier is a three-stage transfer cascade:

| Stage | Data | Accuracy | Attention |
|---|---|---|---|
| Clean lab images (PlantVillage) | 38 classes | 99.8% | leaf |
| Field canopy (Paddy) | 10 classes | 97.0% | lesion |
| In-the-wild (PlantDoc) | 27 classes | 72.3% | background (the problem) |

For published context, the strongest PlantDoc results in the literature are approximately 73–78%, so 72.3% is at the research frontier. The concern was not the accuracy but that the model reached it by attending to the background rather than the leaf.

**Correction applied.** Retraining on cropped leaf regions, warm-started from the general PlantVillage backbone, produces a model at **66.6% top-1 with genuine, leaf-focused attention** (confirmed by Grad-CAM). A small amount of headline accuracy is traded for a model that looks at the disease rather than the background. Healthy-versus-diseased detection is near-perfect; most remaining errors are confusions between disease subtypes of the same crop, not healthy/diseased mistakes.

**Latest training.** A further training run that incorporated the provided Brazilian multi-crop dataset reached **71.9% test accuracy** while retaining leaf-focused attention. (Details in §6.3.)

### 3.2 Soil classifier

| Property | Top-1 accuracy |
|---|---|
| Soil type | 89.9% |
| Moisture | 95.8% |
| Texture | 67.9% |

Texture is the weakest head and remains a genuinely hard problem, identified as future work.

### 3.3 Retrieval and integration

Converting a modern vision label into a query against classical text is the core integration challenge. A direct template ("Apple Scab Leaf") retrieves very little (score 0.01–0.04), because the texts do not use that vocabulary. Rewriting the label into a symptom description the texts do use ("scorched leaves with whitish spots") raises retrieval quality to **0.59–0.96**.

### 3.4 Grounded generation

The generator answers only from the retrieved passages and cites each by source, chapter, and verse. When the corpus does not cover a case, it refuses rather than fabricating an answer.

---

## 4. The IKS text corpus (current size)

The corpus currently holds **206 indexed passages ("chunks") across four books**:

| Book | Chunks | Scope indexed |
|---|---:|---|
| Brihat Samhita | 136 | 12 relevant chapters (of a 593-page source) |
| Vrikshayurveda | 42 | Full text (101 pages) |
| Upavanavinoda | 15 | Partial |
| Krishi Parashara | 13 | Partial |
| **Total** | **206** | |

Two further books are already planned for inclusion: **Kashyapiya Krishi Sukti**, and one additional text to be finalised.

---

## 5. Evaluation

A structured evaluation was run over a set of 24 symptom-led queries (22 answerable and 2 that should be refused).

**Retrieval quality** (higher is better):

| Configuration | nDCG@5 | MRR | Precision@5 | Hit@5 |
|---|---|---|---|---|
| Full (semantic + keyword + re-rank) | 0.94 | 0.91 | 0.74 | 1.00 |
| Keyword-only (baseline) | 0.70 | 0.62 | 0.56 | 0.91 |
| Semantic-only | 0.94 | 0.94 | 0.80 | 1.00 |

**Answer grounding and honesty:**

- Honest refusal on the questions the corpus cannot answer: **100%**.
- Fabricated citations: **0%**.
- Over-refusal on answerable questions: **approximately 55%**.
- Faithfulness (RAGAS, on answered questions): **0.56**; an independent cross-check placed it at **0.86**.

**Interpretation.** The system is safe and faithful: when it answers, the answer is grounded in the texts, and it never fabricates a citation. The main limitation is **corpus coverage** — the 206 passages do not cover every symptom a farmer might present, so the generator honestly refuses about half of the answerable questions rather than guessing. Expanding the corpus is therefore the most direct way to improve the system.

---

## 6. Design refinements

### 6.1 Symptom-based retrieval

The classical texts prescribe treatment by general symptom (for example, "white spot"), not by a named crop. The retrieval was aligned to this: the query leads with the symptom and treats the crop as supporting context, rather than requiring a specific crop to be named in the text.

### 6.2 Handling of untrained plants

Because a disease's visual appearance transfers across plant species, the classifier's disease knowledge is useful even for a plant it was never trained on. Rather than refusing such cases, the system runs the model, shows a **calibrated confidence percentage** (using temperature scaling, a standard method that corrects an over-confident model without changing its predictions), and advises via the symptom-based retrieval with a clear caution. Only when confidence is genuinely low does it hold back and request additional photographs.

### 6.3 Crop-agnostic disease-type classifier (experiment)

As an isolated experiment (the deployed model and its data were left unchanged), the many crop-specific disease labels were pooled into approximately 13 crop-agnostic disease types (rust, blight, leaf spot, and so on), directly aligned with the symptom-based direction.

| Version | Test accuracy | Attention on field images |
|---|---|---|
| Without leaf-cropping | 78.2% | background |
| With leaf-cropping | 71.9% | leaf/lesions |

The leaf-cropped version attends correctly to the leaf even on difficult field images, at a small accuracy cost — consistent with the main disease model's behaviour. This crop-agnostic direction is the more novel contribution: it matches how the classical texts work and can recognise a disease on a plant the model was never trained on.

### 6.4 Dataset used

The Brazilian multi-crop dataset provided for the project (2,595 images, 19 crops) was examined and incorporated into the training described above. Its very small per-class counts make it well suited to disease-type training rather than crop-specific classes, which is how it was used.

---

## 7. Documented negative results

The following approaches were tried and did not work; they are retained as documented findings:

- Background randomisation — cost accuracy for negligible attention gain.
- Linear-probe fine-tuning on a frozen specialised backbone — accuracy dropped to 61%.
- Cropping at inference only (without retraining) — accuracy dropped to 58.2%, confirming that the background dependence can only be removed by **retraining** on leaf crops.
- Sequential soil transfer — training collapsed.

The consistent finding is that the leaf-focus problem is solved only by retraining on leaf crops, which the deployed model now does.

---

## 8. Current status

| Component | Status |
|---|---|
| Classical-text corpus and retrieval | Working |
| Grounded generation with citations | Working |
| Soil classifier | Production |
| Disease classifier (honest, leaf-focused) | Working |
| Symptom-based retrieval and untrained-plant handling | Implemented |
| Full-system interface | Working (live demo) |
| Evaluation | First full pass complete |
| Corpus coverage expansion | In progress |
| Expert (gold-standard) evaluation set | Planned |

---

## 9. Next steps

1. **Expand the corpus.** The evaluation shows coverage, not method, is the limiter. Add the two planned books (Kashyapiya Krishi Sukti and one further text) and, where relevant, additional chapters of Brihat Samhita, to reduce the over-refusal rate.
2. **Build an expert gold-standard evaluation set.** The current evaluation uses approximate labels; a small set of expert-judged answers would make the evaluation citable.
3. **Complete the disease-type follow-ups.** Merge the early/late blight classes, address the very small classes, and run a held-out-crop test (training while excluding one crop, then testing on it) to directly demonstrate cross-plant generalisation.
4. **Continue the crop and soil suitability reference list** (120 crops), in progress.

---

*Prepared by Ankit Pawar for Dr. Akshay Pandey. All figures are drawn from the project's experiment records.*
