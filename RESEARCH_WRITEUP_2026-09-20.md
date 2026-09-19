# An IKS-Grounded Multimodal Agricultural Advisory System

**Complete research write-up**

**Ankit Pawar** — M.Tech (Computer Science & Engineering), PDPM IIITDM Jabalpur
**Supervisor:** Dr. Akshay Pandey, Department of CSE, PDPM IIITDM Jabalpur
**Agronomy consultation:** Dr. Sunita T. Pandey
**Date:** 20 September 2026

---

## Abstract

India's classical agricultural treatises — Vrikshayurveda, Brihat Samhita, Krishi
Parashara, Upavanavinoda, Kashyapiya Krishisukti — record a substantial body of organic
plant-care knowledge that is effectively unusable today: it exists only as scanned
translations, it is organised by observed symptom and Ayurvedic cause rather than by modern
disease name, and no computational system draws on it.

This work builds and evaluates an end-to-end system that connects image-based plant-disease
and soil recognition to that textual knowledge, and that cites or abstains rather than
inventing an answer. Three components are contributed: (i) a verse-level digital corpus of
the treatises with provenance metadata; (ii) a **symptom-based semantic bridge** that
rewrites modern diagnostic labels into the descriptive vocabulary the texts actually use,
raising retrieval scores from 0.01–0.04 to 0.59–0.96; and (iii) a grounded generator that
answers only from retrieved passages, cites them at verse level, and refuses when evidence
is insufficient — producing **zero fabricated citations in every evaluation run**.

The central empirical result is not an accuracy figure. Across three corpus versions (206,
327 and 233 passages) the grounded-answer rate remained frozen at exactly 13.64% (3 of 22
queries) while over-refusal rose from 54.5% to 81.8%. Investigation established that this
is **not** a retrieval failure: measured with the pipeline's own cross-encoder, only 4 of
the 22 evaluation queries have a genuine match in the corpus, and 13 have none at all. The
classical texts classify plant disorder by Ayurvedic cause — *vata*, *pitta*, *kapha*, fire,
wound, insect attack — whereas modern diagnosis classifies by lesion appearance. For most
modern fungal leaf diseases **no mapping exists**, and the system's refusal is correct
behaviour. Quantifying that boundary is offered as a contribution in its own right.

A parallel interpretability study on the vision side localised a background-shortcut bias in
the disease classifier to a specific training stage, and corrected it by retraining on
leaf crops — trading 72.3% background-driven accuracy for 66.6% with verified leaf
attention.

**Keywords:** Indian Knowledge Systems, Vrikshayurveda, retrieval-augmented generation,
faithfulness, abstention, multimodal agriculture, explainable AI

---

## 1. Introduction

### 1.1 The problem

A farmer photographs a diseased leaf. Modern AI can name the disease with reasonable
accuracy. It cannot tell the farmer what the Indian agricultural tradition would have
prescribed for it — because that knowledge is locked in Sanskrit treatises available only as
scanned English translations, with no searchable structure.

Three barriers stand between those texts and use:

1. **No digital form.** The treatises exist as scanned PDFs of scholarly translations. There
   is no verse-level, machine-readable corpus.
2. **A vocabulary mismatch.** The texts describe *"leaves turning yellow with excessive
   paleness"*; modern diagnosis says *"Septoria leaf spot"*. Neither term retrieves the
   other.
3. **Fabrication risk.** A general-purpose language model asked about Vrikshayurveda will
   produce fluent, confident, and frequently invented remedies, attributed to verses that do
   not exist. For heritage knowledge this is not merely an error — it is a distortion of the
   tradition, and the user has no practical means of checking it.

### 1.2 Objectives

1. Construct a verse-level, metadata-annotated digital corpus of the classical treatises.
2. Connect image-based disease and soil recognition to that corpus.
3. Ensure every recommendation is either traceable to a cited passage, or refused.
4. Measure — rather than assume — how much of modern crop disease the texts can address.

### 1.3 Contributions

- **C1.** A verse-level digital corpus of six sources (five classical treatises plus one
  modern natural-farming manual), 233 passages with book, chapter, verse, translator and
  source-tier metadata.
- **C2.** A **symptom-based semantic bridge** (Strategy B) that rewrites modern vision
  labels into classical descriptive vocabulary. Proven by ablation: retrieval score
  0.59–0.96 versus 0.01–0.04 for direct label matching.
- **C3.** A grounded generator with verse-level citation and enforced abstention: 100%
  honest refusal on negative controls, 0% unfounded citations across all runs.
- **C4.** A rigorous interpretability diagnosis of the disease classifier that localises a
  background-shortcut bias to a specific cascade stage, three documented failed fixes, and a
  working correction (C-PD) that trades accuracy for verified leaf attention.
- **C5.** **A measured coverage boundary between classical IKS aetiology and modern disease
  taxonomy** — 4 of 22 modern leaf-disease queries have genuine textual support, 13 have
  none — together with the finding that a faithful system's correct response to the
  remainder is refusal.

---

## 2. Background: why these texts resist retrieval

Vrikshayurveda (Surapala, c. 10th century) is the principal source. Its organisation is the
crux of this project's central finding.

The text classifies plant disorder by **cause**, in the same humoral framework Ayurveda
applies to human medicine:

| Classical cause | What it covers | Prescribed treatment (examples) |
|---|---|---|
| *Vata* imbalance | slender crooked trunk, defoliation, general yellowing | flesh, marrow and ghee; sprinkling of *kunapa* water |
| *Kapha* imbalance | delayed fruiting, oozing without wounds | white mustard paste at the root; sesame-and-ash watering |
| Fire / lightning | drying, scorching | lotus mud smeared on the trunk, then *kunapa* |
| Physical wound | dryness after axe damage | paste of *nyagrodha* and *udumbara* bark, cow dung, honey, ghee |
| Insects / ants | foul smell, reduced leaf size | seed treatment with milk, mustard, *vidanga* |

A modern diagnostic label — "Tomato Septoria leaf spot" — does not appear anywhere in this
scheme, and cannot: the category did not exist. **The two traditions partition the same
physical reality along different axes.** Section 13 quantifies the consequence.

A second property, confirmed with our agronomy expert Dr. Sunita T. Pandey and by
independent literature review, is that the treatises are **predominantly symptom-general
rather than crop-specific**: a remedy given for a symptom applies to any plant showing it.
A peer-reviewed Vrikshayurveda expert system (Rananavare & Chitnis, *J. Ayurveda Integr.
Med.*, 2024) diagnoses purely from symptom questions and never asks which crop. A minority
of crop-named recipes does exist (mango, pomegranate, coconut), so the claim is
"predominantly", never absolute. This property drove a significant mid-project realignment
(Section 8.2).

---

## 3. Related work

**Plant disease recognition.** PlantVillage (lab images) is effectively solved at >99%.
PlantDoc (in-the-wild) is the realistic benchmark; published accuracy sits at roughly
70–78%. Reported accuracy on such datasets is known to be inflated by background
correlation, which motivates the interpretability work in Section 6.

**Retrieval-augmented generation.** Standard RAG couples dense retrieval with a generator
conditioned on retrieved text. Applications to classical or low-resource corpora are rare,
and evaluation of *abstention* — whether a system correctly declines — is rarer still. Most
RAG benchmarks assume every query is answerable, an assumption this work shows to be
consequential (Section 13).

**AI and Indian Knowledge Systems.** Recent work treats Indian epistemological frameworks
as computational objects. This project differs in domain (agriculture rather than logic) and
in emphasis: it measures the *boundary* of what the tradition can answer rather than
assuming coverage.

---

## 4. System architecture

```
        leaf photo                    soil photo              farmer-declared crop
             │                             │                           │
    ┌────────▼────────┐          ┌─────────▼────────┐                  │
    │ disease model   │          │ soil model       │                  │
    │ EfficientNet-B4 │          │ EfficientNet-B0  │                  │
    │ YOLO leaf crop  │          │ 3 heads:         │                  │
    │ → disease label │          │ type/moisture/   │                  │
    │ + Grad-CAM      │          │ texture          │                  │
    └────────┬────────┘          └─────────┬────────┘                  │
             └──────────────┬──────────────┴──────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │  INTEGRATION LAYER     │   ← the contribution
                │  Strategy B:           │
                │  "Apple Scab Leaf"     │
                │        ↓               │
                │  "dark rough scabby    │
                │   patches spreading    │
                │   over the leaf"       │
                └───────────┬────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │  HYBRID RETRIEVAL over IKS corpus  │
          │  dense bge-large top-20            │
          │  + BM25 top-20  → RRF (k=60)       │
          │  → bge-reranker cross-encoder      │
          │  → top-5 passages                  │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │  GROUNDED GENERATOR (Llama-3.1-8B) │
          │  • answer ONLY from passages       │
          │  • cite [Source, ch.X, v.Y]        │
          │  • REFUSE if insufficient          │
          └─────────────────┬──────────────────┘
                            │
                 cited advice + heatmap + source passages
```

Infrastructure decisions: all datasets and checkpoints on HuggingFace Hub (Colab sessions
expire; per-epoch checkpoint push/resume survives this). Embedder and reranker on CPU,
vision models and Llama on GPU — this specific split avoids the T4 out-of-memory failure
encountered in Phase 8.

---

## 5. The IKS corpus

### 5.1 Sources

| Source | Type | Passages | Role |
|---|---|---|---|
| Vrikshayurveda (Surapala) | classical | 52 | primary — disorders, causes, remedies |
| Brihat Samhita (Varahamihira) | classical | 46 | rainfall prediction, crop growth, tree treatment |
| Kashyapiya Krishisukti | classical | 39 | cultivation, soil, water, sowing |
| Upavanavinoda | classical | 15 | arbori-horticulture |
| Krishi Parashara | classical | 13 | seasonal agronomy, rain signs |
| NITI Aayog natural-farming manual | modern | 68 | insect-pest recipes (Neemastra etc.) |
| **Total** | | **233** | |

### 5.2 Pipeline

PDF → OCR (Gemini Flash, page-level, cached) → cleaning (running headers, folios, figure
captions, footnote URLs stripped) → chapter location by English heading scan → verse-level
chunking → metadata attachment → embedding → ChromaDB.

Chapter location required care: the Brihat Samhita PDF has a page-offset that *drifts* from
~8 pages at the front to ~45 at the back, so any page-index scheme is unsafe. Chapters are
located by scanning OCR text for English headings and Roman numerals instead.

### 5.3 Two corpus defects found and corrected

Both are reported because both materially affected earlier results.

**(a) OCR truncation.** An output-token cap silently truncated dense pages — a page of
twenty verses would yield two. Detected by manual spot-check against the printed PDF, not by
any automated test. Fixed by removing the cap, adding an explicit completeness instruction,
and adding a `finish_reason` truncation detector. Both affected books were re-OCR'd and
verified.

**(b) Chapter over-capture.** The chapter-location routine ended each *wanted* chapter at
the next *wanted* chapter, silently swallowing every chapter in between. Brihat Samhita
chapter XL ("Growth of Crops", 4 printed pages) had absorbed chapters XLI–LIII — commodity
price divination, "Signs of Swords", architecture, mythology — 167 pages in place of 4. The
corpus contained roughly 25% material never requested.

Fixed by ending a chapter at the *next detected heading of any kind*. Detection required
three guards, each earned from a real false positive: require the literal word "chapter"
(running headers are `Treatment of Trees LV 533`); require a separator and capitalised title
after the numeral (otherwise cross-references such as `[Cf. chapter IX]` truncate a chapter
mid-way — this was caught before rebuilding, by checking the affected pages against the
printed book); and require the heading at the top of the page.

Effect: Brihat Samhita 140 → 46 passages; corpus 327 → 233. Four regression tests added;
full suite 415 tests passing.

---

## 6. Disease recognition, and an interpretability diagnosis

### 6.1 The cascade and the audit

EfficientNet-B4 at 380×380, trained as a three-stage transfer cascade. A stage-by-stage
audit combined accuracy with Grad-CAM inspection:

| Stage | Dataset | Classes | Accuracy | Grad-CAM attention |
|---|---|---|---|---|
| 1 | PlantVillage (lab) | 38 | **99.8%** | leaf ✅ |
| 2 | Paddy Doctor (field canopy) | 10 | **97.0%** | lesion ✅ |
| 3 | PlantDoc (in-the-wild) | 27 | **72.3%** | **background ❌** |

The model was right for the wrong reason at stage 3. It attended to soil, sky and image
corners rather than to the lesion — and this persisted even on clean PlantVillage images.
72.3% is at the published frontier; the defect is *how* that number is obtained.

Methodological note: before blaming the model we tested three Grad-CAM target layers,
because an artefact identical across models and inputs is usually the visualisation. We were
partly right — `conv_head` exaggerates corners — and partly wrong: the bias persists at
`blocks[-2]`, so it is a real model property. Corner hotspots were separately traced to
noisy Grad-CAM variants and resolved with eigen-smoothing.

### 6.2 Three failed fixes (reported in full)

| Fix | Result | Why it failed |
|---|---|---|
| Background randomisation (Phase 5-R) | worse at every stage (PlantVillage 99.8→90.7, PlantDoc 72.3→66.8) | consistent with published findings; degrades features without teaching leaf focus |
| Linear-probe fine-tuning (freeze backbone) | 61% (−11pp) | the damage is in the backbone, not the head |
| Inference-time YOLO cropping | 58.2% (−14pp) | removes the background the model depends on, exposing its true leaf-only ability |

The third is the informative one: cropping at inference *lowers* accuracy by 14 points,
which is direct evidence of how much of the 72.3% was background-derived.

### 6.3 The fix that worked — C-PD

Retraining on GT-box leaf crops, warm-started from the healthy PlantVillage backbone:

- **66.6% top-1, 0.652 macro-F1** on 452 held-out leaf crops
- Grad-CAM attention moved **onto the leaf**
- Healthy-versus-diseased discrimination is near-perfect; the 27-class errors are
  predominantly within-crop *subtype* confusion (corn rust ↔ corn blight, potato early ↔
  late blight) rather than health-status errors

The honest framing: **66.6% earned from the leaf is worth more than 72.3% earned from the
background.** The two numbers are measured on different inputs and are not directly
comparable.

### 6.4 Crop-agnostic disease-type experiment (isolated)

A separate experiment pooled ~194 crop×disease labels into ~13 crop-agnostic disease
*types* (rust, blight, leaf_spot, …), aligning the vision side with the symptom-based
direction of the retrieval side.

| Run | Test accuracy | Macro F1 | Grad-CAM on field images |
|---|---|---|---|
| uncropped | **0.782** | — | background ❌ |
| YOLO leaf-crop | **0.719** | 0.635 | lesions ✅ |

The same trade-off as C-PD, and the same decision: keep the cropped model. Weak classes are
honestly reported — downy_mildew 0.000 (8 samples, too few to claim anything), early_blight
0.364 (early ↔ late confusion).

---

## 7. Soil recognition

EfficientNet-B0, multi-task with three heads:

| Head | Top-1 | Macro F1 |
|---|---|---|
| soil_type | 89.92% | 0.851 |
| moisture | 95.76% | 0.958 |
| texture | 67.86% | 0.678 |

Texture is the weak head and remains open.

---

## 8. The integration layer — the core contribution

### 8.1 Strategy A versus Strategy B

The question: how does a modern vision label reach a classical text?

**Strategy A (template).** Insert the label into a fixed template: *"treatment for Apple
Scab Leaf in apple"*. Retrieval score **0.01–0.04**. The corpus does not contain the phrase
"apple scab", so nothing matches.

**Strategy B (LLM-mediated rewrite).** Ask a language model to re-express the label in the
descriptive vocabulary the texts use: *"dark rough scabby patches spreading over the leaf"*.
Retrieval score **0.59–0.96**.

This ablation is the clearest evidence in the project. The bridge is not incidental
engineering — it is what makes the connection possible at all.

### 8.2 Symptom-driven realignment

An early version led queries with the crop name and refused when that name was absent from
the texts. Dr. Sunita T. Pandey identified this as a conceptual error: the texts are
symptom-general, not crop-indexed. Verified by literature review before any change was made.

Two corrections followed: Strategy B now enforces "lead with the symptom" and demotes the
crop to background context; and the grounded prompt gained a rule scoping refusal — a
passage addressing the *observed condition* is sufficient evidence **even if it never names
the crop**. Refusal applies only when no passage addresses the condition.

The faithfulness guardrail was unchanged; its *misapplication* was corrected. Before this,
the system was under-reporting genuine coverage — refusing cases the texts do cover, which
undercut the very bridge that constitutes the contribution.

### 8.3 Scope handling and calibrated confidence

The classifier has a fixed 27-class head and structurally cannot say "I don't know". Asking
it whether a plant is supported asks a question it cannot answer. The design instead has
**the farmer declare the crop and the model name the disease** — a clean division of labour,
since the person reliably knows their own crop.

For crops outside training, Dr. Pandey's refinement applies: disease appearance transfers
across species, so rather than refuse, the system runs the model, displays calibrated
confidence, advises through the symptom bridge with an explicit caution, and routes
low-confidence cases to expert review. Samples are collected for offline expert-validated
retraining — never live self-learning, which would poison the model.

---

## 9. Grounded generation and refusal

Llama-3.1-8B (4-bit) under a prompt that enforces three rules:

1. answer **only** from retrieved passages;
2. cite every claim as `[Source Text, ch.X, v.Y]`;
3. if the passages are insufficient, **say so and stop** — do not supplement from general
   knowledge.

Rule 3 is what the evaluation is designed to test. An ungrounded control (the same model
with no retrieved context) provides the fabrication baseline.

---

## 10. Evaluation methodology

**Query set.** 24 silver queries: 22 initially labelled answerable, 2 deliberate negatives
that nothing in the corpus addresses. Silver, not gold — authored by the project, pending
expert ratification. Labels are book-level, which supports Precision@k, nDCG, MRR and Hit@k,
but leaves Recall@k undefined until passage-level labels exist.

**Retrieval metrics.** Precision@5, nDCG@5, MRR, Hit@5, with three ablations: keyword-only
(BM25), dense-only, and hybrid without reranking.

**Generation metrics.**
- *grounded answer rate* — answered, with a citation that resolves to a retrieved passage
- *valid citation rate* — citations that resolve correctly
- *honest refusal* — refused on the negative controls (higher is better)
- *over-refusal* — refused on queries labelled answerable (lower is better — **see Section
  13, this metric turned out to be mislabelled**)
- *unfounded citations* — citations to passages not retrieved (fabrication)

---

## 11. Results — retrieval

Measured on the clean 233-passage corpus:

| Variant | P@5 | nDCG@5 | MRR | Hit@5 |
|---|---|---|---|---|
| **full** (hybrid + rerank) | 0.6727 | 0.8779 | 0.8561 | **1.0000** |
| dense only | 0.6636 | 0.8818 | — | 1.0000 |
| hybrid, no rerank | 0.5364 | 0.7250 | — | — |
| keyword only (BM25) | 0.3273 | 0.5151 | — | 0.6818 |

**The bridge is proven.** Full retrieval beats the keyword baseline decisively — nDCG 0.88
versus 0.52, Hit@5 1.00 versus 0.68. Because the corpus is classical translated prose and
the queries are modern symptom descriptions, semantic retrieval is not a refinement but a
requirement.

**Reranking contributes +0.14 P@5** over hybrid-without-rerank.

**An honest ablation surprise:** dense-only performs on par with, and on nDCG marginally
above, the full hybrid. BM25 injects lexically-matched but off-topic passages that the
reranker only partly removes. With n=22 and book-level labels this gap may be noise, but it
is reported rather than suppressed.

---

## 12. Results — generation, across three corpus versions

| Metric | 206 passages | 327 passages | **233 passages (clean)** |
|---|---|---|---|
| grounded answer rate | 13.6% | 13.64% | **13.64%** |
| valid citation rate | 14.2% | 41.7% | **55.00%** |
| over-refusal | 54.5% | 72.7% | **81.82%** |
| honest refusal (negatives) | 100% | 100% | **100%** |
| unfounded citations | 0% | 0% | **0%** |

RAGAS (gpt-4o-mini judge): faithfulness 0.559, answer relevancy 0.163. An independent
cross-check excluding refusals gives faithfulness **0.86** (n=10) and relevancy 0.48 — the
low headline relevancy is dominated by refusal strings, each scoring near zero, not by poor
answers.

**The observation that drove the rest of this work:** across three corpora of different
size, different composition, and with a 94-chapter defect fixed in between, the grounded
answer rate did not move — 13.64%, exactly 3 of 22, every time.

What *did* change was weakly-grounded answers: 7 → 3 → 1. As the corpus grew cleaner, the
generator stopped producing answers it could not support and refused instead. That single
mechanism explains both remaining trends: over-refusal rises because refusals replace weak
answers, and valid-citation rate rises because only well-supported answers survive. **The
system became more honest at every step.**

---

## 13. The central finding — a measured coverage boundary

If corpus expansion, displacement and a major corpus defect all fail to move the grounded
answer rate, the remaining explanation is that the texts do not contain the answers.

This was tested directly. For each of the 22 queries the pipeline retrieves its top-5, and
the **cross-encoder's own top-1 relevance score** is used as the measurement — it is a
trained relevance judge and requires no heuristic on top of it.

| Top-1 relevance | Queries | Which |
|---|---|---|
| **≥ 0.35** — genuine match | **4** | rain signs (0.72), boring insects (0.70), soil preparation (0.63), yellow/stunted leaves (0.58) |
| 0.15–0.35 — marginal | 5 | powdery mildew, mosaic virus, pepper leaf spot, apple rust, corn blight |
| **< 0.15** — no match | **13** | scab, Septoria, gray leaf spot, early/late blight, bacterial spot, leaf mould, black rot, corn rust, mites |

**Four genuine matches against a grounded answer rate frozen at three.** The generator
produces an answer almost exactly when the corpus contains one.

### Why the gap is structural

The passage retrieved rank-1 for 10 of 22 queries is Vrikshayurveda's **Table 1 —
disorder / cause / symptom / remedy**. This is real remedy content and the correct table to
retrieve. But it is indexed **by cause** — *vata*, *pitta*, *kapha*, fire, lightning, axe
wound, ants, faulty seed. The queries are indexed **by lesion appearance** — "numerous small
dark spots with pale centres". Table 1 has no such row, and none for scab, Septoria or
mildew. The reranker returns it because it is the most disease-like text available, then
scores it 0.10: *the best available, and not a match.*

### Consequence for the metric

`expect_answerable: true` on all 22 queries was an assumption, never verified against the
corpus. An 81.8% over-refusal rate was therefore measured against 13 queries that have no
answer. The system's actual behaviour is: **answers approximately 3 of the 4 it can, refuses
the rest, and fabricates nothing.** That is the desired behaviour of a grounded system, and
the metric was concealing it.

The query set has been re-labelled accordingly: the 13 are now marked unanswerable, with the
distinction preserved between *out-of-scope* (nothing in the corpus addresses the topic —
the 2 original controls) and *no-coverage* (the books are topically relevant but contain no
matching passage — these 13). The original file is retained unchanged for comparison.

### A secondary defect

Four citation labels are shared by 17 different passages — five distinct passages all cite
as `[Vrikshayurveda, ch. full, v. 1.2]`. A citation therefore cannot identify which passage
was used, which means the 55% valid-citation rate is scored against labels that are not
unique. This must be corrected before any citation figure is reported.

### Why this is a contribution

The honest framing of this project is **not** "a RAG system that answers disease queries
from Sanskrit texts" — the texts cannot answer most of them, and claiming otherwise would
require fabrication. It is:

> *A grounded multimodal system that maps modern vision-model disease labels onto classical
> IKS treatment knowledge, and that refuses rather than fabricates where the two traditions
> do not overlap — with that overlap measured, at 4 to 9 of 22 modern leaf-disease
> categories.*

The refusal behaviour is the safety result. The coverage gap is a quantified finding about
the limits of IKS digitisation, not a failure of the pipeline.

---

## 14. Negative results

Retained in full; each constrains what a future researcher should attempt.

1. **Background randomisation** degrades accuracy at every cascade stage without fixing
   attention.
2. **Linear-probe fine-tuning** loses 11 points — the damage is in the backbone.
3. **Inference-time cropping** loses 14 points, quantifying the background dependence.
4. **Corpus expansion can hurt.** Adding a topically-adjacent but non-matching source (an
   insect-pest manual, evaluated against fungal-disease queries) displaced classical
   passages from the top-5 and raised over-refusal by 18 points. *Corpus growth must be
   matched to the query distribution, not to volume.* The process failure is recorded too:
   the source was recommended on keyword counts without checking overlap against the
   specific diseases in the query set.
5. **A silver query set can encode a false assumption.** Thirteen of 22 queries were
   labelled answerable without verification, producing a metric that penalised correct
   behaviour for three evaluation cycles.

---

## 15. Limitations

- **Silver labels.** Expert ratification is pending; all evaluation numbers are preliminary.
- **Small evaluation set.** 22 queries; several per-class figures rest on very few examples.
- **Recall undefined.** Book-level labels only; passage-level labels are needed.
- **Citation granularity broken** (Section 13).
- **Corpus coverage** is the binding constraint on usefulness, not retrieval quality.
- **Weak texture head** (67.86%) on the soil model.
- **English only.** The corpus is translated; Sanskrit primary sources are not processed.
- **Vishvavallabha not yet ingested** — the classical text that treats plant disease
  directly, planned from the outset, still unobtained.

---

## 16. Future work

**Immediate, in order:**

1. **Ingest Vishvavallabha.** It is the classical text covering plant disease and pest
   management, and it is now the only credible route to raising the grounded answer rate.
2. **Extend the query set into the corpus's actual strengths** — pest damage, soil
   preparation, sowing season, rainfall prediction. The four strong matches show what this
   corpus genuinely supports; the current query set tests almost exclusively what it does
   not.
3. **Repair citation granularity** — real verse ranges in place of `ch.full v.1.2`.
4. **Expert gold query set** with passage-level labels, enabling Recall@k.

**Further:**

5. Held-out-crop evaluation of the disease-type model — train excluding one crop, test on
   it — as direct evidence for cross-plant generalisation.
6. Expert validation of generated recommendations by agronomists.
7. Indian-language extension.

---

## 17. Conclusion

The system works end to end: a leaf photograph and a soil photograph produce a disease
diagnosis, a soil assessment, a visual explanation, and advice that is either cited to a
specific classical passage or honestly refused. The symptom-based bridge that makes the
connection possible is validated by ablation — 0.59–0.96 against 0.01–0.04 — and the
faithfulness guarantee holds without exception: zero fabricated citations across every
evaluation run.

The most valuable result was not planned. A metric that appeared to show failure — 81.8%
over-refusal — turned out to be measuring the system against questions the source tradition
cannot answer. Establishing that required ruling out corpus size, passage displacement and a
major corpus defect in turn, and then measuring the texts directly. The outcome is a
quantified boundary between classical Indian plant-disease aetiology and the modern lesion-
based taxonomy: roughly 4 to 9 of 22 modern leaf-disease categories have genuine support.

That boundary is worth stating plainly, because the alternative — a system that produces a
confident classical-sounding remedy for Septoria leaf spot — would be worse than useless. It
would be a fabrication presented as heritage.

---

## Appendix A — Reproducibility

- **Code:** `src/` (disease, soil, rag, integration, explain), `scripts/`, `tests/` — 415
  tests passing.
- **Ledger:** `EXPERIMENT_LOG.md` — every experiment with date, hypothesis, method, numbers
  and verdict, including failures.
- **Artifacts:** datasets and checkpoints on HuggingFace Hub; corpus chunks with `chunk_id`,
  book, chapter, verse, translator, source tier.
- **Key scripts:** `scripts/check_coverage.py` (the coverage measurement),
  `scripts/check_displacement.py`, `src/rag/corpus/chapter_split.py` (with its four
  regression tests).

## Appendix B — Headline numbers

| Quantity | Value |
|---|---|
| Corpus | 233 passages, 6 sources |
| Disease cascade (PlantDoc, background-driven) | 72.3% |
| Disease C-PD (leaf-crop, honest attention) | 66.6%, macro-F1 0.652 |
| Disease-type, crop-agnostic (cropped) | 71.9%, macro-F1 0.635 |
| Soil: type / moisture / texture | 89.92% / 95.76% / 67.86% |
| Strategy B versus Strategy A retrieval | 0.59–0.96 vs 0.01–0.04 |
| Retrieval nDCG@5 (full vs keyword) | 0.878 vs 0.515 |
| Hit@5 | 1.00 |
| Honest refusal on negatives | 100% |
| Unfounded citations | 0% |
| Queries with genuine textual support | 4 of 22 strong, 9 of 22 including marginal |
