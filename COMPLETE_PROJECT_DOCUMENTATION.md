# Complete Project Documentation
## An IKS-Grounded Multimodal Agricultural Advisory System

**M.Tech Thesis · IIITDM Jabalpur**<br>
**Author:** Ankit Pawar (M.Tech, Computer Science & Engineering)<br>
**Supervisor:** Dr. Akshay Pandey, Dept. of Computer Science & Engineering<br>
**Full title:** *An IKS-Grounded Multimodal Agricultural Advisory System — Joint Disease and Soil Analysis with Retrieval-Augmented Generation over Classical Indian Agricultural Texts*


---

## 1. The system in brief

The project is an agricultural advisory system that grounds modern AI in the **classical Indian agricultural texts (the Indian Knowledge System, IKS)**. A farmer provides a photograph of an affected plant and, where available, a photograph of the soil. The system identifies the plant's condition and the basic soil characteristics, then retrieves the corresponding **traditional, organic treatment described in the classical texts** — presenting it with the original source cited, or **honestly declining when the texts do not cover the case.**

The guiding principle throughout: **the Indian Knowledge System provides the insight; the technology only makes that insight accessible, verifiable, and scalable.**

A single request flows as follows:

> **Leaf photo + Soil photo + (optional) suspected cause** → the leaf is detected and cropped from its background → the **disease model** names the specific disease (or reports "healthy") → the **soil model** reads soil type, moisture, and texture → if the leaf is healthy the system stops with "no treatment needed"; if it is diseased, the modern vision labels are **translated into the descriptive vocabulary the classical texts use** (the core novelty) → a **hybrid retrieval** step searches the IKS corpus → a **grounded language model** composes a cited recommendation, or an honest refusal → a **visual heatmap** shows the leaf region the model used.

---

## 2. Motivation and guiding principles

**Motivation.** The classical Indian agricultural treatises contain a large, organic, time-tested body of plant-care and soil knowledge — but it is locked in Sanskrit translations and indexed by *symptom description*, not by modern disease names. Modern vision models, conversely, output modern labels ("Apple Scab", "rice blast") that do not appear anywhere in those texts. The central research problem is to **bridge** the two so the classical knowledge becomes searchable and usable.

**Guiding principles (enforced in the system, not merely intended):**

1. **The soil module is visual-only.** It is forbidden from emitting NPK / pH / fertility / chemical claims — it cannot determine those from a photograph, so it must not pretend to.
2. **Per-class metrics, never just accuracy.** Every vision report carries per-class precision/recall/F1 and a confusion matrix.
3. **Cross-region validation for soil.** Soil performance is checked on held-out regions, not only a random split.
4. **No fabricated citations.** The language model must cite its sources, and a verification step confirms the cited passages were actually retrieved. When the corpus does not support an answer, the system refuses rather than inventing one.

These principles are what make the system **defensible**: it is explicit about what it can and cannot know.

---

## 3. Research contributions (the intended novelty)

- **C1 — The corpus.** A first chunked, metadata-tagged, searchable digital corpus of these specific classical texts.
- **C2 — Joint disease–soil context module** with three *measured* integration strategies (template / language-model-mediated / multimodal-embedding), so the contribution is demonstrated by ablation, not merely asserted.
- **C3 — Faithfulness-aware evaluation** combining automatic metrics with domain-expert judgement.
- **C4 — A quantitative measurement of hallucination** on IKS sources.
- **C5 — Cause-conditional retrieval** — the system retrieves a treatment *given* a user-supplied causal context; it deliberately does **not** infer the cause from images.

The contribution most fully demonstrated to date is **C2's language-model-mediated bridge** (Module 5), which raises retrieval relevance from roughly 0.01–0.04 (a plain template) to **0.59–0.96**.

---

## 4. Module overview

| Module | Scope | Status |
|---|---|---|
| 1 | Corpus of classical texts | ✅ Complete (4 of 6 texts) |
| 2 | Plant-disease recognition | ✅ Complete (re-trained for correct focus) |
| 3 | Soil-property recognition | ✅ Complete |
| 4 | Knowledge retrieval system | ✅ Complete |
| 5 | Multimodal integration (core novelty) | ✅ Complete |
| 6 | Explainability layer | ✅ Complete |
| 7 | Full system & user interface | ✅ Base built; additions pending |
| 8 | Rigorous evaluation | ⏳ Pending |

Each module is described once, in full — what was built, the results, the experiments and reasoning behind them, the refinements made, and what (if anything) remains for that module.

---

## 5. The work, module by module

### Module 1 — Corpus of classical texts

**Built.** A digital, searchable corpus was assembled from four texts — **Vrikshayurveda, Brihat Samhita, Krishi Parashara, and Upavanavinoda** — producing a structured collection of verse/passage units, each carrying its full source details (text, chapter, verse).

*How it was built, and the reasoning.* Each scanned text was passed through optical character recognition (OCR), the Sanskrit/Devanagari lines were cleaned away to leave the English translation, chapters were located by their headings, and the text was split into verse-first passage units. Each unit is given a stable, content-derived identifier so the corpus can be rebuilt repeatedly without ever creating duplicates. A free, local OCR engine was used first; for two of the texts the Devanagari script was confusing that engine and degrading quality, so a higher-grade OCR was adopted for those two specifically (used sparingly, on a tight budget). The cleaned passages are embedded into a searchable vector store. The final corpus is **206 passage-units across the four books.** No copyrighted text is exposed in any public location.

**Pending.** Two further texts of the same tradition — **Kashyapiyakrishisukti** and **Vishvavallabha** — remain to be added once obtained (see §6, Collaboration).

---

### Module 2 — Plant-disease recognition

This module was the largest single effort in the project, because the first working model had a subtle but serious flaw that had to be diagnosed and corrected. The full account follows.

**What was built first.** A disease-recognition model (EfficientNet-B4) was trained in three transfer-learning stages, each building on the last:

| Stage | Data | Test accuracy | Where the model focused |
|---|---|---|---|
| 1 | 38-class lab-condition images | **99.8%** | The leaf ✅ |
| 2 | 10-class Indian rice images | **97.0%** | The lesion ✅ |
| 3 | 27-class real-field images | **72.3%** | The background ❌ |

Accuracy on the lab and rice data was excellent, and the real-field figure (72.3%) matched the level of current published results for that difficult dataset.

**The flaw — and how it was confirmed.** A review of the explainability heatmaps revealed that, on the hardest real-field stage, the model was reaching its answers by attending to the **image background** rather than the leaf — exploiting incidental cues (soil colour, field context, photo provenance) that happen to correlate with the disease, instead of reading the lesion itself. On the real-field test set, only about **3 of 256** images had the model's attention actually on the leaf. Before attempting any fix, two things were checked:
1. **Was it just a heatmap artifact?** The effect was tested across several internal layers of the network. It persisted at the meaningful layer, so the bias was **real, not a visualization quirk.** (A separate, genuine cosmetic artifact at the very last layer was cleaned up later with a smoothing option.)
2. **Where, exactly, does it enter?** Running accuracy and heatmaps stage by stage localized the fault precisely: **stages 1 and 2 were healthy** (correct leaf/lesion focus); the bias was introduced **only at the third, real-field stage**, where fully re-training the network on a small (~2,000 images), cluttered, in-the-wild dataset taught it a background shortcut.

**Three fixes were tried, and rejected on evidence.** The approaches were deliberately ordered cheapest-and-most-likely-first, escalating only as each failed:

| Approach tried | What it did | Result | Why it was rejected |
|---|---|---|---|
| **Background randomization** | Re-train while pasting leaves onto random backgrounds each epoch, and add a "no-leaf" reject class, so background can no longer correlate with the disease | Accuracy fell at every stage (lab 99.8 → 90.7%, real-field 72.3 → 66.8%); the gain in leaf-focus was marginal | It cost real accuracy for almost no improvement in focus — a poor trade. (Consistent with published findings that background randomization often hurts in-distribution accuracy.) |
| **Freeze-and-retrain-head (LP-FT)** | Keep the existing learned features frozen and re-train only a small new classifier on top — a fast, low-risk fix that matched the diagnosis (preserve the good early-stage features) | Accuracy dropped to **61%** (an 11-point fall) with **no improvement in focus** | The features being frozen were the **rice-specialised** ones from stage 2, which are wrong for a 27-class multi-crop problem; and, fundamentally, a frozen backbone with only a new head **cannot change where the model looks.** |
| **Detect-then-crop at inference** | Use a separate detector to find and crop the leaf at prediction time, then classify the crop (removing the background without retraining) | Held-out accuracy **fell to 58.2%** (from 72.3%). An apparent **91.9%** seen earlier was traced to a **data-overlap error** — the detector's test images were inside the classifier's training set — and was correctly discarded | The accuracy *drop* when the background was removed actually **proved the model was leaning on the background.** Cropping at prediction time alone does not undo a dependence baked in during training. |

**The fix that worked — re-training on the leaf itself.** The model was **re-trained on the leaf region cropped out of the background**, and — importantly — **warm-started from the clean lab-stage features rather than the rice-stage ones**, so the starting point already knew how to look at leaves. With no background left in the training images, the model was *forced* to learn the leaf. The reasoning behind these two choices:
- *Crop the training data* (not just inference): remove the shortcut at its source, during learning.
- *Start from the lab features, not the rice features*: the rice features were narrow and were part of how the flaw arose; the lab features had correct leaf focus to build on.

**Result.** Focus is now correctly on the leaf, confirmed in the heatmaps. Accuracy on the hardest set settled at about **67%** (measured on leaf crops). This was accepted on a clear, defensible principle: **a correctly-focused model at ~67% is preferable to a higher but mis-focused one at 72%.** (The two figures are measured on different test sets — leaf crops versus full images — so they are not directly comparable.)

**A reassuring and important finding.** The model's **healthy-versus-diseased judgement is near-perfect.** Almost all of the remaining error is *fine confusion between disease sub-types of the same crop* (for example, two different corn diseases), not confusion about whether the plant is healthy — so the model is reliable on the decision that matters most.

**Real-world leaf localization.** Because real photographs contain substantial background, the deployed pipeline now **detects and isolates the leaf before analysis.** A further experiment that trains with randomized backgrounds (so the model cannot rely on background cues at all) has been prepared and set up; it preserves the current model so the system can revert if the experiment does not help.

---

### Module 3 — Soil-property recognition

**What was built.** A multi-task model (EfficientNet-B0) predicts three properties from a single soil image — **soil type, moisture appearance, and texture** — using a training scheme where each image supervises only the property it actually has a label for.

| Property | Test accuracy |
|---|---|
| Soil type | **89.9%** |
| Moisture appearance | **95.8%** |
| Texture | **67.9%** |

*The experiments behind these numbers.* A baseline version reached solid soil-type and moisture accuracy but a weak texture result. A second version added strong image augmentation, mixing techniques, and test-time averaging; this **lifted moisture markedly** (to 95.8%) and slightly improved soil type, and became the production model. Texture, however, did not improve — and two further attempts to lift it were tried and reported honestly: a **staged training curriculum** (train one property, then the next, then all together), which **destabilized and collapsed** all three heads and was abandoned; and a **patch-tiling expansion** that cut texture images into smaller pieces to enlarge the training set, which did **not** improve the result.

**Honest limitation.** Texture is genuinely hard, limited by the scarcity of labelled close-up texture images. It was therefore shipped as an **auxiliary** output and documented as future work, rather than overstated.

---

### Module 4 — Knowledge retrieval system

**What was built.** A retrieval-augmented system that searches the corpus and composes a grounded, source-cited answer using a language model, and that **openly declines when the corpus does not cover a query** rather than inventing an answer.

*The design and the reasoning.* Retrieval is a robust two-stage process: first a broad search that combines **semantic similarity** (meaning-based) with **keyword matching** (so distinctive terms and transliterated Sanskrit words are not missed), and then a **precision re-ranking** step that re-scores the candidates and keeps only the few most relevant passages. This two-stage design was chosen because either method alone misses cases the other catches. The language model is used in a strictly *grounded* mode: it answers **only** from the retrieved passages, must cite them, and refuses when the evidence is insufficient. The model is used as-is (not re-trained) — all of its domain knowledge comes from the retrieved classical passages, never from the model's own memory, which is what keeps the answers faithful to the texts.

**Scope and honest refusal.** It was confirmed that the corpus is oriented towards **trees and fruit-bearing plants** (reflecting the texts themselves). For such crops the system gives a real, cited treatment; for crops the texts do not cover (for example, certain grains) it **honestly states that the classical texts do not contain the information.** This honest refusal is treated as a feature of the system — a sign of integrity, not a failure — and is demonstrated deliberately (a grain such as corn produces an honest refusal, while a tree disease such as apple scab produces a real cited treatment).

---

### Module 5 — Multimodal integration (the core novelty)

**What was built.** The vision outputs (the plant's condition, the soil properties, the crop) are turned into a search query for the corpus. The key contribution lives here: because the classical texts **do not use modern disease names**, the system **bridges the modern terminology to the descriptive, symptomatic vocabulary the texts actually use** — translating a modern disease label into the characteristic visual symptoms a classical text would describe. This substantially improves the relevance of what is retrieved.

*Why this was measured, not assumed.* Three query-construction strategies were built and compared so the benefit of the bridge could be proven by ablation:
- a **plain template** (deterministic fill-in) — the baseline;
- the **language-model-mediated bridge** — the contribution;
- a **multimodal-embedding** variant — an honest ablation exposing where it falls short.

The bridge raised retrieval relevance to **0.59–0.96**, against **0.01–0.04** for the plain template — a decisive margin that establishes the contribution.

*Two refinements made during development to keep the bridge faithful.*
- The crop used in the query is now **taken from the recognized disease itself** (the disease label already names the crop), rather than from a separate menu that could be left on a stale value and describe the wrong crop. A manual override remains for the user, since a human who knows the crop should be able to correct the model.
- The query-construction instructions were corrected so the system describes the **symptoms of the specific recognized disease** — different diseases now produce genuinely different, accurate queries, instead of collapsing onto a single generic description.

**Cause-conditional retrieval (C5).** A user may optionally supply a suspected cause (soil-related, pest-related, spreading from neighbours, or unspecified); this is woven into the query. The cause is a user input only — it is never guessed from the image, by design.

---

### Module 6 — Explainability

**What was built.** The system can show **why** it reached a conclusion: a visual heatmap highlighting the leaf region the model focused on, and the specific retrieved text passages with the matching terms marked. This module is also what surfaced the disease-focus finding described in Module 2 — explainability was not an afterthought but the very tool that caught the flaw.

*Two refinements keep the explanations meaningful and honest.*
- **The heatmap is shown only for a diseased leaf**, where there is a lesion to highlight. On a healthy leaf there is nothing to localize, so a heatmap there would be diffuse and misleading; it is therefore suppressed.
- **A treatment advisory is generated only for a diseased leaf.** For a healthy leaf, the system simply confirms that no treatment is needed, rather than producing a meaningless "treatment for a healthy plant."

---

### Module 7 — Full system and user interface

**What was built.** The individual components were wired into a single working application: upload a plant image (and a soil image), and receive the condition, the soil assessment, the visual explanation, and the grounded, cited recommendation. A base version of the interface is functioning and has been demonstrated live.

**Pending — interface additions (designed, not yet built):**
- **Two ways to use the system.** *Option one* — both a plant image and a soil image (the full, real-field case). *Option two* — a plant image only (a casual check), in which the soil values are auto-filled to the baseline conditions suitable for the selected crop, with the result clearly marked as based on typical conditions rather than a measured soil sample.
- **A "same-location" safeguard.** In the full case, the interface will require that the plant photo and the soil photo come from the same place, so a mismatched pair cannot produce a wrong result and an unfair impression that the system has failed.
- **A crop-selection menu.** The user selects the crop from a fixed list — and that list is the **honest boundary** of the system, containing only the crops the model can reliably recognize.
- **Crop–soil suitability guidance.** Using the crop–soil study (§6), if the selected crop does not suit the detected soil, the system can advise on the mismatch, along with any relevant remedy found in the texts.

> The plain interface elements can be built now; the suitability-dependent parts depend on the crop–soil study, which is in progress.

---

### Module 8 — Rigorous evaluation (the key remaining research step)

**Pending.** This is the most important work remaining for publication: assembling an **expert-checked set of test queries**, measuring the system's retrieval and answer quality against accepted metrics, formally testing the effect of the causal-context option (the C5 ablation), measuring how faithful the generated answers are to the sources (C3, C4), and obtaining **domain-expert assessment** of whether the recommendations are agronomically sound. This step produces the quantitative results for the thesis and the paper, and is where collaboration with an agricultural expert is most valuable. Further comparative studies, the paper, and the thesis writing follow from it.

---

## 6. Supporting study and collaboration

**Crop–soil suitability study (in progress).** A dedicated reference is being prepared: a list of the crops and soils that are generally grown, together with the conditions each crop requires — which soil types it suits and the minimum suitable conditions for it (mapped to soil type, moisture, and texture). A first draft **covering 120 crops** has been compiled so far and is being extended. It is intended as a general reference usable across the group's work, to be validated against authoritative sources, and it directly underpins the interface additions noted in Module 7 (the baseline-soil defaults for the plant-only option, and the crop–soil mismatch guidance).

**Collaboration.** A collaboration has been initiated with **Dr. Sunita T. Pandey** (Agronomy, GBPUAT Pantnagar), a specialist in Vrikshayurveda-based practices and natural farming. She has shared her research for our study and, in her capacity at the **Asian Agri-History Foundation**, has been requested to help provide the classical texts needed to complete and strengthen the corpus. Her expertise is expected to support the agronomic validation and the IKS grounding of the system.

---

## 7. Status summary

| Done | Pending |
|---|---|
| Corpus (4 texts); disease recognition (including the full diagnosis and the re-training that corrected the model's focus, plus leaf localization); soil recognition; grounded retrieval with honest refusal; the multimodal integration novelty (the modern-to-classical query bridge, proven by ablation); the explainability layer; and the base interface | Interface additions (two usage options + safeguards + crop menu + suitability guidance); the crop–soil suitability study; the remaining two texts (Kashyapiyakrishisukti, Vishvavallabha); the rigorous evaluation and expert assessment; comparative studies; and the paper and thesis writing |

---

## 8. Headline results at a glance

| Component | Result |
|---|---|
| Disease — lab / rice / real-field stages | 99.8% / 97.0% / 72.3% |
| Disease — after re-training for correct leaf focus | ~67%, focus confirmed on the leaf; healthy-vs-diseased near-perfect |
| Soil — type / moisture / texture | 89.9% / 95.8% / 67.9% |
| Retrieval — language-model bridge vs plain template | 0.59–0.96 vs 0.01–0.04 |
| Corpus | 206 passage-units across 4 classical texts |


