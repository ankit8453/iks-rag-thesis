# Progress Update

*IKS-Grounded Multimodal Agricultural Advisory System*
*M.Tech Thesis · IIITDM Jabalpur · Ankit Pawar · Supervisor: Dr. Akshay Pandey*
*Date: 1 September 2026*

> A short update covering the work completed since the last meeting. The focus of this period was **expanding and repairing the classical-text corpus** — the limitation identified in the evaluation — and adding the Kashyapiya text.

---

## 1. Headline

The corpus was expanded and repaired: it now holds **259 passages across five books**, up from **206 across four** — an increase of **+53 passages (+26%)**. This includes the newly added **Kashyapiya Krishisukti** and the recovery of text that earlier optical-character-recognition (OCR) had missed.

| | Before | After |
|---|---:|---:|
| Books | 4 | **5** |
| Passages (chunks) | 206 | **259** |

---

## 2. What was done

### 2.1 Added the Kashyapiya Krishisukti (new fifth book)

The 64-page Sanskrit-only scanned text was processed end to end:

- **OCR + translation** of all 64 pages using Google Gemini (Flash 3.6) — the pages are Devanagari Sanskrit with no existing English translation, so each verse was both transcribed and translated. **857 verses** were captured, and the whole book was verified complete (from the opening to the closing colophon).
- The English translation was chunked into **39 passages** and added to the corpus in the same format as the other books.

**Honest assessment of this book.** Verified both against the scholarly edition (Asian Agri-History Foundation, Sadhale & Nene) and by direct testing of the new passages: the Kashyapiya is a **cultivation** treatise — strong on soil selection, water management, seeds, and sowing, but **light on plant-disease treatment**. It therefore strengthens the corpus for cultivation and soil/water queries, but is not expected on its own to close the disease-advice gap. For that gap, **Vishvavallabha** (Chakrapani Mishra, which explicitly covers plant disease and pest management) is the more valuable text to obtain next.

### 2.2 Repaired the existing books (re-OCR)

While preparing the Kashyapiya, a defect was found in the original OCR of the existing books: some pages had been **cut short** or **saved as an error message**, because the earlier OCR had an output-length limit. All existing books were re-processed with the newer model and no length limit, one book at a time, with each result verified before it was accepted:

- **Vrikshayurveda:** 15 pages recovered — 9 that had been dropped entirely (including the Introduction) and 6 that had been cut short. Its passage count rose from **42 to 52**.
- **Brihat Samhita:** one page that had saved an error message, plus 18 truncated pages, were restored. Count went from **136 to 140**.
- **Krishi Parashara and Upavanavinoda:** reviewed and left unchanged (already complete).

Every change was compared against the previous version before acceptance, with **no loss of existing content**.

### 2.3 Rebuilt the search index

The full corpus was re-embedded and rebuilt into the vector database, and verified to hold exactly **259 passages** across the five books.

---

## 3. Corpus now

| Book | Passages | Focus |
|---|---:|---|
| Brihat Samhita | 140 | rainfall, water divining, crops, tree treatment |
| Vrikshayurveda | 52 | plant care, horticulture, pest control |
| Kashyapiya Krishisukti *(new)* | 39 | soil, water, seed, sowing (cultivation) |
| Upavanavinoda | 15 | arboriculture, plant protection |
| Krishi Parashara | 13 | sowing, seed preservation, calendar |
| **Total** | **259** | |

---

## 4. In progress

- **Re-running the evaluation** on the expanded corpus, to measure whether the larger corpus reduces the earlier ~55% over-refusal rate (the system correctly declining answerable questions because the texts did not cover them). The evaluation is being re-run now; the comparison of before-vs-after numbers will follow.

---

## 5. Points to discuss

1. **Vishvavallabha as the next text.** The evidence points to it as the highest-value addition for the disease-treatment gap (it is the same translator family as texts already in the corpus). Is it available to obtain, and is that the right priority?
2. **Kashyapiya's role.** Given it is cultivation-focused rather than disease-focused, it is confirmed useful for soil/water/sowing advice. Is that an acceptable contribution, or should the focus stay strictly on disease-oriented texts?
3. **The sixth text.** One further text slot remains open in the plan — any recommendation on which to add?

---

*Prepared for the progress meeting on 1 September 2026. All corpus figures are drawn directly from the rebuilt index; the updated evaluation numbers are pending the current re-run.*
