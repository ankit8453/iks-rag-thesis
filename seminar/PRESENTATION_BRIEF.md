# Presentation Brief — ICSSR International Seminar (5-minute online talk)

> Single source of truth for building the slide deck. Paste this whole file into any
> AI slide tool (Gamma, Claude Design, Gemini/Google Slides, GPT) — or hand it to a
> designer. Every number here is verified from the project's experiment records.

---

## 1. The event

| | |
|---|---|
| Seminar | International Seminar on **Revitalising Indigenous Knowledge Systems: Tribal Heritage, Cultural Continuity and Social Transformation in India** |
| Organiser | Centre for Tribal Studies, Iswar Saran Degree College (University of Allahabad), Prayagraj · sponsored by **ICSSR**, Ministry of Education |
| Dates / slot | 11–12 September 2026 · online session **7:00–9:00 PM** · **5 minutes per presenter** (+ possible 1–2 min questions) |
| Sub-theme | **Digital Preservation, Artificial Intelligence, and Documentation of Indigenous Knowledge** |
| Audience | Sociologists, IKS scholars, heritage researchers, policy people. **Not** computer scientists. Plain language; zero acronyms unless explained; no architecture jargon. |
| Outcome | Selected papers → peer-reviewed edited volume with ISBN |

## 2. Presenter

- **Ankit Pawar** — M.Tech (Computer Science & Engineering), PDPM Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Jabalpur
- **Dr. Akshay Pandey** — Supervisor / co-author, Dept. of CSE, PDPM IIITDM Jabalpur

## 3. Paper title (as submitted)

**An IKS-Grounded Multimodal Advisory System: Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts**

## 4. The one-sentence pitch (say this in the first 15 seconds)

> India's classical agricultural texts hold centuries of organic plant-care knowledge that no farmer can actually consult today — we built a system where a farmer photographs a sick leaf and receives advice drawn *only* from those texts, cited to the exact verse, and honest enough to say "the texts don't cover this."

## 5. The story arc (why → what → how → proof → so what)

1. **Heritage that can't be used.** Vrikshayurveda, Brihat Samhita, Krishi Parashara, Upavanavinoda, Kashyapiya Krishisukti, Vishvavallabha: sustainable, organic plant-care and soil knowledge. Locked in Sanskrit and rare translations. And organised by *observed symptom*, not by the disease names modern diagnosis uses. So it is documented but **dead on the shelf**.
2. **The idea.** Make the texts *consultable from a photograph* — bridge modern diagnosis to classical prescription.
3. **The key insight (our contribution).** The texts prescribe **by symptom, not by crop**. So instead of asking "what does Vrikshayurveda say about apple scab?" (nothing — the word doesn't exist there), the system **re-describes the diagnosis as the symptoms the texts describe** — "dark scabby lesions spreading on the leaf" — and *then* searches. This single step raised retrieval relevance from near-zero (0.01–0.04) to 0.59–0.96.
4. **Trust by design.** Three guarantees: (a) every recommendation is composed *only* from retrieved passages and **cites text, chapter and verse**; (b) if the texts are silent, the system **refuses rather than invents** — 100% honest refusal on unanswerable queries, **0% fabricated citations**; (c) the image model is **verified to look at the lesion, not the background** — the explainability heat-map is a check, not decoration (it actually caught and fixed a flaw during development).
5. **What exists today.** A working prototype: leaf photo + soil photo → disease + soil condition → symptom bridge → cited advice → visual explanation. A searchable corpus of **327 verse-level passages** across the classical texts (Vishvavallabha next). Advises cautiously even on crops it was never trained on, with calibrated confidence. Uncertain cases go to expert review — human-in-the-loop, never silent self-learning.
6. **Why it matters here.** Documentation and *use* reinforce each other. A text that is cited to the verse every day is preserved *and* alive. This is a template for AI on Indigenous knowledge that transmits without distorting.
7. **Next.** Expert (agronomist) validation of recommendations; add Vishvavallabha (plant disease & pest management); expand the query set with domain experts.

## 6. Slide-by-slide (8 slides, ~5:00)

Timing assumes ~140 spoken words/min. Speaker notes are what to *say*; slide text is what to *show* (keep it to ≤ 25 words per slide).

### Slide 1 — Title (0:00–0:15)
- **Show:** Title · Ankit Pawar, Dr. Akshay Pandey · PDPM IIITDM Jabalpur · seminar name + sub-theme · a Devanagari verse fragment as a design accent (e.g. वृक्षायुर्वेद)
- **Say:** "Good evening. I'm Ankit Pawar from IIITDM Jabalpur. In five minutes I'll show you how we made India's classical agricultural texts answerable from a photograph — without letting AI put words in their mouth."

### Slide 2 — Heritage on the shelf (0:15–1:00)
- **Show:** six text names as a row of "book spines"; one line: *"Centuries of organic plant-care knowledge — unreachable from the field."* Three locks: **Sanskrit** · **scattered editions** · **organised by symptom, not disease name**
- **Say:** "These six treatises describe organic remedies, soil selection, seed care. But a farmer or extension worker can't consult them: they're in Sanskrit or rare translations, and crucially they don't use disease names. Vrikshayurveda never says 'apple scab'. It says 'dark scabby patches spreading on the leaf'. So the knowledge is documented — and dead on the shelf."

### Slide 3 — The idea in one picture (1:00–1:45)
- **Show:** a 4-step horizontal flow: **📷 leaf + soil photo → 🔍 diagnosis → 🌿 re-described as the texts' symptoms → 📜 advice cited to the verse**. Use `demo_images/leaf_apple_scab.jpg` and `demo_images/soil_alluvial.jpg` as the inputs.
- **Say:** "A farmer photographs the affected plant and the soil. The system identifies the condition and the soil's visible properties. Then — the key step — it translates that diagnosis into the language the texts actually use, searches them, and composes a recommendation drawn only from what it found, citing text, chapter and verse."

### Slide 4 — The key insight: speak the texts' language (1:45–2:30)
- **Show:** two columns. Left: *"Apple scab"* → texts return **nothing** (relevance 0.01–0.04). Right: *"dark, rough, scabby patches spreading over the leaf"* → **0.59–0.96**. One line: *"The texts prescribe by symptom, not by crop."*
- **Say:** "This is the contribution. Modern labels don't exist in classical texts, so direct search fails almost completely. Re-expressing the diagnosis as symptoms — the way the texts themselves reason — lifts retrieval from near zero to strong matches. It also means the system can help with a crop it was never trained on, because the symptom carries across plants — and it says so, with a confidence figure and a caution."

### Slide 5 — Trust by design (2:30–3:20)
- **Show:** three cards. **Cites the verse** (example citation line: *[Vrikshayurveda, ch. 3, v. 12]*) · **Refuses when the texts are silent** (100% honest refusal · 0% fabricated citations) · **Looks at the lesion, not the background** (heat-map before/after if available)
- **Say:** "Generative AI's danger for indigenous knowledge is confident fabrication. So we constrain it: it may only speak from retrieved passages, and must cite them. If the texts don't cover a case, it refuses — in evaluation, 100% honest refusal and zero fabricated citations. And the image model is verified visually: early on the heat-map showed it reading the *background* of field photos; we retrained until it reads the lesion. Explainability as a check, not decoration."

### Slide 6 — What exists today (3:20–4:00)
- **Show:** four stat tiles: **327** verse-level passages · **5** classical texts digitised (Vishvavallabha next) · **0%** fabricated citations · **working prototype** (screenshot of the app result page if available). Small line: *uncertain cases → expert review*
- **Say:** "This isn't a proposal. The corpus exists — 327 searchable, citable passages. The prototype runs end to end. When the system is unsure, the case is collected for an agricultural expert to review, so it improves under human oversight rather than learning from unverified guesses."

### Slide 7 — Why this matters for revitalising IKS (4:00–4:40)
- **Show:** one big line: *"A text cited every day is preserved — and alive."* Sub-points: documentation ↔ use · verifiable, not distorted · farmer's hands, not only archives
- **Say:** "For this seminar's theme: preservation and use are not separate goals. A verse that is retrieved and cited in a farmer's advisory is being preserved, transmitted and validated at once. And because every answer traces to its source, the AI transmits the tradition without distorting it. We think this is a template for Indigenous knowledge beyond agriculture."

### Slide 8 — Next steps + thank you (4:40–5:00)
- **Show:** three next steps: expert validation with agronomists · add Vishvavallabha (disease & pest management) · expand evaluation with domain experts. Contact line. "Thank you."
- **Say:** "Next: validation with agronomists, adding Vishvavallabha which specifically covers plant disease, and an expert-built evaluation set. Thank you — happy to take questions."

## 7. Numbers you MAY show (verified) — and how to phrase them

| Claim | Number | Phrase it as |
|---|---|---|
| Symptom bridge vs direct label search | 0.01–0.04 → **0.59–0.96** | "retrieval relevance from near zero to strong" |
| Semantic search vs keyword search | nDCG **0.94 vs 0.70** | "meaning-based search clearly beats keyword search on classical text" |
| Honest refusal on unanswerable queries | **100%** | "never bluffs" |
| Fabricated citations | **0%** | "never invents a source" |
| Corpus | **327** passages, **5** texts digitised | "verse-level, searchable, citable" |
| Disease model | healthy-vs-diseased near-perfect; **~67%** on 27 fine-grained field classes, attention verified on the lesion | only if asked; lead with "verified to look at the lesion" |
| Soil model | soil type **89.9%**, moisture **95.8%**, texture **67.9%** | only if asked |

## 8. Things NOT to say (accuracy / audience)

- Do **not** name models or libraries (EfficientNet, Llama, ChromaDB, RAG, BM25…). Say "image-recognition model", "language model", "search".
- Do **not** claim the texts have been *fully* digitised — five of six are; Vishvavallabha is next.
- Do **not** claim Sanskrit was machine-translated (keep the corpus described as "verse-level passages"); leave that detail out unless Dr. Pandey wants it in.
- Do **not** quote the over-refusal percentage. If asked "does it always answer?": *"No — it refuses about half of the time when the texts are thin. That's deliberate; coverage is the limiter, and we're expanding the corpus."*
- Do **not** present the NITI-manual result (too new, still being analysed).
- Do **not** mention RAGAS or faithfulness scores by name — say "answers were judged faithful to the source passages".

## 9. Visual assets

**In the repo now**
- `demo_images/leaf_apple_scab.jpg`, `demo_images/leaf_corn_rust.jpg` — input leaf photos (slide 3)
- `demo_images/soil_alluvial.jpg` — input soil photo (slide 3)
- `architecture.html` — full system diagram; screenshot it for a backup slide only

**Needed from Colab (strongest slides — capture if at all possible)**
- Streamlit app result page: leaf → diagnosis → cited advice with the verse citation visible (slide 6)
- Grad-CAM heat-map **before/after** — background-focused vs lesion-focused (slide 5)

**Generate (if no screenshots):** simple icon flow for slide 3; three cards for slide 5; stat tiles for slide 6.

## 10. Design brief

- 16:9, **maximum 8 slides**, one idea per slide, ≤ 25 words of on-slide text, large type (title ≥ 36 pt, body ≥ 24 pt) — it will be viewed on a shared screen at 7 PM.
- Palette: deep leaf green `#1F4D2E` · turmeric/ochre `#C8891E` · off-white parchment `#F6F1E7` · charcoal text `#222`. Feels "classical text meets field", not "tech startup".
- Fonts: a clean serif for titles (e.g. Playfair Display / Georgia), clean sans for body (Inter / Calibri). One Devanagari accent on the title slide only.
- No stock "AI brain" imagery. Use the real leaf/soil photos and, if available, the real app screenshot and heat-map.
- Speaker notes on every slide (the "Say" text above).

## 11. Paste-ready prompt for an AI slide tool

> Build an 8-slide, 16:9 presentation for a 5-minute online talk at an ICSSR seminar on Revitalising Indigenous Knowledge Systems (sub-theme: Digital Preservation, AI and Documentation of Indigenous Knowledge). Audience: sociologists and heritage scholars, not engineers — plain language, no technical acronyms. Title: "An IKS-Grounded Multimodal Advisory System: Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts" by Ankit Pawar and Dr. Akshay Pandey, PDPM IIITDM Jabalpur. Use exactly the slide structure, on-slide text, speaker notes, verified numbers, "do not say" rules, and design brief in the attached file. Keep ≤ 25 words per slide, large type, palette deep green / turmeric / parchment, classical-text feel. Put the speaker notes in each slide's notes field.

## 12. Q&A — likely questions, short answers

- **"Does it replace the expert?"** No — it retrieves and cites; uncertain cases go to an agronomist. It's an access tool, not an authority.
- **"How do you know the AI isn't making things up?"** It can only compose from passages it retrieved, must cite them, and a check confirms each citation was really retrieved. In evaluation: zero fabricated citations.
- **"What if the texts don't cover the disease?"** It says so. Refusing is a feature. Coverage is the limiter, so we're adding texts (Vishvavallabha next).
- **"Is the translation reliable?"** We use published scholarly translations; the corpus keeps verse numbers so any passage can be checked against the edition.
- **"Which crops?"** Trained on common crops; because the texts prescribe by symptom, it can advise cautiously on others, with a confidence figure and a caution.
- **"Can farmers actually use it?"** Prototype is a web app; farmer-facing deployment and field validation are the next phase.
