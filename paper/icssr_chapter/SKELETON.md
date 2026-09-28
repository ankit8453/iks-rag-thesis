# ICSSR chapter — skeleton

*Bullets are facts, not sentences. Turn each bullet into 1–2 sentences in YOUR words.
Don't copy bullet wording — it is deliberately rough.*

**Target: ~5,000 words.** Word budget per section is a guide, not a rule.

| mark | meaning |
|---|---|
| 🟢 | **write now** — nothing will change |
| 🟡 | write now, but leave `[NUM]` where a number may change after Vishvavallabha / final eval |
| 🔴 | **wait** — depends on results not in yet |

**Suggested writing order:** 1 → 2 → 3 → 4 → 5 → 6 → 8 → 7 → 9 → 10 → abstract → title.
(Abstract and title go LAST — you can only summarise what's already written.)

---

## Title  🔴 (last)

- Submitted title: *"An IKS-Grounded Multimodal Advisory System: Connecting Image-Based
  Plant Diagnosis with Classical Indian Agricultural Texts"*
- Can keep it, or shift toward the new main finding (where the two traditions meet and
  don't). Decide after writing section 8.

## Abstract (≤300 words) + 5 keywords  🔴 (last)

- purpose → problem → what we built → what we found → why it matters for IKS
- keyword ideas (pick 5): Indian Knowledge Systems · Vrikshayurveda · digital preservation ·
  artificial intelligence · faithful AI / hallucination · agricultural heritage

---

## 1. Introduction  🟢  (~600 words)

**Para 1 — the knowledge exists**
- classical agri texts: Vrikshayurveda (Surapala, ~10th c.), Brihat Samhita, Krishi Parashara,
  Upavanavinoda, Kashyapiya Krishisukti, Vishvavallabha (Chakrapani Mishra, ~1577)
- organic plant care, soil, water, seasons — sustainable
- today: mostly unused

**Para 2 — why unused: three barriers**
- only scanned translations, not searchable
- describe problems by symptom + cause, not modern disease names
- AI tools ignore them — and general chatbots invent "ancient remedies" with fake verse numbers

**Para 3 — the danger of AI here**
- fluent + confident + wrong
- for heritage knowledge = distortion of the tradition, and farmer can't check
- so: any AI for IKS must cite its source or say "not found"

**Para 4 — what this chapter does**
- builds on paper presented at the seminar (11 Sep 2026) — say "revised and extended"
- photo of leaf + soil → AI names problem → searches the texts → answer WITH verse citation,
  or refuses
- main lesson (preview): measured where the two traditions overlap and where they don't

**Para 5 — roadmap**
- one line per section that follows

---

## 2. The texts and how they think about plant health  🟢  (~700 words)

**Para 1 — the six texts, one line each**
- Vrikshayurveda: trees, disorders, causes, remedies — main source
- Brihat Samhita: rain prediction, crops, water springs (ch.54), tree treatment (ch.55)
- Krishi Parashara: seasons, sowing, rain signs
- Upavanavinoda: gardens, tree care
- Kashyapiya Krishisukti: cultivation, land, water, seeds
- Vishvavallabha: 9 chapters — groundwater, wells, soil, planting, water, care,
  nourishment, **diseases & treatment (79 verses, the biggest chapter)**, botanical wonders

**Para 2 — the key idea: disease by CAUSE**
- plant disorder explained like Ayurveda for humans: vata / pitta / kapha imbalance
- plus physical causes: fire, lightning, axe wound, ants, faulty seed
- Vishvavallabha ch.8 opens: trees suffer "like human beings" from wind/bile/phlegm imbalance
  (quote it — in quotation marks, cite it)

**Para 3 — example remedies (makes it concrete for the reader)**
- vata → kunapa water (liquid manure), flesh/marrow/ghee
- kapha → white mustard paste at root, water with sesame + ash
- broken branch → dress with honey + ghee, sprinkle milk-water
- (cite Sadhale's translation for each)

**Para 4 — symptom-general, not crop-specific**
- a remedy for a symptom applies to any plant showing it
- confirmed by agronomy expert Dr. Sunita T. Pandey + literature
- Rananavare & Chitnis 2024 expert system asks symptoms, never the crop
- careful: *mostly* general — some crop-named recipes exist (mango, pomegranate)

**Para 5 — contrast with modern diagnosis**
- modern names by appearance / pathogen: scab, rust, Septoria leaf spot, blight
- two traditions cut the same reality along different lines
- this difference is the heart of the chapter → comes back in section 8

---

## 3. Why AI and heritage knowledge need special care  🟢  (~400 words)

- chatbots "hallucinate" = produce confident false statements
- for IKS: fake verse numbers, invented remedies, attributed to real texts
- user (farmer, student) can't verify Sanskrit sources → trust gets abused
- our design rule: **answer only from retrieved passages; cite chapter + verse; otherwise say
  the texts are silent**
- idea to state: refusing is a feature, not a failure — silence is more faithful to the
  tradition than invention
- link to the book's theme: digital preservation must *preserve*, not rewrite

---

## 4. What we built — in plain terms  🟡  (~800 words)

*One plain sentence per part. No equations. Maybe one simple diagram.*

**Para 1 — overall flow**
- leaf photo → disease model · soil photo → soil model · farmer picks the crop
- → "bridge" turns the modern label into symptom words
- → searches the digital corpus → answer with citation or refusal

**Para 2 — digital corpus**
- scanned translations → text (OCR) → split into verse-sized passages
- each passage tagged: book, chapter, verse, translator
- size: `[NUM]` passages from `[NUM]` sources (now 233 from 6; will change after Vishvavallabha)
- one modern source too: NITI Aayog natural-farming manual (pest recipes) — kept separate
  as "modern" tier

**Para 3 — the two vision models**
- disease model: trained on public leaf-photo datasets
- important discovery: early model was right for the wrong reason — looked at background
  (soil, sky), not the leaf — found using "heatmaps" (Grad-CAM) that show where the model looks
- fixed by training on cut-out leaf images: 72.3% → 66.6% accuracy, but now looks at the leaf
- point to make: honest lower number > impressive number earned from the background
- soil model: soil type 89.9%, moisture 95.8%, texture 67.9%

**Para 4 — the bridge (our key idea)**
- texts never say "Apple Scab" — so searching that name finds nothing
- language model rewrites it: "dark rough corky patches spreading over the leaves"
- the texts CAN match that

**Para 5 — the answering step**
- language model (Llama 3.1) reads only the retrieved passages
- must cite [book, chapter, verse] for every claim
- if nothing fits → says so

**Para 6 — for crops not in training**
- farmer names the crop; system still describes the symptom; shows confidence + caution;
  uncertain cases → expert review

---

## 5. Lessons from digitising the texts  🟢  (~500 words)

*This section is valuable for an IKS/digital-preservation audience — it's about how
digitisation can quietly go wrong.*

**Lesson 1 — OCR can silently lose text**
- one tool setting cut off long pages: a page of ~20 verses came out as 2
- caught only by comparing with the printed book by eye
- fixed + re-checked both books

**Lesson 2 — wrong boundaries pull in unwanted content**
- chapter-finding bug: "Growth of Crops" (4 printed pages) swallowed 13 following chapters —
  astrology, price prediction, swords, architecture — 167 pages
- ~25% of the corpus was material never chosen
- fixed; corpus 327 → 233 passages

**Lesson 3 — every passage must be uniquely citable**
- 5 different passages carried the same label "v.1.2" → a citation couldn't say which one
- fixed so every passage has its own label

**Para — the general point**
- a digital archive can look complete and still be wrong
- digitisation needs checking against the physical book, not just automatic tests

---

## 6. How we evaluated  🟢  (~400 words)

- test questions: `[NUM]` total (now 40) — 17 named after modern diseases, 23 ordinary
  farming questions (pests, soil, sowing season, rain, water, planting)
- + 2 "trap" questions no book covers → system SHOULD refuse
- measured:
  - does search find the right book? (plain words, not formulas)
  - does the answer cite a passage that was really retrieved?
  - does it refuse when it should — and not when it shouldn't?
  - a "no-corpus" test: same model, no texts → does it invent citations?
- honest note: test labels made by us, not yet checked by an expert (preliminary)

---

## 7. What we found  🟡/🔴  (~1,000 words)

**7.1 The system never invented a citation  🟢**
- 0% made-up citations — in every test run
- refused every trap question it should refuse (100% in earlier runs; `[NUM]` final)

**7.2 The texts answer ordinary farming questions well  🟢**
- disease-name questions: **1 of 17** had a genuine match
- ordinary-language questions: **8 of 16** answered directly, 15 of 16 on-topic
- examples:
  - "how often to water a young tree" → Vrikshayurveda gives schedule by soil + season
  - "signs rain is coming" → Brihat Samhita: ants moving eggs, snakes mating, chameleons
    looking up
  - "broken branch" → honey + ghee
  - "tree won't flower" → Upavanavinoda 177

**7.3 Where the traditions don't meet  🟡**
- 22 modern disease questions: only **4** had a real match, **13** had none
- not a search failure — the concept doesn't exist in the texts
- the texts index by cause (vata/pitta/kapha); modern diagnosis by what the spot looks like
- so there is simply no passage for "Septoria leaf spot"
- `[NUM]` — update after Vishvavallabha (does its 79-verse disease chapter change this?)

**7.4 The AI's wording mattered more than expected  🟢**
- same question, different words → score 17× different
  - "weak tree, pale drooping foliage" → 0.056
  - "tree does not flower or bear fruit" → 0.935
- first, the language model wrote poor search queries: started with soil, asked "what do the
  treatises say", used modern medical words
- 3.8× worse than human-written queries
- after rewriting its instructions (no retraining): level with human wording
- lesson: the bridge is essential, and must itself be tested, not assumed

**7.5 Final answer-quality numbers  🔴**
- grounded-answer rate, citation accuracy, over-refusal — waiting for final run
  (citation fix + Vishvavallabha)

---

## 8. Discussion — what the gap between traditions tells us  🟢  (~600 words)

*This is the most important section for THIS book. Take time here.*

- the system refusing most modern-disease questions ≠ failure
- it is a **measured boundary** between two knowledge systems
- classical texts: *why* a plant suffers (imbalance, injury)
- modern science: *what* it looks like / which organism
- neither is "wrong" — different questions, different organisation
- a system that invented a "classical remedy for Septoria" would be worse than useless —
  fabrication presented as heritage
- implication for digital preservation: archives should be searchable *in the tradition's
  own terms*, not forced into modern categories
- implication for AI: faithful = knowing where the text stops
- link to cultural continuity: respecting the knowledge = representing it accurately,
  including its limits

---

## 9. Limitations and future work  🔴  (~300 words, write near the end)

- test questions + labels are ours → need expert validation (agronomist, Sanskrit scholar)
- English translations only, not Sanskrit originals
- small test set
- future: expert-built questions, Sanskrit + Hindi support, farmer field testing,
  let farmers describe symptoms in their own words

---

## 10. Conclusion  🔴  (~250 words, write last before abstract)

- restate: built a way to reach classical texts from a photo, with citations or honest
  silence
- key finding: measured where old and new knowledge overlap
- closing thought: faithful digitisation means preserving the knowledge *and* its boundaries

---

## After the conclusion

- **Acknowledgements:** Dr. Sunita T. Pandey (agronomy guidance); Asian Agri-History
  Foundation translations
- **AI-use statement** (check organisers' rule first)
- **References (APA)** — I will prepare the full list

---

## What I (Claude) will prepare for you

- [ ] full APA reference list
- [ ] 1 simple system diagram (from the seminar slides, simplified)
- [ ] tables: the 6 texts; the findings numbers (7.2, 7.3, 7.4)
- [ ] fill every `[NUM]` once final results are in
- [ ] fact-check each section you send me
