# Thesis TO-DO — agreed 26 Sep 2026

Nothing here gets skipped or reordered without saying so. Tick items off as they finish.

**Money: ₹0.** No Gemini, no OpenAI. Local CPU + Colab free tier only.
**Safety:** git tag `thesis-safe-2026-09-26-pre-citationfix`, full backups, restore steps in
`corpus/RESTORE.md`.

---

## The question we are answering first (Ankit's call, 26 Sep)

We have two sets of evaluation queries and **no evidence which is right**:

- **Llama's queries** — what the live system actually produces from a disease label
- **Ankit/Claude's hand-written queries** — what the evaluation has been using all along

Until now the evaluation assumed the hand-written ones were correct. That was never tested.
**So test both, then fix whichever is wrong.** Three outcomes, all useful:

| Outcome | Meaning | What we do |
|---|---|---|
| Llama's score better | the hand-written queries were holding the system back | replace the eval queries with Llama's real output |
| Hand-written better | Llama is writing weak queries | fix Llama with examples + corpus word list |
| Both similar | the bridge is fine; the limit really is corpus coverage | leave both, report it |

---

## STAGE 0 — Capture and score Llama's real queries  ✅ DONE 26 Sep — hand-written won 14-0

Nothing has ever recorded what Llama actually writes. Do this before trusting any
retrieval number.

- [x] **0.1** Write `scripts/capture_llama_queries.py` + `scripts/compare_query_sources.py` — DONE — feed the 18 disease labels (+ soil
      values) through Strategy B, save each generated query to a file. *(Runs on Colab, free
      T4, Llama-3.1-8B.)*
- [x] **0.2** Run it on Colab → `data/eval/llama_generated_queries.json`
- [x] **0.3** Score them locally: `python scripts/check_coverage.py data/eval/llama_generated_queries.json llama`
- [x] **0.4** `python scripts/compare_query_sources.py` — side by side, per query, with a verdict
- [x] **0.5** Write the verdict into `EXPERIMENT_LOG.md` §6o — including which was better and by how much
- [x] **0.6** ~~If Llama wins:~~ **Llama lost** (mean 0.045 vs 0.169), so the 11 provisional
      labels stand — hand-written wording is the generous ceiling. Original note: re-measure the 13 disease queries marked `no_coverage`.
      Their labels rest on the hand-written wording scoring low, and wording alone moves
      the score up to 17x. They are flagged `label_provisional: true` in the query set.

---

## STAGE 1 — Fix the citation labels  ✅ DONE 26 Sep  *(code already done, 422 tests pass)*

17 chunks share 4 labels, so a citation cannot identify a passage and the 55%
valid-citation rate cannot be checked.

- [x] **1.1** Rebuild: `IKS_EMBED_DEVICE=cpu python -m src.rag.corpus.build_corpus`
      *(~74 min, local CPU, no API — the OCR text is already cached)*
- [x] **1.2** `python scripts/verify_corpus.py` — **must print PASS.** If it says CHANGED,
      stop and restore from `corpus/RESTORE.md`
- [x] **1.3** Confirm "every chunk is uniquely citable" in the same output
- [x] **1.4** Prune 17 orphaned ChromaDB vectors (`prune_stale_vectors.py --apply`) — the
      rebuild added the renamed chunks but left their old ids behind, so the store held 250
      vectors for 233 chunks. `verify_corpus.py` now checks the store too.

---

## STAGE 2 — Score the 16 new domain queries  ✅ DONE 26 Sep

These cover what the books actually contain: pests, seed treatment, soil, manure, season,
rain, water, planting, tree wounds, storage.

- [x] **2.1** `python scripts/check_coverage.py data/eval/new_queries_draft_2026-09-26.json newq`
- [x] **2.2** Read the tier counts (strong / marginal / none)
- [x] **2.3** `python scripts/merge_new_queries.py` (dry run — check the plan)
- [x] **2.4** `python scripts/merge_new_queries.py --apply`
- [x] **2.5** Keep the failures as honest negatives — **do not delete them**
- [x] **2.6** Grade every candidate by READING its passages. The score said 16/16 strong;
      reading said 8 answer / 7 partial / 1 no. Score proves absence, not presence.

---

## STAGE 3 — Fix Llama  ✅ DONE 26 Sep — gap closed 74%, now within noise

In order, cheapest first. Stop as soon as the score is good enough.

- [x] **3.1** Add 4–5 real corpus lines to the Strategy-B prompt as examples (few-shot)
- [x] **3.2** Add a corpus word list to the prompt ("prefer these words: yellowness,
      paleness, oozing, withering, scorched…")
- [x] **3.3** Re-run and measure — round 1: 0.0445 -> 0.0977; round 2: -> **0.1368**
      vs a 0.1691 hand-written ceiling. Mean difference now inside the 0.05 noise floor.
      (old note: round 1 mean 0.0445 -> 0.0977
      (43% of the gap closed, compliance 17/17 -> 0 on three of four faults). Round 2
      pending: picks up the rule-3 'necrotic' contradiction fix
- [x] **3.4** ~~two-round retrieval~~ **DROPPED — not needed.** Prompt work alone was enough — rough query → show Llama the real
      passages → let it rewrite using their words → retrieve again
- [ ] **3.5** **No fine-tuning.** Needs data we don't have, and would lock us to today's corpus

---

## STAGE 4 — Phase 11, the honest re-run  ⬜

- [x] **4.1** Push the final corpus: `python scripts/push_corpus_chunks.py`
- [~] **4.2** Run **A** — the new domain queries alone (does the corpus answer its own subjects?)
- [ ] **4.3** Run **B** — the whole query set together
- [ ] **4.4** Report disease queries and domain queries **separately** — they measure
      different things and one number would mislead
- [ ] **4.5** Log both to `EXPERIMENT_LOG.md`

---

## STAGE 5 — Write-up  ⬜

- [ ] **5.1** Add the 6% vs 60% table (disease-label queries vs ordinary-language queries) —
      it states the finding better than "13 of 22 failed"
- [ ] **5.2** Correct the Vishvavallabha claim: **useful, not decisive.** One book cannot
      explain 13 failures; the query set was the bigger problem
- [ ] **5.3** Add the Stage 0 result — whether the bridge writes good queries
- [ ] **5.4** Update `RESEARCH_WRITEUP` and rebuild the .docx

---

## Found during Phase 11 — act on these

- [x] **Re-run cells 2 and 3** after the citation-format fix — DONE, grounded 37.5% — the first trustworthy
      citation figure. 20.74% and 55% are both artefacts.
- [ ] **Report retrieval split by query type**, never pooled. The keyword baseline nearly
      caught up (nDCG 0.842 vs 0.783) because domain queries already use the corpus's
      vocabulary; the bridge's advantage is specific to modern disease labels.
- [ ] **Investigate dropping BM25.** dense_only beats the full hybrid on nDCG, MRR and
      Hit@5 (1.00 vs 0.9167) — third run in a row. The hybrid leg may be a net negative.
- [x] **Harden the refusal detector.** DONE 27 Sep — answered/partial/refused. q15 was counted as an answer while actually
      declining ("is not directly addressed. However...") and citing nothing. Honest
      refusal 93.75% is softer than it looks.
- [ ] **Correct the research write-up**: its citation figure came from the duplicate-label
      era and is not defensible.

## Final sequence (agreed 27 Sep)

- [x] Ingest Vishvavallabha → rebuild → verify_corpus PASS (270 chunks) — DONE 27 Sep
- [x] Push 270 chunks to HF — done
- [x] `review_labels.py` + full-text read of every doubtful label — done 27 Sep, 27/40 answerable
- [x] Final Phase 11 run (generated queries) — DONE 28–29 Sep: grounded 51.9%, over-refusal 40.7%,
      honest refusal 92.3%, unfounded 0%; retrieval P@5 0.80 / Hit@5 0.963.
      Open: exact-citation rate 39.6% unexplained (per-query file was not downloaded before
      the Colab session closed). Add a download step to the notebook's last cell.

## Open / parked

- [ ] **Kashyapiya translation check** — 12-verse worksheet against the Chowkhamba edition
      exists (`seminar/kashyapiya_translation_validation.md`) with empty verdicts. The chapter
      states this honestly; completing it removes the caveat.

- [ ] **Evaluate the soil advisory path.** The soil model outputs (type, moisture, texture)
      map onto the corpus's *strongest* areas — soil preparation, land suitability,
      watering. **Never evaluated.** May be the strong half of the system.
- [ ] **Wire in the disease-TYPE model** (~13 crop-agnostic types) — closer to how the texts
      think than 27 crop×disease labels
- [ ] **Let the farmer type what they see**, alongside the photo — bypasses the label
      vocabulary problem entirely
- [ ] **Front-matter prefixes** on 4 Vrikshayurveda chunks (needs another rebuild — bundle
      with a future one)
- [ ] **Expert gold query set** with passage-level labels (enables Recall@k)
- [x] **Vishvavallabha — OBTAINED 26 Sep** (archive.org, full AAHF 2004 Sadhale edition,
      144 pp, all 9 chapters; ch. VIII = 79 verses on diseases and treatment). Local under
      `newbooks/vishvavallabha/` (gitignored — copyrighted). **Not yet ingested — awaiting go.**
      Plan: text_layer path, Cyrillic char-map in cleaning, split on 9 headings, then
      verify → coverage → Phase 11 against the 233-chunk baseline. See §6q.
- [ ] **Upavanavinoda English Introduction** OCR — blocked, Gemini quota

---

## Not thesis (tracked so it isn't forgotten)

- **InferAI** (B.Tech guidance) — `inferai/WORK_ORDER_for_Monday_meeting.md`. Journal
  deadline likely 31 Oct. Blocker: 5 authors, journal allows 4 — ask Dr. Pandey.
- **ICSSR book chapter** — deadline 30 Oct. **Draft v1 complete (29 Sep):**
  `paper/icssr_chapter/CHAPTER_DRAFT_v1.docx`. Next: Shivam Dubey review → Turnitin →
  revise flagged paragraphs → submit with both reports. 7 references tagged [verify].
- **NCSTC proposal** — awaiting Dr. Pandey's decision.
