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

## STAGE 0 — Capture and score Llama's real queries  ⬜

Nothing has ever recorded what Llama actually writes. Do this before trusting any
retrieval number.

- [x] **0.1** Write `scripts/capture_llama_queries.py` + `scripts/compare_query_sources.py` — DONE — feed the 18 disease labels (+ soil
      values) through Strategy B, save each generated query to a file. *(Runs on Colab, free
      T4, Llama-3.1-8B.)*
- [ ] **0.2** Run it on Colab → `data/eval/llama_generated_queries.json`
- [ ] **0.3** Score them locally: `python scripts/check_coverage.py data/eval/llama_generated_queries.json llama`
- [ ] **0.4** `python scripts/compare_query_sources.py` — side by side, per query, with a verdict
- [ ] **0.5** Write the verdict into `EXPERIMENT_LOG.md` §6o — including which was better and by how much

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

## STAGE 3 — Fix Llama, only if Stage 0 says it needs it  ⬜

In order, cheapest first. Stop as soon as the score is good enough.

- [ ] **3.1** Add 4–5 real corpus lines to the Strategy-B prompt as examples (few-shot)
- [ ] **3.2** Add a corpus word list to the prompt ("prefer these words: yellowness,
      paleness, oozing, withering, scorched…")
- [ ] **3.3** Re-run Stage 0 and measure again
- [ ] **3.4** *Only if still weak:* two-round retrieval — rough query → show Llama the real
      passages → let it rewrite using their words → retrieve again
- [ ] **3.5** **No fine-tuning.** Needs data we don't have, and would lock us to today's corpus

---

## STAGE 4 — Phase 11, the honest re-run  ⬜

- [ ] **4.1** Push the final corpus: `python scripts/push_corpus_chunks.py`
- [ ] **4.2** Run **A** — the new domain queries alone (does the corpus answer its own subjects?)
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

## Open / parked

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
- [ ] **Vishvavallabha** — blocked, book not obtained. Useful, not urgent
- [ ] **Upavanavinoda English Introduction** OCR — blocked, Gemini quota

---

## Not thesis (tracked so it isn't forgotten)

- **InferAI** (B.Tech guidance) — `inferai/WORK_ORDER_for_Monday_meeting.md`. Journal
  deadline likely 31 Oct. Blocker: 5 authors, journal allows 4 — ask Dr. Pandey.
- **ICSSR book chapter** — deadline 30 Oct. 3,000–6,000 words, APA, needs a plagiarism
  report *and* an AI-content report, so it must be rewritten in Ankit's own words.
- **NCSTC proposal** — awaiting Dr. Pandey's decision.
