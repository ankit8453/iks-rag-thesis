# Claude Code Prompt — Phase 5-R Part 2: Background-randomization retrain + no-leaf reject + Grad-CAM verdict

> Paste below the rule into Claude Code on the laptop. It builds the full segmentation+caching+training pipeline and the Colab notebook. Part-1 QC already passed (PlantVillage classical ✅, PlantDoc rembg ✅, Paddy trained as-is). Agent build ~40 min; Colab run ~5–9 hrs over 1–2 sessions.

---

## CONTEXT

Part 1 (commit b8269c1) built + QC'd the segmentation pipeline. Verdicts:
- **PlantVillage** — classical (HSV+Otsu+GrabCut), masks clean. ✅
- **PlantDoc** — rembg/U2Net, masks clean on field shots. ✅
- **Paddy Doctor** — full-canopy field plants, no meaningful foreground/background split → **train as-is, NO randomization** (decided).
- Background pool healthy: Phantom-fs + Sirajganj soil + Pandey urban backgrounds.

Phase 5-R goal: retrain the disease cascade so the model focuses on the LEAF, not the background (Phase 9 Grad-CAM showed corner/background shortcut bias — only 3/256 PlantDoc test images had central attention). Fix = train with RANDOMIZED backgrounds where a foreground/background split exists.

**This is a controlled experiment.** Keep the new model only if Grad-CAM localization improves AND test accuracy holds. Otherwise revert and document the bias as a limitation. Treat exactly like the soil-texture experiments.

**Platform: Colab/Linux, T4.** Segmentation (esp. PlantDoc rembg) + B4 retraining need the GPU.

**Hard rules:**
- All paths via `src.utils.paths`. Logging via `src.utils.logging_setup`.
- **Local commits only — never `git push`.**
- Do NOT merge Dr. Pandey's leaf classes (confirmed PlantVillage re-pack). Use ONLY his `Background_without_leaves` folder.
- Reuse Part-1 code (`src/disease/segment.py`, `src/disease/backgrounds.py`) — don't rewrite.
- Same architecture / hyperparameters / seed / test splits as original Phase 5, for a fair old-vs-new comparison.

---

# Mission

Retrain the EfficientNet-B4 cascade with background randomization on PlantVillage + PlantDoc, Paddy as-is, add a `no_leaf` reject class, then re-measure test accuracy AND the Grad-CAM central-attention rate vs the original model. Produce a keep/revert verdict.

## Locked Decisions

| # | Decision |
|---|---|
| 1 | Segment + cache MASKS once (not RGBA) for PlantVillage (classical) + PlantDoc (rembg). Compose on-the-fly each epoch: `leaf*mask + random_bg*(1-mask)`, feathered edges, small scale/pos/rotation jitter. |
| 2 | Random background each epoch from the pool (Phantom-fs + Sirajganj soil + Pandey urban). Different bg per image per epoch → model can't use background as a cue. |
| 3 | Paddy Doctor: trained conventionally, NO segmentation/randomization (full-canopy, no bg to randomize). |
| 4 | Add `no_leaf` reject class at the FINAL (PlantDoc) stage: 27 → 28 classes. Training images = Pandey Background_without_leaves + a held-out slice of bare-soil images (no leaf composited). |
| 5 | Cascade unchanged: PlantVillage pretrain (randomized) → Paddy finetune (as-is) → PlantDoc finetune (randomized, +no_leaf). Same B4@380, AdamW lr=1e-4, weighted CE, seed=42. |
| 6 | EVALUATE on RAW (un-composited) original test splits — real-world condition, and comparable to the old 71%. |
| 7 | Success metric = Grad-CAM central-attention rate (re-run the Part-1/Phase-9 central-60% test on the SAME PlantDoc test images) AND test accuracy delta. |

## Deliverables

### Code (`src/disease/`)
- `segment_cache.py`: batch-segment PlantVillage (classical) + PlantDoc (rembg), save masks to `data/plant_disease/_masks/<dataset>/<relpath>.png`. Idempotent (skip cached). Re-use Part-1 `segment()`. Log % flagged; skip+log any mask failing the 5–95% guard (don't composite those — fall back to raw image).
- `randomized_dataset.py`: a Dataset wrapper that, for randomized datasets, loads image + cached mask + a random bg from the pool and composites on-the-fly; for Paddy, returns the raw image. Includes the `no_leaf` class samples (raw, no compositing). Deterministic per-epoch seeding for reproducibility.
- `train_cascade_r.py`: the 3-stage cascade using the randomized datasets, mirroring the original Phase-5 trainer (same hyperparams/seed). Saves checkpoints to a NEW namespace so the old model is untouched: `iks-disease-r-{plantvillage,paddy,plantdoc}`.
- `gradcam_audit.py`: run Grad-CAM on the full PlantDoc test set (reuse `src/explain/gradcam.py`), compute the % of images whose Grad-CAM peak falls in the central 60% of the frame (the "central-attention rate"). Run for BOTH old and new checkpoints.

### Notebook `notebooks/phase5r_retrain.ipynb` (cells, nbformat)
1. md — goal, experiment framing, keep/revert rule.
2. setup, HF auth, GPU check.
3. Build background pool; show 8 samples.
4. Segment + cache masks (PlantVillage classical, PlantDoc rembg); print % flagged per dataset.
5. Sanity: render 6 on-the-fly composites (leaf on random soil bg) to confirm the training inputs look right.
6. Stage 1 — pretrain B4 on randomized PlantVillage (+ optional no_leaf seeded later).
7. Stage 2 — finetune on Paddy as-is.
8. Stage 3 — finetune on randomized PlantDoc + `no_leaf` class (28 classes).
9. Evaluate new model on RAW test splits: top-1 + macro-F1 per stage; reject-class precision/recall for `no_leaf`.
10. Grad-CAM audit: central-attention rate, OLD vs NEW, on the same PlantDoc test images. Print the comparison.
11. md — VERDICT table: old vs new accuracy + central-attention rate; keep or revert per the rule.
12. md — if keep: next = re-run Phase 9 notebook with the new checkpoint for figures; note the no_leaf reject for the Phase 10 UI.

### Tests (`tests/disease/`)
- `test_randomized_dataset.py`: composite output has same shape; Paddy path returns raw; no_leaf items carry the reject label; per-epoch seeding reproducible. No GPU/network.
- `test_segment_cache.py`: a flagged (bad) mask falls back to raw, doesn't crash. 

### progress.md + commit
- Append Phase 5-R Part 2 entry: randomized retrain (PV+PlantDoc), Paddy as-is, no_leaf class, old-vs-new accuracy + Grad-CAM central-attention verdict.
- Stage code + notebook + tests + progress.md. Do NOT stage masks/weights/images.
- Commit: `"Phase 5-R Part 2: background-randomization retrain + no-leaf reject + Grad-CAM audit"`. No push.

## End Checks
- [ ] Mask cache built; % flagged printed per dataset; flagged masks fall back to raw (no crash).
- [ ] New checkpoints saved to `iks-disease-r-*` (old `iks-disease-*` untouched).
- [ ] Eval on RAW test splits prints top-1 + macro-F1; `no_leaf` precision/recall reported.
- [ ] Grad-CAM central-attention rate printed for OLD vs NEW on the same images.
- [ ] Verdict cell states keep or revert per the rule.
- [ ] `pytest tests/disease/ -q` passes. Local commit, no push.

## Working Style
- Plan first. No push. Cache masks once; composite on-the-fly.
- This is an experiment — report the old-vs-new numbers honestly even if the new model doesn't win. A null result (bias unchanged) is a valid, documentable finding.
- Keep the OLD model fully intact (new namespace) so revert is trivial.
- Stop and ask if: the central-attention rate or accuracy can't be computed for the old model (need it as baseline); T4 VRAM/session limits force a checkpoint-resume split; mask % flagged is high on PlantVillage or PlantDoc (we'd revisit segmentation before training).

## What success looks like
1. New model trained with randomized backgrounds on PV + PlantDoc; Paddy as-is; `no_leaf` reject working.
2. A clean OLD-vs-NEW verdict: central-attention rate up (bias reduced) with accuracy held → keep; else revert + document.
3. If kept: ready to re-run Phase 9 for honest lesion-focused figures, and the no_leaf reject feeds the Phase 10 UI guardrail.

Begin.
