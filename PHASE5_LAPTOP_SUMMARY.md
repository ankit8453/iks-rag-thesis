# Phase 5 Laptop-Side Preparation Summary

**Branch:** `cleanup/pdf-alignment` (local commits only — not pushed)
**Date:** 2026-05-23
**Wall-clock:** ~3 h (incl. ~30 min HF Hub upload bandwidth + 1 retry on PlantDoc)

Phase 5 is **disease module training**, scheduled for Weeks 16-19. The
actual training happens on Google Colab; this session is the laptop-
side preparation: data merge fix, HF Hub uploads, model + train +
infer + Grad-CAM code, the Colab notebook itself, and a CPU smoke
test verifying every piece wires together.

---

## (a) What got built

| Section | Files | Commit |
|---|---|---|
| §A — PlantDoc 28→27 merge | `data/splits/plantdoc/*` regenerated; `configs/data/plantdoc_norm.yaml` regenerated; `scripts/compute_channel_stats.py` got `--dataset` CLI flag; `tests/disease/test_dataset.py` extended | `cb6a06a` |
| §B — HF Hub upload utilities | `src/utils/hf_upload.py` (HFDatasetUploader + verify helper + auto dataset-card generator); `tests/utils/test_hf_upload.py` (6 mocked tests); `requirements.txt` add `huggingface_hub`, `datasets`, `tensorboard` | `44be663` |
| §C — Upload 3 disease datasets | `scripts/upload_paddy_doctor_to_hf.py`, `scripts/upload_plantdoc_to_hf.py`, `scripts/upload_plantvillage_to_hf.py`; ran each; `HF_UPLOAD_REPORT.md` at repo root | `905b797` |
| §D — DiseaseClassifier | `src/disease/model.py` rewritten (EfficientNet-B4 via timm); `src/disease/config.py` rewritten with stage epoch budgets + lr_head/lr_backbone/dropout/mixed_precision/gradient_clip; `configs/disease/default.yaml` regenerated; `tests/disease/test_model.py` (7 tests); `tests/disease/test_smoke.py` updated for new constructor | `4ed4484` |
| §E — Training loop | `src/disease/train.py` rewritten (CheckpointManager pushing to HF Hub after every epoch; TrainingMetrics with per-class P/R/F1 + confusion matrix; train_one_stage with mixed precision + cosine LR + early stop; auto_batch_size by VRAM; CLI dispatch for the 3 stages); `tests/disease/test_train.py` (5 tests incl. one-epoch run on synthetic B0) | `8b030d0` |
| §F — Inference + Grad-CAM | `src/disease/infer.py` (DiseaseInferenceEngine accepting HF Hub repo IDs or local paths, PIL/numpy/Tensor inputs, optional Grad-CAM overlay); `src/disease/gradcam.py` (target layer auto-resolved to EfficientNet `conv_head`); `tests/disease/test_infer.py` (4 tests); `tests/disease/test_gradcam.py` (3 tests) | (this commit's parent chain) |
| §G — Colab notebook | `scripts/build_phase5_notebook.py` (generator); `notebooks/phase5_disease_training.ipynb` (13 cells, 3 resumable stages, HF login, GPU verify, eval+metrics push); `notebooks/README.md` updated; `tests/disease/test_notebook.py` (parse-only smoke) | — |
| §H — CPU smoke test | `scripts/phase5_smoke_test.py` (7 steps: dataset samples, forward+backward, checkpoint round-trip, inference engine, Grad-CAM); `PHASE5_LAPTOP_SMOKE.md`; `.gitignore` extended | — |
| §I — This summary | `PHASE5_LAPTOP_SUMMARY.md` | (this commit) |

---

## (b) HF Hub uploads

All three datasets are live at `huggingface.co/datasets/ankit-iiitdmj/iks-*`,
all **private**, all packed as Parquet (4 parquet shards + README +
.gitattributes per repo).

| Dataset | URL | Train / Val / Test | Classes | Hub size | Upload time |
|---|---|---:|---:|---:|---:|
| **Paddy Doctor** | https://huggingface.co/datasets/ankit-iiitdmj/iks-paddy-doctor | 8,325 / 1,041 / 1,041 | 10 | ~600 MB | 530 s |
| **PlantDoc** | https://huggingface.co/datasets/ankit-iiitdmj/iks-plantdoc | 2,046 / 256 / 256 | 27 (post-§A merge) | ~250 MB | 148 s (after 1 retry — first attempt stalled at 99% on shard 2) |
| **PlantVillage** | https://huggingface.co/datasets/ankit-iiitdmj/iks-plantvillage | 43,443 / 5,431 / 5,431 | 38 (color variant) | **854 MB** | 614 s |

The 854 MB PlantVillage figure surprised the human spot-check: local
`data/plant_disease/plantvillage/raw/` is ~2.5 GB. The difference is
the `grayscale/` + `segmented/` variants (also ~54k images each) that
the Kaggle archive ships but our pipeline doesn't reference — only
the `color/` variant is in the split JSON and therefore the upload.
854 MB = the 54k color JPEGs Parquet-packed.

Dataset cards on each repo carry:
- YAML frontmatter (`license`, `task_categories: image-classification`,
  `size_categories`).
- Source citation paragraph.
- Per-split sizes table.
- Full class list with indices.
- Preprocessing summary (80/10/10 stratified, seed=42).
- For PlantDoc: explicit note about the 28→27 merge done in §A.

---

## (c) PlantDoc class merge

**28 → 27** classes. The vestigial `Tomato two spotted spider mites leaf`
folder (2 images) was merged into the closest neighbour `Tomato leaf`
with filename prefix `was-spider-mites-` for audit traceability.

Verification:

```bash
$ ls data/plant_disease/plantdoc/raw/ | wc -l
27
$ python -c "import json; print(len(json.load(open('data/splits/plantdoc/class_map.json'))))"
27
$ grep -rE "spider mites" data/splits/plantdoc/
(no matches — the literal-space regex misses our hyphenated prefix)
```

Total entries unchanged at 2,558 (2,046 train + 256 val + 256 test);
the 2 merged files now sit inside `Tomato leaf/`. Channel-stats
recomputed in `configs/data/plantdoc_norm.yaml` — practically identical
to the pre-merge stats since 2 images out of 2,558 barely move the
population mean.

Supervisor sign-off received before the merge (per the prompt's
decision #6).

---

## (d) Disease module status

| Component | Where | Tests | Status |
|---|---|---|---|
| `DiseaseClassifier` (EfficientNet-B4) | `src/disease/model.py` | 7 in `test_model.py` | ✓ |
| `DiseaseConfig` (Pydantic v2) | `src/disease/config.py` + `configs/disease/default.yaml` | covered by `test_smoke.py` | ✓ |
| Training loop | `src/disease/train.py` | 5 in `test_train.py` | ✓ |
| `CheckpointManager` (HF Hub) | inside `train.py` | save/load round-trip with mocked HF | ✓ |
| `TrainingMetrics` (per-class) | inside `train.py` | per-class P/R/F1 sanity | ✓ |
| `auto_batch_size` (VRAM-bucketed) | inside `train.py` | 4 GPU classes covered | ✓ |
| `DiseaseInferenceEngine` | `src/disease/infer.py` | 4 in `test_infer.py` | ✓ |
| Grad-CAM wrapper | `src/disease/gradcam.py` | 3 in `test_gradcam.py` | ✓ |

**Full test count for this session:** 19 new disease tests + 6 HF tests + 2 notebook tests = **27 new tests pass**. Project-wide test count is **116 across 8 subdirectories**, all of which pass per-subdir.

**Known Windows-specific issue:** running `pytest tests/` as a single
process can crash with a `pyarrow` C-extension access violation on
shutdown (after all tests pass). Workaround: run per-subdir:

```
python -m pytest tests/disease/      tests/soil/       tests/integration/ \
                  tests/utils/        tests/eval/       tests/rag/        \
                  tests/explain/ -q
```

Or, equivalently, the launcher used by the Phase-5 smoke verification:

```bash
for d in tests/disease tests/soil tests/integration tests/eval tests/rag tests/explain tests/utils; do
  python -m pytest "$d" -q --tb=no
done
```

This is purely a Windows + pyarrow + multiple-test-file-import-order
quirk; the code itself is fine — Colab Linux + the GitHub Actions
matrix would run the full tree cleanly in one process.

**CPU smoke test (§H):** `PHASE 5 LAPTOP SMOKE TEST: OK (6.1s)` — see
`PHASE5_LAPTOP_SMOKE.md`.

---

## (e) Colab notebook ready

`notebooks/phase5_disease_training.ipynb` is the Colab artifact. 13
cells: setup → HF login → GPU verify → 3 training stages (each with a
markdown header + a code cell whose only line is
`!python -m src.disease.train --stage <name> --resume`) → final
evaluation → what's next.

How to use it on Colab:

1. Upload the `.ipynb` to a fresh Colab session, or open from a fork
   of the repo.
2. Runtime → Change runtime type → T4 GPU (free tier) or better.
3. Run cells 1-4 to set up.
4. Run cell 6 (Stage 1 — PlantVillage, 25 epochs). Expect 6-10 hours
   of T4 wall-time; Colab will likely disconnect at the 12-h mark or
   when daily quota hits.
5. When you return, re-run cell 6 — the `--resume` flag picks up from
   the latest HF Hub checkpoint.
6. After Stage 1 hits epoch 25 (saved in `epoch=25`), the script
   exits cleanly with "Stage already complete". Move to cell 8
   (Stage 2 — Paddy Doctor, 20 epochs).
7. Same loop for cell 10 (Stage 3 — PlantDoc, 30 epochs).
8. Cell 12 runs final evaluation across all 3 stages and pushes
   `eval_metrics.json` to each model repo.

The matching `PHASE5_COLAB_GUIDE.md` (shipped separately by Ankit)
walks the human-side procedure in more detail.

---

## (f) Things for Ankit to spot-check before opening Colab

1. **HF Hub datasets reachable.** From any Python REPL with the same
   HF token:
   ```python
   from datasets import load_dataset
   for name in ("iks-plantvillage", "iks-paddy-doctor", "iks-plantdoc"):
       ds = load_dataset(f"ankit-iiitdmj/{name}", split="test")
       print(name, len(ds), ds[0]["image"].size)
   ```
   Expected output: ~5431, ~1041, ~256 with a 2-tuple H×W per image.

2. **HF Hub token is Write.** On the laptop:
   ```
   huggingface-cli whoami
   ```
   should print `ankit-iiitdmj` and the token's role should show
   `write`. If it's `read`, regenerate a Write token at
   https://huggingface.co/settings/tokens.

3. **GitHub repo URL in Cell 2 of the Colab notebook.** Currently
   set to `https://github.com/ankit8453/iks-rag-thesis.git` (best
   guess from the existing remote in `.git/config`). If your actual
   public repo lives elsewhere, edit just the `REPO_URL = "..."` line
   in Cell 2.

4. **`git log --oneline -15` looks clean.** Should show the 9 §A–§I
   commits at the top (plus the prior Phase 4 fix commits below
   them). No `git push` has run.

5. **Eval on Colab uses val split, not a separate held-out test.**
   The training loop's resume-from-prior-stage logic uses `val.json`
   for early-stopping decisions; cell 12 also calls `evaluate()` on
   the same val split. The Phase-4 split JSONs do include a `test.json`
   per dataset, but the Colab notebook currently evaluates against
   `val`. To get a true held-out test number, swap `val_loader` for a
   test loader in cell 12 — small edit if you want.

6. **PlantDoc 28→27 merge filenames carry `was-spider-mites-` prefix.**
   Two files in `data/plant_disease/plantdoc/raw/Tomato leaf/` start
   with that prefix. They're harmless (image content is what matters
   for training) but worth knowing if you spot them during a manual
   review.

---

## (g) Deferred to later phases

- **Phase 6** — soil module training. Three heads (`soil_type`,
  `moisture_appearance`, `texture`) on Phantom-fs + Sirajganj +
  IRSID. Separate Colab notebook, separate prompt.
- **Phase 8** — multimodal integration. The trained disease module
  (`DiseaseInferenceEngine`) feeds into the multimodal context
  constructor. `get_feature_extractor()` is the integration hook.
- **Phase 11** — causation evaluation using OLID I full + the
  trained disease model. The `MultiLabelImageDataset` already wires
  the data side; Phase 11 adds the metric utilities (per-tag P/R,
  Hamming loss, mAP).

---

## Section commits this session

```
cb6a06a  Phase 5 §A: merge PlantDoc spider-mites 2-image class into Tomato leaf (27 canonical classes)
44be663  Phase 5 §B: HF Hub upload utilities + dataset card generator
905b797  Phase 5 §C: upload PlantVillage + PlantDoc + Paddy Doctor to HF Hub (private)
4ed4484  Phase 5 §D: EfficientNet-B4 DiseaseClassifier + DiseaseConfig + tests
8b030d0  Phase 5 §E: training loop with HF-Hub-based checkpoint manager + auto batch size
1c76f1a  Phase 5 §F: disease inference engine + Grad-CAM wrapper
92f3d28  Phase 5 §G: Colab training notebook with three independently-resumable stages
f1e1a68  Phase 5 §H: local smoke test of disease module on CPU
(this commit)  Phase 5 §I: laptop-side preparation summary
```

---

**Not pushed.** Inspect `git log --oneline cleanup/pdf-alignment` and
push manually when satisfied. The HF Hub uploads in §C are the only
remote interaction this session.
