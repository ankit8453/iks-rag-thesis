# Weekly Progress Log

## Overview
This document tracks weekly progress on the IKS Agricultural Advisory System thesis project. Each week includes completed tasks, blockers, and goals for the next week.

---

## Phase 6 V3-tiling: patch-based texture expansion (single-stage, V2 recipe)

**NOTE:** the earlier 3-stage sequential-transfer V3 (`PHASE6_V3_SEQUENTIAL_TRANSFER_PROMPT.md`) was run on Colab and **catastrophically collapsed** all three heads to ~20% val accuracy. That multi-stage / curriculum / per-stage-head-freezing pattern is ABANDONED. V3-tiling takes a fundamentally different, safer approach: the V2 training recipe is reused **byte-for-byte**, single-stage; the only thing that changes is the texture training data (source images are tiled into a grid of patches, with image-level split integrity to prevent leakage).

### Part 1 — tiled texture dataset (executed in this session)

`src/soil/tiling.py` adds `tile_image(img, grid)`, `check_patch_resolution(images, grid)`, and `build_tiled_split(items, grid)`. Patches are 224×224 LANCZOS resamples of the per-image grid cells; every patch carries its source image's label **and** a unique `source_id` so callers can assert no source image's patches cross train/val/test boundaries.

`scripts/build_tiled_texture_dataset.py` loads `ankit-iiitdmj/iks-soil-texture-irsid-vit` (the V1 prep dataset), assigns each source image a `source_id` of the form `f"{split}_{idx:04d}"` (split prefix guarantees disjointness by construction), tiles every split independently, runs an explicit `_assert_split_disjointness` check, and pushes to the new private repo `ankit-iiitdmj/iks-soil-texture-tiled` using the proven `Dataset.push_to_hub` streaming pattern from `feedback_hf_dataset_uploads.md`. Resolution guard fires when the median patch is below 120 px pre-resize and aborts with `exit 2` unless `--force` is passed.

**Run result (this session):** the resolution guard fired at the prompt's default `grid=4` — median source min-dim was 101 px (median image ~506×407), implying a 2.2× upscale to 224. Per the prompt's working style ("stop and ask if the resolution guard fires"), Ankit was asked and chose **grid=3**. The build then finished in 38 s and pushed **2,511 patches** (train 2007 / val 252 / test 252, from 223 / 28 / 28 source images respectively) across 3 parquet shards (~43 MB total private dataset). Audit JSON written to `data/soil/tiled_texture_audit.json`. Disjointness assertion passed.

### Part 2 — V3-tiling training stack (notebook + thin wrapper)

`src/soil/train_v3_tiling.py` is a thin wrapper that re-exports V2's training functions unchanged plus two new pieces:

- `SoilCheckpointManagerV3Tiling` — subclass of V2's manager pinned to the new model repo `ankit-iiitdmj/iks-soil-multitask-v3-tiling` (created on first Cell 8 run from Colab; not from this prompt session).
- `health_check(val_metrics, threshold=0.40)` — raises `RuntimeError` if any head's val top-1 falls below 40%. Called after the FIRST epoch's val pass in the notebook's Cell 9 so a collapse aborts in minutes, not after 10+ hours.

`notebooks/phase6_soil_training_v3_tiling.ipynb` (13 cells) is identical to the V2 notebook except: Cell 6 swaps the texture dataset to `iks-soil-texture-tiled`, Cell 9 calls `health_check` after epoch 1, Cell 12 prints a V2-vs-V3-tiling table + automatic SHIP/KEEP verdict and reports BOTH patch-level and **image-level (majority vote over a source image's patches)** texture metrics. Backbone, optimizer, scheduler, augmentation, Mixup/CutMix, label smoothing, epoch count = byte-for-byte the V2 recipe.

### Untouched

V1 (`iks-soil-multitask`) and V2 (`iks-soil-multitask-v2`) model repos, original `iks-soil-texture-irsid-vit` dataset repo, and every V1/V2 source file or notebook are **unchanged in git status and on HF Hub**.

### Tests

`pytest tests/soil/test_tiling.py tests/soil/test_train_v3_tiling_smoke.py -q` → 17 passed. Covers patch counts/shapes, grid validation, resolution-guard warning + safe path, `build_tiled_split` label/source_id propagation, the **two leakage tests** (disjoint by construction + regression that detects bad construction), `health_check` boundary cases at three thresholds, and a one-step train smoke through V2's loop. Full `tests/soil/` → 51 passed.

**Ship criteria** (Cell 12 evaluates and prints "SHIP V3-TILING" / "KEEP V2"): texture top-1 strictly improves AND neither soil_type nor moisture drops more than 2 pts from V2's TTA test baseline (soil_type 89.92% / 0.851, moisture 95.76% / 0.958, texture 67.86% / 0.678). If V3-tiling doesn't clear the bar, V2 remains production and V3-tiling is preserved as a documented negative result for the paper's ablation table.

Expected Colab T4 wall-time: ~6–12 hours (same as V2). Epoch-1 health check makes the cost of a failure ~30 minutes, not 10+ hours.

---

## Phase 6 V3 prep: sequential transfer learning experiment

Notebook `notebooks/phase6_soil_training_v3.ipynb` generated (14 cells, built via `scripts/build_phase6_v3_notebook.py`). 3-stage sequential transfer learning:

- **Stage A** — Phantom-fs only (15 epochs, soil_type head; 5 frozen + 10 unfrozen)
- **Stage B** — + Sirajganj moisture (15 epochs, soil_type + moisture heads; all unfrozen)
- **Stage C** — full multi-task (20 epochs, all 3 heads; all unfrozen) → final V3 model

Hypothesis: a backbone warmed on Indian soil_type then moisture is more soil-aware than ImageNet pretraining alone, lifting the texture head. Expected texture gain +2–5 points. **This is an experiment** — if V3 doesn't clearly beat V2 it gets shelved as a documented negative result and V2 ships.

V3 reuses V1/V2 building blocks unchanged — `src/soil/train_v3.py` only composes them into the staged schedule (`SoilCheckpointManagerV3`, `build_stage_loader`, `train_stage_a/b/c` thin wrappers over V2's `train_one_epoch_v2`). Per-stage loss masking is **data-driven**: Stage A loads only Phantom-fs, so moisture/texture labels are all `-1` and their heads get zero gradient via V2's `ignore_index=-1` NaN guard; Stage B adds Sirajganj (texture still `-1`); Stage C adds texture. A fresh CosineAnnealingLR is created per stage (`T_max` = that stage's epoch count). All stages use V2's strong augmentation + Mixup/CutMix (p=0.3) + label smoothing 0.1; TTA (5 views) is used for the final test eval only.

V1 and V2 source files + notebooks are **untouched** (verified via git status). V3 pushes to NEW repo `ankit-iiitdmj/iks-soil-multitask-v3` (created on first Cell 8 run from Colab, not from this prompt session). V1 (`iks-soil-multitask`) and V2 (`iks-soil-multitask-v2`) repos stay intact for the paper's ablation.

V2 baseline (TTA) to beat: soil_type 89.92% / 0.851, moisture 95.76% / 0.958, texture 67.86% / 0.678. **Success criteria** (Cell 13 auto-checks + prints "SHIP V3" / "REVERT TO V2"): texture top-1 up ≥3 pts AND neither soil_type nor moisture drops >2 pts.

Tests: `pytest tests/soil/test_train_v3_smoke.py -q` → 5 passed (one-epoch smoke of each stage + A→B→C history chaining + V3 repo-id check). Full `tests/soil/` → 34 passed.

Expected Colab T4 wall-time: ~15–20 hours total across the 3 stages.

---

## Phase 6 V2 prep: augmentation-boosted retraining notebook ready

Notebook `notebooks/phase6_soil_training_v2.ipynb` generated (13 cells, built via `scripts/build_phase6_v2_notebook.py`). Goal: push the texture head from V1's **67.86% test top-1 / 0.670 macro F1** toward the 75–82% range without changing the locked EfficientNet-B0 backbone or 224×224 input size. soil_type (V1 89.08% / 0.818) and moisture (V1 88.98% / 0.890) may shift ±1–2 points because the backbone is shared — that's expected.

V2 adds:

- **Strong augmentation** (`src/soil/transforms_v2.py`): wider-scale RandomResizedCrop (0.7–1.0), VerticalFlip, Rotate ±30°, ShiftScaleRotate, GridDistortion/ElasticTransform (one-of, p=0.3), stronger ColorJitter, GaussianBlur/GaussNoise (one-of), CoarseDropout. Re-written against albumentations 2.x's new signatures (`RandomResizedCrop(size=...)`, `CoarseDropout(num_holes_range=...)`, `GaussNoise(std_range=...)`) with equivalent magnitudes to the spec's albumentations 1.x example.
- **Mixup + CutMix at batch level** (`src/soil/mixup.py`): `maybe_apply_mix(p=0.3, mixup_alpha=0.2, cutmix_alpha=1.0)` selects 50/50 between Mixup and CutMix when triggered. `mixed_loss()` blends per-head losses across the two label dicts.
- **Label smoothing 0.1** on cross-entropy in `compute_multitask_loss_smoothed` (`src/soil/train_v2.py`). Same `ignore_index=-1` NaN guard as V1.
- **Test-Time Augmentation** in Cell 12: `build_tta_views()` returns 5 deterministic albumentations Composes (original, HFlip, VFlip, Rot90, Rot270). `evaluate_per_task_tta()` averages logits across views before argmax.
- **40 epochs total** (V1 was 30) — heavier augmentation slows convergence. Warmup stays at 5 frozen-backbone epochs.

V1 source files are **untouched** — `src/soil/{transforms,train,model,dataset,__init__}.py` and `notebooks/phase6_soil_training.ipynb` are unchanged so the paper's ablation can compare V1 vs V2 with the exact V1 model on `ankit-iiitdmj/iks-soil-multitask`. V2 pushes to **`ankit-iiitdmj/iks-soil-multitask-v2`** (new private repo, created on first Cell 9 run from Colab — not from this prompt session).

Tests: `pytest tests/soil/test_mixup.py tests/soil/test_transforms_v2.py tests/soil/test_train_v2_smoke.py -q` → 13 passed. Covers Mixup/CutMix shapes + lam range, `maybe_apply_mix` p=0 / p=1 branches, `mixed_loss` blend math, TTA returns 5 distinct views, strong-aug is stochastic, label-smoothed loss + `train_one_epoch_v2` CPU smoke through Mixup path.

Expected Colab T4 wall-time: ~8–15 hours total (1–2 sessions), up from V1's 6–12 due to the extra 10 epochs and the small per-step overhead of the Mixup/CutMix collation. Resume-aware via HF Hub checkpoints just like V1.

---

## Phase 6 training notebook ready (B0 edition)

Notebook `notebooks/phase6_soil_training.ipynb` generated via `scripts/build_phase6_notebook.py`. Joint multi-task training on **EfficientNet-B0 at 224×224** per master plan §22 with 3 task heads: `soil_type` (7 classes), `moisture` (3 classes), `texture` (3 classes). Each head is `nn.Sequential(Dropout(0.3), Linear(1280, n_classes))`; total params 4,024,201 (4.0M backbone + 16,653 across the three heads — the headline "~5.3M B0 params" you see in the timm docs is for the original 1000-class ImageNet classifier, which we drop via `num_classes=0`).

Training code in `src/soil/` mirrors `src/disease/`:

- `src/soil/model.py` — `SoilMultiTaskClassifier` (timm B0 + 3 heads + `freeze_backbone()` / `unfreeze_backbone()`).
- `src/soil/train.py` — `TASK_WEIGHTS`, `compute_multitask_loss` (NaN-guarded ignore_index=-1), `train_one_epoch`, `evaluate_per_task`, `SoilCheckpointManager`, `auto_batch_size` (returns 64/32/16 by VRAM).
- `src/soil/transforms.py` — `build_soil_train_aug` (RandomResizedCrop + HFlip + Rotate ±15 + mild ColorJitter + Normalize) and `build_soil_eval_aug` (Resize 256 → CenterCrop 224).
- `src/soil/__init__.py` — exports the full Phase 6 API alongside the existing Phase 4 helpers.

Per-sample loss masking: each batch sample supervises exactly one head; the other two receive `-1` and are ignored by `F.cross_entropy(ignore_index=-1)`. When **all** samples in a batch carry `-1` for a head, cross-entropy normally returns NaN; the helper substitutes a graph-consistent zero so backward still produces a zero gradient on that head without poisoning the total loss.

12-cell notebook covers: setup → HF auth → GPU + auto-batch → load 3 HF datasets → transforms + ConcatDataset train loader + 3 per-task val/test loaders → model build → optimizer + scheduler + scaler + `SoilCheckpointManager` with resume → 30-epoch loop (5 frozen + 25 full unfrozen, per-task losses + per-task val metrics logged per epoch, latest + best checkpoints pushed to `ankit-iiitdmj/iks-soil-multitask` HF Hub repo) → held-out test-set evaluation (mirrors Phase 5 fix, separate `eval_metrics_test.json` file).

Expected wall-time on Colab T4: 6–12 h total. Resume-aware so 1–2 sessions work. HF Hub model repo created on first Cell 9 run from Colab (not in this prompt session).

Tests: `pytest tests/soil/` → 16 passed (3 new: `test_model.py`, `test_loss_masking.py`, `test_train_eval_smoke.py` — covering construction, forward shapes, freeze toggle, loss NaN-guard, end-to-end train+eval CPU smoke).

---

## Phase 6 prep: soil data uploaded to HF Hub

Three private dataset repos created on Hugging Face Hub:

- `ankit-iiitdmj/iks-soil-phantomfs` (soil_type, 7 classes, 1,188 images) — 6 parquet shards, 397 MB
- `ankit-iiitdmj/iks-soil-sirajganj-moisture` (moisture_appearance, 3 classes, 1,177 images) — 3 shards, 188 MB
- `ankit-iiitdmj/iks-soil-texture-irsid-vit` (texture, 3 USDA-collapsed classes, 279 images: 16 IRSID + 263 VIT) — 3 shards, 27 MB

Splits: stratified 80/10/10, seed=42 (`sklearn.train_test_split`). Phantom-fs train=951/val=119/test=119, Sirajganj train=941/val=118/test=118, Texture train=223/val=28/test=28. Each parquet row carries the image (HF `Image()` column), `label_idx`, `class_name`, and a `source` column ('phantomfs' / 'sirajganj' / 'irsid' / 'vit') for ablation.

Sirajganj and texture were pre-resized to max-dim 768 (JPEG q=90) before encoding — full-res phone photos pushed sirajganj's single parquet shard to ~450 MB which reliably crashed the LFS upload mid-transfer across three attempts. Phantom-fs stayed at native resolution because its first upload (full-res) had already succeeded by the time the resize policy was added; training-time pipeline crops to 224×224 either way.

Combined channel norm stats (`configs/data/soil_norm.yaml`) computed over the union of 2,114 train images at 224×224: `mean=[0.535, 0.459, 0.400]`, `std=[0.216, 0.200, 0.210]`.

Multi-task training will use per-sample loss masking — each row supervises exactly one head; the other two get `-1` (ignored by `CrossEntropyLoss`). Helper `src.soil.dataset.build_multitask_labels()` produces the `{soil_type, moisture_appearance, texture}` dict per sample. Class index configs at `configs/data/soil_{soil_type,moisture,texture}_classes.yaml`.

Tests: `pytest tests/data/test_soil_hf_datasets.py` → 12 passed.

---

## VIT texture dataset integration (Phase 5/6 boundary)

Added latha-soil (Reddy & Gopinath, Nature Sci. Rep. 2025, doi:10.1038/s41598-025-17384-5) as a supplementary texture-axis dataset alongside the IRSID Kaggle mirror. Local-only — no Hugging Face Hub push in this session (deferred to Phase 6 prep). Paper claims 4,000 images; the public GitHub release at `https://github.com/phd-latha/latha-soil` (commit `14a1fe2`) contains **263 images across 7 classes**: see `data/soil/vit_texture/INTEGRATION_AUDIT.json` for the full breakdown.

Class counts after canonicalisation: clayey_soils 40, loamy_sand_soil 40, loamy_soil 39, sandy_clay_soil 33, sandy_loam 36, sandy_soil 40, silt_soil 35. 0 files routed to `_review/`; 0 PIL-rejected.

Both datasets are kept on disk as separate directories (`data/soil/vit_texture/` and `data/soil/irsid/`); Phase 6 training code will combine them via PyTorch `ConcatDataset`, not by filesystem merging. The new `vit_texture:` section in `configs/data/soil_texture_label_mapping.yaml` maps each class to coarse / fine / mixed using the same USDA-soil-triangle logic as the existing IRSID block.

Decision: integrate as-is; email VIT authors in parallel asking whether the full 4,000-image release is hosted elsewhere.

---

## Phase 4 fix (post-Weeks 14–15) — Reconciliation with finalised scope

- OLID I: switched source from Zenodo (19 archives) to Kaggle `raiaone/olid-i` (single zip) and downloaded the **full 4,749 images / 23 multi-label classes** (was smoke-sample 83/3). `_labels_for()` updated to split compound symptoms like `bottle_gourd__JAS_MIT`.
- Sirajganj 2025 added as net-new (Mendeley DOI 10.17632/skcc44yvvg.2): 1,177 images / 3 classes (dry/moderate/wet) supervising the `moisture_appearance` head.
- Soil module heads pinned to 3: `soil_type` + `moisture_appearance` + `texture`. Dropped `surface` and `cover` per the supervisor-signed-off soil-parameter coverage audit. `texture` survives via the IRSID → coarse/fine/mixed mapping in `configs/data/soil_texture_label_mapping.yaml`.
- Phantom-fs 7-class verified: Alluvial / Arid / Black / Laterite / Mountain / Red / Yellow (the Phase 4 prompt's "Clay" and "Peat" do not exist upstream).
- PlantDoc 28th class (`Tomato two spotted spider mites leaf`) found to be a **vestigial 2-image folder**, documented, not modified.
- `requirements.txt` gained matplotlib / jupyter / nbformat / iterative-stratification / requests / kaggle.
- `results/dataset_stats.md` (228 lines) and `notebooks/dataset_eda.ipynb` (with a real OLID 23×23 multi-label co-occurrence heatmap) regenerated and end-to-end-executed.
- New `PHASE4_SUMMARY.md` at the repo root supersedes the previous one.
- Total disk: 30 GB (over the 25 GB envelope because Sirajganj v2 grew to 4.49 GB vs the prompt's ~500 MB estimate).

---

## Phase 4 (Weeks 14–15) — Dataset Acquisition & Preprocessing

- 6 datasets acquired: PlantVillage, PlantDoc, Paddy Doctor, Phantom-fs Soil, IRSID, OLID I (smoke sample)
- Splits generated: 5 standard 80/10/10 stratified splits + 1 cross-region soil split (Phantom-fs train, IRSID test)
- Normalisation stats computed per dataset (configs/data/*_norm.yaml)
- Augmentation pipelines defined (disease modest, soil heavier, causation geometric-only)
- Dataset classes implemented (JSONIndexedImageDataset + MultiLabelImageDataset, factory functions per dataset)
- Validation: 0 corrupt files across 185,735 images scanned
- Total disk: 5.4 GB (well under the 20 GB cap)
- TODO: full OLID I (~14 GB across 19 Zenodo archives) deferred to Phase 11 — flip `OLID_FULL_DOWNLOAD = True` in `scripts/download_olid_i.py` and re-run when C5 evaluation begins

---

## Week 2 (continued) — PDF-alignment cleanup

- Removed [ADDED] engineering hygiene (pre-commit, GitHub Actions CI, pyproject.toml + tool configs, decisions/, session reports)
- Fixed paper/thesis nesting per §41
- Added §42 references.bib, §43 BACKUP.md, §44 weekly + monthly journal templates, notebooks/00_environment_check.ipynb
- Rewrote requirements.txt to track §22 exactly
- Rewrote environment.yml to mirror requirements.txt
- All tests still passing post-cleanup

### Post-cleanup repository tree (`find . -maxdepth 2 -type d`)

```
.
./configs
./configs/disease
./configs/eval
./configs/integration
./configs/rag
./configs/soil
./corpus
./corpus/chunks
./corpus/cleaned
./corpus/raw
./corpus/vector_db
./data
./data/plant_disease
./data/soil
./data/splits
./demo
./models
./notebooks
./notes
./notes/cv
./notes/iks
./notes/rag
./notes/xai
./paper
./research_journal
./research_journal/daily
./research_journal/monthly
./research_journal/weekly
./results
./results/figures
./results/logs
./scripts
./src
./src/disease
./src/eval
./src/explain
./src/integration
./src/rag
./src/soil
./src/utils
./tests
./tests/disease
./tests/eval
./tests/explain
./tests/integration
./tests/rag
./tests/soil
./tests/utils
./thesis
```

Matches §41 exactly; extras (`notes/`, `tests/`, `research_journal/{daily,weekly,monthly}`) are all `[PDF-implied]` or `[PDF §44]`.

---

## Week 1 — Project Setup
**Dates:** May 15 - May 21, 2026

### Completed Tasks
- [x] Repository initialized on GitHub
- [x] Complete folder structure created with all subdirectories
- [x] `requirements.txt` written with all dependencies (PyTorch, transformers, RAG, evaluation tools)
- [x] `environment.yml` created for Conda environment (`iks-agri`)
- [x] `.gitignore` configured (data/, models/, vectors, pycache, etc.)
- [x] README.md written with overview, architecture, setup, and references
- [x] Configuration files created (`disease_config.yaml`, `soil_config.yaml`, `rag_config.yaml`)
- [x] Python module structure initialized (`__init__.py` in all src/ subdirectories)
- [x] Skeleton implementations: logger.py, config.py, model stubs
- [x] Streamlit app skeleton created (`demo/app.py`)
- [x] Environment check notebook created (`notebooks/00_environment_check.ipynb`)

### Blockers / Issues
- None encountered during setup

### Notes
- Seed=42 set globally for reproducibility
- All YAML configs include detailed annotations for future reference
- Soil model includes critical warnings about visual-only analysis (no NPK/pH prediction)

### Next Week Goals
- [ ] Set up development environment (create conda env, verify imports)
- [ ] Literature review: CV basics (transfer learning, ResNet, fine-tuning techniques)
- [ ] Literature review: RAG fundamentals (embeddings, vector databases, chunking strategies)
- [ ] Literature review: XAI techniques (Grad-CAM, attention mechanisms, interpretability)
- [ ] Schedule supervisor meeting with Dr. Akshay Pandey
- [ ] Identify and download PlantVillage and Soil datasets
- [ ] Sketch initial experiment plan for Phase 2 (disease module)

---

## Week 2 — Foundation Infrastructure
**Dates:** May 20 - May 26, 2026

### Completed Tasks
- [x] §1 Locked-stack `pyproject.toml`, regenerated `requirements.txt`,
      `requirements-dev.txt`, `.python-version`, `INSTALL.md`.
- [x] §2 Reproducibility utilities in `src/utils/`: `set_global_seed`,
      project paths, stdlib logging, Pydantic v2 `BaseConfig`. Unit
      tests cover seeding reproducibility, strict-extra config validation,
      directory creation, logger handler attachment.
- [x] §3 Pydantic config schemas + `configs/<module>/default.yaml` for
      disease, soil, rag, integration, eval. Soil schema enforces
      disallowed chemical outputs (guardrail #2). RAG config locks the
      Llama-3.1-8B / 4-bit / BGE choices. Integration config flags
      `require_causal_context` (contribution C5).
- [x] §4 Module skeletons for disease, soil, rag, integration, explain,
      eval — every public class, dataclass, and function has a
      NumPy-style docstring and a `NotImplementedError("Phase X — Week Y")`
      pointer. `src/rag/prompts.py` ships a working citation-enforcing
      prompt template and renderer. `src/eval/citation_verification.py`
      has a working extractor + minimal verifier.
- [x] §5 Testing infrastructure: `tests/` mirrors `src/`, shared
      `conftest.py` with `tmp_corpus_dir`, `seeded_rng`,
      `tiny_dummy_image`, `sample_retrieved_chunks` fixtures. Each
      module has an instantiation smoke test.
- [x] §6 Code quality + CI: pre-commit (ruff, ruff-format, black, mypy,
      file-hygiene hooks), GitHub Actions matrix on Python 3.11 + 3.12.
      Ruff/black/mypy/pytest config lives in `pyproject.toml`.
- [x] §7 Notes templates under `notes/{cv,rag,xai,iks}/` with the
      standard skeleton (Key concepts / Papers read / Tricky bits / Open
      questions / Code I want to try).
- [x] §8 Research ops: this Week 2 entry, `literature_tracker.csv`
      (empty), `research_journal/` with a first daily entry,
      `decisions/0001..0003.md` ADRs.

### Blockers / Issues
- None for the scaffolding. Running the full `pytest` / `pre-commit`
  loop locally was not attempted in this session — see
  `WEEK2_SUMMARY.md` for what to verify on the M.Tech workstation.

### Notes
- README, requirements.txt, and environment.yml from Week 1 referenced
  ResNet50 + LangChain. Both have been brought in line with the locked
  stack (EfficientNet-B4/B0, plain-Python RAG).
- ADR-0001 (EfficientNet backbones), ADR-0002 (Pydantic over Hydra),
  ADR-0003 (no LangChain in RAG) capture the reasoning.

### Next Week Goals
- [ ] Run `pip install -e ".[dev]"` on the workstation and verify
      `pytest -q`, `ruff check .`, `black --check .`, `mypy src/`,
      `pre-commit run --all-files` are all green.
- [ ] Start filling `notes/cv/transfer_learning.md` and `notes/cv/efficientnet.md`.
- [ ] Begin literature_tracker.csv (target: 20 rows by end of Week 3).
- [ ] Supervisor meeting with Dr. Pandey to walk through ADR-0001..0003.

---

## Phase Milestones

### Phase 1: Foundation (Weeks 1-3)
- **Goal:** Establish development environment and foundational knowledge
- **Key Activities:**
  - Environment setup and dependency validation ✅ (Week 1)
  - Literature review on CV, RAG, and XAI
  - Supervisor alignment on approach and timeline
  - Dataset acquisition and initial exploration
- **Deliverable:** Full working development environment + knowledge summary document

### Phase 2: Disease Detection Module (Weeks 4-7)
- **Goal:** Implement and validate plant disease classification
- **Key Activities:**
  - PlantVillage dataset exploration and preprocessing
  - ResNet50 fine-tuning on disease dataset
  - Grad-CAM integration for model explainability
  - Validation metrics and baseline performance
- **Deliverable:** Trained disease model (>90% accuracy on test set) + Grad-CAM visualization

### Phase 3: Soil Analysis Module (Weeks 8-10)
- **Goal:** Implement multi-task soil visual analysis
- **Key Activities:**
  - Soil dataset preparation and balancing
  - Multi-task architecture design (soil type + texture + surface + moisture)
  - Training pipeline with multi-task loss
  - Performance baseline on each task
- **Important:** Model predicts ONLY visual attributes; does NOT claim NPK/pH/fertility
- **Deliverable:** Trained multi-task soil model + per-task evaluation metrics

### Phase 4: RAG Pipeline (Weeks 11-14)
- **Goal:** Build hybrid retrieval system over classical agricultural texts
- **Key Activities:**
  - Digitized text preprocessing (Vrikshayurveda, Krishi Parashara, Upavanavinoda)
  - Sentence-window chunking with semantic tagging
  - Embedding model selection and fine-tuning
  - Dense + BM25 hybrid retrieval implementation
  - LLM integration (Llama-3.1-8B) with prompt engineering
- **Deliverable:** Functional RAG pipeline + retrieval baseline evaluation

### Phase 5: Integration & Optimization (Weeks 15-17)
- **Goal:** Unify all components and optimize for real-time inference
- **Key Activities:**
  - End-to-end system integration
  - Latency profiling and optimization
  - Streamlit web interface refinement
  - Error handling and edge case management
- **Deliverable:** Deployable web app with <5s total inference time

### Phase 6: Evaluation & Ablation (Weeks 18-20)
- **Goal:** Rigorous evaluation using established metrics
- **Key Activities:**
  - RAGAS evaluation framework (Faithfulness, Relevance, Context Recall)
  - Expert annotation of recommendations for groundtruth
  - Ablation studies (disease only vs. soil+disease vs. full system)
  - Comparison with template-based baselines
- **Deliverable:** Comprehensive evaluation report with tables/plots

### Phase 7: Thesis Writing & Defense (Weeks 21-24)
- **Goal:** Document research and prepare for defense
- **Key Activities:**
  - Literature review chapter
  - Methodology chapter (architecture, datasets, training procedures)
  - Results chapter (with tables, confusion matrices, case studies)
  - Discussion and conclusions
  - Final revisions and formatting
- **Deliverable:** Complete thesis manuscript

---

## Known Constraints & Notes

1. **Soil Module Limitation:** The multi-task soil classifier predicts ONLY visually observable attributes (soil type from color/texture, surface condition, moisture appearance). It CANNOT predict NPK, pH, or soil fertility—these require lab testing. This is documented in README.md and all relevant code files.

2. **Classical Texts:** Vrikshayurveda, Krishi Parashara, and Upavanavinoda form the grounding corpus. Digitization and cleaning are prerequisites for RAG.

3. **Reproducibility:** All random seeds set to 42; dependency versions pinned in requirements.txt for reproducibility across systems.

4. **Supervisor:** Dr. Akshay Pandey, CSE Department, IIITDM Jabalpur

5. **Disclaimer:** This is a research prototype. Not for production use without expert validation.

---

## Communication Log

| Date | Contact | Topic | Outcome |
|------|---------|-------|---------|
| (TBD) | Dr. Akshay Pandey | Project kickoff meeting | - |

---

**Last Updated:** May 15, 2026
**Status:** Week 1 — Setup Phase Complete ✅
