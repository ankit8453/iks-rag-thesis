# Weekly Progress Log

## Overview
This document tracks weekly progress on the IKS Agricultural Advisory System thesis project. Each week includes completed tasks, blockers, and goals for the next week.

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
