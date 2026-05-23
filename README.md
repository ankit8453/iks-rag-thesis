# An IKS-Grounded Multimodal Agricultural Advisory System

**Joint Disease and Soil Analysis with Retrieval-Augmented Generation over Classical Indian Agricultural Texts**

M.Tech Thesis · IIITDM Jabalpur · Department of Computer Science & Engineering
Supervisor: **Dr. Akshay Pandey**

> **Project status:** Phase 4 complete — datasets ready for Phase 5 (disease module training). See [`progress.md`](progress.md) and [`PHASE4_SUMMARY.md`](PHASE4_SUMMARY.md).

---

## Overview

A multimodal system that takes a plant/leaf photograph and a soil
photograph, runs each through a dedicated vision model, and feeds the
joint context plus a user-supplied causal hypothesis into a retrieval-
augmented generator grounded in classical Indian agricultural texts
(Vrikshayurveda, Krishi Parashara, Upavanavinoda, plus three optional
extensions).

**Novel contributions** (formal statement in the thesis intro):

- **C1** First chunked + metadata-tagged digital corpus of these texts.
- **C2** Joint disease–soil context module with three ablated integration
  strategies (template, LLM-mediated, multimodal embedding).
- **C3** Faithfulness-aware RAG evaluation combining RAGAS with domain
  experts.
- **C4** First quantitative hallucination measurement on IKS sources.
- **C5** Cause-conditional retrieval — the system retrieves *treatment
  given* a user-provided causal context; it does **not** infer cause
  from images.

---

## Disclaimer

This is a research prototype for academic exploration. Recommendations
derive from classical texts and modern models and are **not** substitutes
for professional agronomic advice, soil testing, or formal disease
diagnosis. Consult qualified agricultural experts for real-world farming
decisions.

---

## Architecture

```
User input: (plant photo, soil photo, crop, causal context)
        ↓
   ┌────────────────────────┐    ┌────────────────────────┐
   │ Disease (EffNet-B4)    │    │ Soil (EffNet-B0, MT)   │
   │  ↳ DiseasePrediction   │    │  ↳ SoilPrediction      │
   └────────────────────────┘    └────────────────────────┘
                          ↓ joint
        ┌─────────────────────────────────────────┐
        │   Integration strategy (one of 3)       │
        │   template │ llm_mediated │ multimodal  │
        └─────────────────────────────────────────┘
                          ↓ query (+ CausalContext)
        ┌─────────────────────────────────────────┐
        │  Hybrid retrieval                       │
        │  dense (BGE) + BM25 → cross-encoder     │
        │  rerank (BGE-reranker)                  │
        └─────────────────────────────────────────┘
                          ↓ top-k chunks
        ┌─────────────────────────────────────────┐
        │  Llama-3.1-8B-Instruct (4-bit)          │
        │  Cited answer + refusal mode            │
        └─────────────────────────────────────────┘
                          ↓
   Advisory response with chunk-ID citations + Grad-CAM overlay
```

## Locked Stack

| Component               | Choice                                        |
|-------------------------|-----------------------------------------------|
| Python                  | `>=3.11`                                      |
| Deep learning           | PyTorch 2.x + `timm`                          |
| Disease backbone        | **EfficientNet-B4** (PlantVillage, 38 cls)    |
| Soil backbone           | **EfficientNet-B0** (multi-task, visual only) |
| LLM                     | `meta-llama/Llama-3.1-8B-Instruct`, 4-bit     |
| Embeddings              | `BAAI/bge-large-en-v1.5`                      |
| Reranker                | `BAAI/bge-reranker-base`                      |
| Sparse retrieval        | `rank-bm25`                                   |
| Vector store            | `chromadb`                                    |
| RAG orchestration       | Plain Python wrappers (LangChain present as optional scaffolding only) |
| RAG evaluation          | `ragas` + expert annotation                   |
| CV metrics              | `torchmetrics` + per-class report (guardrail) |
| Explainability          | `pytorch-grad-cam`, `captum`                  |
| Config                  | `pydantic` v2 + YAML                          |
| Test runner             | `pytest`                                      |

## Repository layout (per master reference §41)

```
.
├── README.md                   # this file
├── progress.md                 # weekly progress log
├── BACKUP.md                   # 3-2-1 backup strategy (§43)
├── requirements.txt            # pip dependencies (§22)
├── environment.yml             # conda mirror (pulls from requirements.txt)
├── pytest.ini                  # test runner config
├── .gitignore
├── literature_tracker.csv      # paper tracking (§42)
├── references.bib              # BibTeX export from Zotero (§42)
│
├── src/
│   ├── utils/                  # seeding, paths, logging, config
│   ├── disease/                # EfficientNet-B4 + Grad-CAM
│   ├── soil/                   # EfficientNet-B0 multi-task (VISUAL ONLY)
│   ├── rag/                    # corpus, chunker, retrievers, reranker, generator
│   ├── integration/            # joint context + 3 ablation strategies
│   ├── explain/                # Grad-CAM + chunk highlighting
│   └── eval/                   # per-class CV metrics, RAGAS, citation verify
│
├── tests/                      # mirrors src/ + smoke tests + utils tests
├── configs/                    # default.yaml per module
├── corpus/{raw,cleaned,chunks,vector_db}/   # IKS classical-text corpus
├── data/{plant_disease,soil,splits}/        # image datasets (gitignored)
├── models/                     # trained checkpoints (gitignored)
├── results/{logs,figures}/     # outputs (gitignored except .gitkeep)
├── notebooks/                  # 00_environment_check.ipynb and exploratory work
├── scripts/                    # training and eval entry points (Phase 4–7)
├── demo/                       # Streamlit application (Phase 9)
├── notes/{cv,rag,xai,iks}/     # foundation-learning notes (Phase 1)
├── research_journal/           # daily/weekly/monthly entries (§44)
├── paper/                      # LaTeX conference paper (Phase 13)
└── thesis/                     # LaTeX M.Tech thesis (Phase 14)
```

## Setup

```bash
git clone https://github.com/ankit8453/iks-rag-thesis.git
cd iks-rag-thesis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/00_environment_check.ipynb
```

Conda alternative: `conda env create -f environment.yml` (mirrors `requirements.txt`).

## Datasets

Datasets and model weights are **not** in the repo.

1. **PlantVillage** — https://github.com/spMohanty/PlantVillage-Dataset →
   unpack to `data/plant_disease/PlantVillage/`.
2. **Soil Type Image Classification (Kaggle)** —
   https://www.kaggle.com/datasets/abdulqayyum/soil-types-image-classification
   → unpack to `data/soil/SoilTypes/`.
3. **Classical texts** — place digitised translations in `corpus/raw/`;
   preprocessing scripts populate `corpus/cleaned/` and `corpus/chunks/`.

## Timeline (40 weeks)

| Phase   | Weeks   | Focus                                                                 |
|---------|---------|-----------------------------------------------------------------------|
| Phase 0 | 1       | Repo scaffolding (done)                                              |
| Phase 1 | 2–4     | Foundation learning (CV / RAG / XAI notes); **infra built in Week 2** |
| Phase 2 | 5–8     | Literature review (240 papers in `literature_tracker.csv`)            |
| Phase 3 | 9–11    | Corpus prep — Vrikshayurveda + Krishi Parashara + Upavanavinoda       |
| Phase 4 | 12–15   | RAG pipeline                                                          |
| Phase 5 | 16–19   | Plant disease module                                                  |
| Phase 6 | 19–21   | Soil module + cross-region validation                                 |
| Phase 7 | 22–24   | Integration module + three ablations                                  |
| Phase 8 | 25–27   | Explainability + demo                                                 |
| Phase 9 | 28–30   | Evaluation (RAGAS + expert annotation)                                |
| Phase 10| 31–40   | Thesis writing + defence                                              |

Detailed plan and risks: `progress.md`, `research_journal/monthly/`.

## Hard guardrails

These are visible in code and enforced by tests:

1. **Reproducibility** — all stochastic scripts call
   `src.utils.set_global_seed(seed)` (defaults to 42). Verified in
   `tests/utils/test_seeding.py`.
2. **Soil module is visual-only** — `SoilConfig.disallowed_outputs`
   blocks NPK / pH / fertility / organic-matter / chemical-composition
   keys. Verified in `tests/soil/test_smoke.py`.
3. **Per-class metrics, not just accuracy** — every CV report includes
   per-class precision/recall/F1 + confusion matrix. Shape enforced by
   `ClassificationReport` in `src/eval/cv_metrics.py`.
4. **Cross-region validation for soil** — supported via
   `held_out_regions` in `SoilTypeDataset`.
5. **No fabricated citations** — the RAG prompt template instructs the
   LLM to cite chunk IDs, and `src/eval/citation_verification.py`
   verifies the cited IDs actually appeared in retrieved context.

## Backup

See [`BACKUP.md`](BACKUP.md) for the 3-2-1 strategy (per master reference §43).

## Citation

```bibtex
@mastersthesis{thesis_iks_agricultural,
  author    = {Ankit Pawar},
  title     = {An IKS-Grounded Multimodal Agricultural Advisory System:
               Joint Disease and Soil Analysis with Retrieval-Augmented
               Generation over Classical Indian Agricultural Texts},
  school    = {IIITDM Jabalpur},
  year      = {2026},
  advisor   = {Dr. Akshay Pandey}
}
```

## Contact

- **Student:** Ankit Pawar — M.Tech CSE, IIITDM Jabalpur
- **Advisor:** Dr. Akshay Pandey — CSE Department, IIITDM Jabalpur
- **Repo:** https://github.com/ankit8453/iks-rag-thesis
