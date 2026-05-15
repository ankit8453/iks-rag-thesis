# Weekly Progress Log

## Overview
This document tracks weekly progress on the IKS Agricultural Advisory System thesis project. Each week includes completed tasks, blockers, and goals for the next week.

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

## Week 2 — [To be filled]
**Dates:** May 22 - May 28, 2026

### Completed Tasks
- [ ]

### Blockers / Issues
- 

### Notes
- 

### Next Week Goals
- [ ]

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
