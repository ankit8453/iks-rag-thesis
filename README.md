# An IKS-Grounded Multimodal Agricultural Advisory System: Joint Disease and Soil Analysis with Retrieval-Augmented Generation over Classical Indian Agricultural Texts

**M.Tech Thesis | IIITDM Jabalpur | Department of Computer Science & Engineering**

**Supervised by:** Dr. Akshay Pandey

---

## Overview

This research project presents an **Intelligent Knowledge System (IKS)-grounded multimodal agricultural advisory system** that integrates computer vision and retrieval-augmented generation (RAG) to provide culturally-grounded agricultural recommendations. The system accepts two input images—a plant/leaf photograph and a soil sample photograph—and produces a unified advisory response grounded in classical Indian agricultural texts.

**System Pipeline:**
1. **Disease Detection Module:** Classifies plant diseases from leaf images using fine-tuned ResNet50 (PlantVillage dataset, 38 classes)
2. **Soil Analysis Module:** Predicts soil type and visual attributes (texture, surface condition, moisture appearance) using multi-task ResNet50 (does NOT predict NPK/pH/fertility)
3. **Retrieval-Augmented Generation (RAG):** Retrieves relevant treatment protocols from classical texts (Vrikshayurveda, Krishi Parashara, Upavanavinoda) using dense + sparse hybrid retrieval
4. **Recommendation Generation:** Synthesizes disease-specific and soil-appropriate organic treatment protocols with source citations

**Key Features:**
- Grounded in classical Indian agricultural knowledge (Vrikshayurveda, Krishi Parashara, Upavanavinoda)
- Explainable AI using Grad-CAM for disease prediction transparency
- Multi-task soil analysis (type + texture + surface + moisture)
- Hybrid retrieval (dense embedding + BM25 sparse retrieval) for robustness
- Source citations for all recommendations
- Streamlit web interface for end-user interaction

---

## ⚠️ Important Disclaimer

**This system is a research prototype designed for academic exploration only.** Recommendations are derived from classical texts and modern models and are **NOT substitutes for professional agronomic advice, soil testing, or formal disease diagnosis.** Users must consult qualified agricultural experts for real-world farming decisions. This project is part of an M.Tech thesis and should not be used as a standalone advisory tool in production settings.

---

## Architecture

### Component Overview

```
User Input (Plant Photo + Soil Photo)
         ↓
    ┌────────────────────────────────────┐
    │  Disease Detection (ResNet50 CNN)  │ → Disease Class + Confidence
    └────────────────────────────────────┘
    ┌────────────────────────────────────┐
    │  Soil Analysis (ResNet50 Multi-Task)│ → Soil Type, Texture, Surface, Moisture
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │   RAG Pipeline (Hybrid Retrieval)   │
    │   - Query embedding                 │
    │   - Dense retrieval (MiniLM-L6)    │
    │   - Sparse retrieval (BM25)        │
    │   - Re-ranking (cross-encoder)     │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │   LLM Generation (Llama-3.1-8B)    │
    │   + Source Citation                │
    └────────────────────────────────────┘
         ↓
    Unified Advisory Response (Grounded in IKS)
```

### Module Breakdown

- **`src/disease/`**: Plant disease classification module
  - `model.py`: ResNet50-based classifier
  - `dataset.py`: PlantVillage dataset loader
  - `train.py`: Training pipeline with early stopping
  - `inference.py`: Inference wrapper

- **`src/soil/`**: Multi-task soil visual analysis
  - `model.py`: ResNet50 with 4 task heads (soil type, texture, surface, moisture)
  - `dataset.py`: Custom soil dataset loader
  - `train.py`: Multi-task training loop
  - `inference.py`: Inference with per-task outputs

- **`src/rag/`**: Retrieval-Augmented Generation pipeline
  - `chunker.py`: Sentence-window chunking with semantic tagging
  - `embedder.py`: Embedding wrapper (sentence-transformers)
  - `retriever.py`: Hybrid retrieval (dense + BM25)
  - `generator.py`: LLM generation with prompt engineering

- **`src/explain/`**: Explainability components
  - `gradcam.py`: Grad-CAM visualization for disease detection
  - `chunk_viz.py`: Visualize retrieved chunks and re-ranking scores

- **`src/eval/`**: Evaluation framework
  - `cv_metrics.py`: Classification metrics (accuracy, F1, confusion matrices)
  - `ragas_eval.py`: RAG evaluation (RAGAS framework: Faithfulness, Relevance, Context Recall)
  - `expert_annotation.py`: Groundtruth annotation collector

- **`src/integration/`**: Strategy-based integration layer
  - `template_strategy.py`: Template-based recommendation generation
  - `llm_strategy.py`: LLM-based recommendation generation
  - `embedding_strategy.py`: Embedding strategy interface for swappability

- **`src/utils/`**: Utilities
  - `logger.py`: Structured logging with file + console handlers
  - `config.py`: YAML configuration loader

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA 12.1 (recommended for GPU acceleration) or CPU-only
- 16+ GB RAM recommended

### Installation

**Option 1: Using Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate iks-agri
```

**Option 2: Using pip + venv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Installation

```bash
cd notebooks
jupyter notebook 00_environment_check.ipynb
# Run all cells to verify dependencies
```

### Download Datasets

1. **PlantVillage (Plant Disease):**
   - Download from [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
   - Extract to `data/plant_disease/`

2. **Soil Type Images:**
   - Download from [Soil Type Image Classification - Kaggle](https://www.kaggle.com/datasets/abdulqayyum/soil-types-image-classification)
   - Extract to `data/soil/`

3. **Classical Texts (IKS Corpus):**
   - Place digitized texts in `corpus/raw/`
   - Preprocessing scripts will clean and chunk them

---

## Usage

### 1. Train Disease Detection Model

```bash
python scripts/train_disease.py --config configs/disease_config.yaml
```

### 2. Train Soil Analysis Model

```bash
python scripts/train_soil.py --config configs/soil_config.yaml
```

### 3. Prepare RAG Corpus

```python
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder

# Chunk classical texts
chunker = TextChunker(config)
chunks = chunker.process_corpus(corpus_dir="corpus/raw/", output_dir="corpus/chunks/")

# Build vector database
embedder = Embedder(config)
embedder.build_db(chunks, persist_dir="corpus/vector_db/")
```

### 4. Run Web Interface

```bash
streamlit run demo/app.py
```

Then open `http://localhost:8501` in your browser.

### 5. Evaluate RAG System

```bash
python scripts/run_rag_eval.py --config configs/rag_config.yaml
```

---

## Project Structure

```
.
├── README.md                          # This file
├── progress.md                        # Weekly progress tracker
├── requirements.txt                   # Pip dependencies
├── environment.yml                    # Conda environment
├── .gitignore                         # Git ignore rules
│
├── data/                              # Datasets (not in repo)
│   ├── plant_disease/                 # PlantVillage dataset
│   ├── soil/                          # Soil type images
│   └── splits/                        # Train/val/test splits
│
├── corpus/                            # Classical texts & RAG resources
│   ├── raw/                           # Raw digitized texts
│   ├── cleaned/                       # Preprocessed texts
│   ├── chunks/                        # Text chunks
│   └── vector_db/                     # ChromaDB vector store
│
├── src/                               # Main source code
│   ├── disease/                       # Disease detection module
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── soil/                          # Soil analysis module
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── rag/                           # RAG pipeline
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── integration/                   # Integration strategies
│   ├── explain/                       # Explainability
│   ├── eval/                          # Evaluation
│   └── utils/                         # Utilities
│
├── notebooks/                         # Jupyter notebooks
│   ├── 00_environment_check.ipynb
│   └── README.md
│
├── scripts/                           # Training & evaluation scripts
│   ├── train_disease.py
│   ├── train_soil.py
│   └── run_rag_eval.py
│
├── configs/                           # YAML configurations
│   ├── disease_config.yaml
│   ├── soil_config.yaml
│   └── rag_config.yaml
│
├── results/                           # Results & logs (not in repo)
│   └── .gitkeep
│
├── models/                            # Trained models (not in repo)
│   └── .gitkeep
│
├── demo/                              # Streamlit web app
│   └── app.py
│
└── paper/                             # Thesis writing
    └── thesis/
```

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Foundation** | Weeks 1-3 | Repo setup, literature review (CV, RAG, XAI basics), environment validation |
| **Phase 2: Disease Module** | Weeks 4-7 | PlantVillage preprocessing, ResNet50 fine-tuning, Grad-CAM integration |
| **Phase 3: Soil Module** | Weeks 8-10 | Soil dataset preparation, multi-task architecture, baseline training |
| **Phase 4: RAG Pipeline** | Weeks 11-14 | Corpus preprocessing, chunking, embedding, hybrid retrieval, LLM integration |
| **Phase 5: Integration & Optimization** | Weeks 15-17 | System integration, latency optimization, Streamlit UI |
| **Phase 6: Evaluation & Refinement** | Weeks 18-20 | RAGAS evaluation, expert annotation, ablation studies |
| **Phase 7: Thesis Writing** | Weeks 21-24 | Paper drafting, results compilation, final revisions |

---

## Citation

If you use this project in your research, please cite:

```bibtex
@mastersthesis{thesis_iks_agricultural,
  author = {Ankit Pawar},
  title = {An IKS-Grounded Multimodal Agricultural Advisory System: Joint Disease and Soil Analysis with Retrieval-Augmented Generation over Classical Indian Agricultural Texts},
  school = {IIITDM Jabalpur},
  year = {2024-2025},
  advisor = {Dr. Akshay Pandey}
}
```

---

## References

- **Classical Texts:**
  - Surapala. *Vrikshayurveda* (translated by Nalini Sadhale). Asian Agri-History Foundation, 1996.
  - Parashara. *Krishi Parashara* (translated by Sadhale & Nene). Asian Agri-History Foundation, 1999.
  - Sarngadhara. *Upavanavinoda* (translated by Nalini Sadhale).

- **Deep Learning:**
  - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

- **RAG:**
  - Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
  - Asai, A., et al. (2023). Retrieval-augmented generation for large language models. arXiv:2312.10997.

- **Explainability:**
  - Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.

- **Evaluation:**
  - Es, S., et al. (2023). RAGAS: A reference-free metric for evaluating retrieval-augmented generation. arXiv:2309.15217.

---

## Contact

For questions or collaboration inquiries, please contact:
- **Advisor:** Dr. Akshay Pandey, IIITDM Jabalpur
- **Student:** Ankit Pawar, M.Tech CSE, IIITDM Jabalpur
- **GitHub:**  https://github.com/ankit8453/iks-rag-thesis.git

---

**Last Updated:** May 15, 2026