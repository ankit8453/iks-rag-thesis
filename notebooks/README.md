# notebooks/

Jupyter notebooks for exploratory experiments and training runs (per §41).

| Notebook | Purpose | Where it runs |
|---|---|---|
| `00_environment_check.ipynb` | Sanity-check the locked stack imports (torch, timm, sentence-transformers, chromadb) | Laptop / any |
| `dataset_eda.ipynb` | Phase 4 EDA — per-dataset class distributions, sample images, image-size histograms, OLID multi-label co-occurrence heatmap | Laptop |
| `phase5_disease_training.ipynb` | **Colab** — three-stage cascade: PlantVillage pretrain → Paddy Doctor fine-tune → PlantDoc fine-tune. Checkpoints pushed to private HF Hub model repos so the run survives Colab session resets. | **Google Colab (GPU)** |

Phase 5 training is documented in `PHASE5_LAPTOP_SUMMARY.md` (laptop-side
preparation) and `PHASE5_COLAB_GUIDE.md` (Colab-side human procedure,
shipped separately by Ankit).
