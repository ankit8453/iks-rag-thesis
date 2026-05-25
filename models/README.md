# Phase 5 Disease Module — Trained Model Checkpoints

This directory mirrors the three private model repositories under
[`ankit-iiitdmj`](https://huggingface.co/ankit-iiitdmj) on Hugging Face
Hub. Files were pulled by [`scripts/download_models_from_hf.py`](../scripts/download_models_from_hf.py)
on **2026-05-24**.

The mirror exists so the trained checkpoints survive an HF outage, so
Phase 8 integration can load them via plain filesystem paths, and so a
reviewer can re-run inference without HF credentials.

## Provenance

| Subdir | HF Hub model repo | Stage | Dataset | n_classes | Epochs trained | Best val acc | Test top-1 | Test macro F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| `plantvillage/` | [`ankit-iiitdmj/iks-disease-plantvillage`](https://huggingface.co/ankit-iiitdmj/iks-disease-plantvillage) | pretrain | PlantVillage (color variant) | 38 | 25 | 0.9989 | **0.9976** | 0.9971 |
| `paddy-doctor/` | [`ankit-iiitdmj/iks-disease-paddy-doctor`](https://huggingface.co/ankit-iiitdmj/iks-disease-paddy-doctor) | finetune_paddy | Paddy Doctor (Kaggle competition) | 10 | 20 | 0.9693 | **0.9731** | 0.9713 |
| `plantdoc/` | [`ankit-iiitdmj/iks-disease-plantdoc`](https://huggingface.co/ankit-iiitdmj/iks-disease-plantdoc) | finetune_plantdoc | PlantDoc (real-field, 27-class post §A merge) | 27 | 30 | 0.7227 | **0.7109** | 0.7055 |

Each subdirectory contains five files (~204 MB per checkpoint):

| File | Purpose |
|---|---|
| `checkpoint_best.pt` | Best val-accuracy state across all epochs. Use this for downstream inference / Phase 8 integration. |
| `checkpoint_latest.pt` | Last-epoch state (= final-epoch state when training completed cleanly). |
| `history.json` | Per-epoch training/val metrics list (train/val acc + loss + macro F1 + LR + wall-clock). |
| `eval_metrics.json` | Final per-class precision/recall/F1 + confusion matrix on the **val** split. |
| `eval_metrics_test.json` | Same shape, on the **test** split. |

## Loading a checkpoint

```python
import torch
from src.disease.model import DiseaseClassifier

# 27 for PlantDoc, 10 for Paddy Doctor, 38 for PlantVillage.
model = DiseaseClassifier(num_classes=27, pretrained=False)
ckpt = torch.load(
    "models/disease/plantdoc/checkpoint_best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# Top-line stats from the same checkpoint
print(f"trained for {ckpt['epoch']} epochs")
print(f"best val acc: {ckpt['best_val_acc']:.4f}")
```

The checkpoint dict also carries `optimizer_state`, `scheduler_state`,
and the full per-epoch `history` list — useful only if you want to
resume training in Colab (set `OLID_FULL_DOWNLOAD = True` in
`scripts/download_olid_i.py` only matters for the C5 work in Phase 11;
the disease module trained here is the standalone vision side).

For richer inference (top-k probabilities + Grad-CAM overlay), use:

```python
from src.disease.infer import DiseaseInferenceEngine

engine = DiseaseInferenceEngine(
    model_source="models/disease/plantdoc/checkpoint_best.pt",
    class_names=[...],   # load from data/splits/plantdoc/class_map.json
    device="cpu",
    image_size=380,
)
result = engine.predict(pil_image, with_gradcam=True)
print(result.prediction.class_name, result.prediction.confidence)
print(result.top_k)
# result.gradcam_overlay is a numpy uint8 (H, W, 3) RGB heatmap.
```

## Update policy

These files are a snapshot of Phase 5 outputs as of **2026-05-24**.

- If a model is retrained, re-run
  `python scripts/download_models_from_hf.py` to refresh — the script
  is idempotent and uses SHA-256 to skip files that are already in
  sync.
- If you delete files here manually, the next run will re-download
  whatever is missing.
- The script verifies every `.pt` against its Hub-side LFS SHA-256
  during download and logs a clear `MISMATCH` if they ever diverge.

## Disk usage and Git LFS

This directory consumes **~1.2 GB on disk** (6 × ~200 MB checkpoints +
~92 KB of metric JSON). The original prompt expected ~640 MB but
checkpoints turned out to be ~204 MB each because our
`CheckpointManager.save_epoch` stores `optimizer_state` and
`scheduler_state` alongside `model_state` — useful for resuming
mid-stage on Colab.

`.pt` files are tracked via **Git LFS**:

```bash
git lfs track "models/disease/**/*.pt"
```

The `.gitattributes` at the repo root records this pattern. **Do not**
manually move or rename `.pt` files without re-running the download
script — Git LFS will see them as new files and bloat the LFS
quota.

JSON metric files (~5–18 KB each) are tracked via plain Git, not LFS —
LFS adds overhead with no benefit for files that small.

## Auxiliary HF cache

Each subdir also contains a `.cache/huggingface/` folder written by
`huggingface_hub.hf_hub_download`. It's a metadata cache (etag, download
state) — HF auto-writes a nested `.gitignore` inside that excludes it
from version control. Safe to delete; the download script regenerates
it on the next run.
