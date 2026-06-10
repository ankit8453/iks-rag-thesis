"""Generate ``notebooks/phase5r_retrain.ipynb`` (Phase 5-R Part 2).

12 cells, HF-first end-to-end (same pattern as the original Phase 5
trainer): no local file paths, every dataset pulled from HuggingFace,
every checkpoint saved AND resumed via the HF model namespace
``ankit-iiitdmj/iks-disease-r-*``. A free-Colab session timeout
mid-stage is harmless — re-running the same cell pulls
``checkpoint_latest.pt`` from HF and resumes from the saved epoch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase5r_retrain.ipynb"
REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

NB = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ----------------------------- Cell 1 — title md ---------------------------
md(
    """# Phase 5-R Part 2 — Background-randomization retrain (HF-first, resumable)

Phase 9 Grad-CAM revealed that the original Phase-5 disease cascade
attends to **image corners / backgrounds / watermarks** rather than the
leaf — only 3 of 256 PlantDoc test images had the CAM peak inside the
central 60 % box. Phase 5-R retrains the cascade with **randomized
backgrounds** so the background can no longer act as a label cue.

## Same plumbing as the original Phase 5

This notebook follows the exact same Colab-friendly pattern as
`notebooks/phase5_disease_training.ipynb`:

- Every dataset is pulled from HuggingFace via `load_dataset(...)` —
  **NO local file paths**, so a fresh Colab runtime works.
- Every checkpoint is saved to **HF Hub** through `CheckpointManager`
  → push `checkpoint_latest.pt` + `checkpoint_best.pt` after every
  epoch.
- Resume is automatic: a free-Colab session timeout mid-stage is
  harmless — re-running the same cell pulls `checkpoint_latest.pt`
  from HF and continues from the saved epoch.

## What's different from Phase 5

- **PlantVillage stage** — train on leaves segmented out (Part 1
  classical pipeline) and composited onto **random soil / urban
  backgrounds** each epoch. Same architecture, same hyperparameters,
  same seed.
- **Paddy Doctor stage** — train as-is. Part 1 verdict: paddy is
  full-canopy field photos with no meaningful foreground/background
  split.
- **PlantDoc stage** — same randomization PLUS a **28th `no_leaf`
  reject class** drawn from Pandey's `Background_without_leaves`
  (laptop only) OR the soil backgrounds (Colab fallback).

## Keep / revert rule

| Outcome | Decision |
|---|---|
| Central-attention rate up by ≥ 5 pp AND test top-1 not down by > 3 pp | **KEEP** the new model; re-run Phase 9 with it. |
| Otherwise | **REVERT**; document the bias as a Phase-11 follow-up. |

Old (`iks-disease-*`) and new (`iks-disease-r-*`) checkpoints live in
separate HF namespaces so revert is trivial (just keep using the old
one).
"""
)

# ----------------------------- Cell 2 — setup ------------------------------
code(
    f"""# Cell 2 — clone repo + install deps + HF login + GPU check.
import os
import subprocess
import sys

REPO_URL = "{REPO_HTTPS_URL}"
REPO_PATH = "{REPO_LOCAL_PATH}"

if not os.path.exists(REPO_PATH):
    subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
else:
    subprocess.run(["git", "-C", REPO_PATH, "pull", "--ff-only"], check=True)
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)

DEPS = [
    "timm>=1.0",
    "albumentations>=1.4",
    "rembg>=2.0",
    "onnxruntime>=1.18",
    "huggingface_hub>=0.24",
    "datasets>=2.20",
    "transformers>=4.44,<4.50",
    "matplotlib>=3.7",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
print("Dependencies installed.")

from huggingface_hub import HfApi, login
login()
me = HfApi().whoami()
print(f"HF user: {{me.get('name')}}  role={{me.get('auth',{{}}).get('accessToken',{{}}).get('role')}}")
assert me.get("name") == "ankit-iiitdmj", (
    "HF token must belong to ankit-iiitdmj — Phase 5-R writes new "
    "checkpoints to ankit-iiitdmj/iks-disease-r-*."
)

import torch
print(f"torch: {{torch.__version__}}  cuda: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"device: {{torch.cuda.get_device_name(0)}}")
else:
    print("WARNING: no GPU. The cascade is infeasibly slow on CPU.")
"""
)

# ----------------------------- Cell 3 — bg pool ----------------------------
code(
    """# Cell 3 — Background pool. On the laptop the local soil trees +
# Pandey folder are walked directly. On Colab the local trees are
# absent; build_background_pool() then transparently downloads from
# the published HF soil datasets and caches under data/_bg_cache/.
import random

import matplotlib.pyplot as plt
from PIL import Image

from src.disease.backgrounds import (
    build_background_pool, pool_size_by_source,
)

pool = build_background_pool()  # uses HF fallback when local roots missing
sizes = pool_size_by_source(pool)
print("Background pool sizes per source:")
for src, n in sizes.items():
    print(f"  {src:<24} {n}")
print(f"  TOTAL                   {len(pool)}")
assert len(pool) > 0, "Empty background pool — check HF auth + dataset access."

# 8-image preview so we can SEE what the trainer will composite onto.
rng = random.Random(42)
preview = rng.sample(pool, min(8, len(pool)))
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, entry in enumerate(preview):
    ax = axes[i // 4, i % 4]
    ax.imshow(Image.open(entry.path).convert("RGB"))
    ax.set_title(f"{entry.source}\\n{entry.rel_path[:36]}", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
"""
)

# ----------------------------- Cell 4 — segment + cache --------------------
code(
    """# Cell 4 — Segment + cache masks (with HF Hub backup so a Colab
# session-timeout doesn't wipe progress).
#
# build_mask_cache_from_hf does THREE things by default:
#   1. At start: pulls any existing mask tarball from
#      ``ankit-iiitdmj/iks-disease-r-mask-cache`` and extracts it under
#      data/plant_disease/_masks/ — so a fresh Colab account resumes
#      from where the previous session stopped.
#   2. While running: tar+pushes the cache to HF every 5,000 newly
#      segmented rows. A crash between pushes only loses at most that
#      many rows of work.
#   3. End of each split: a final push so the next run sees the
#      complete split.
#
# To bypass HF backup (laptop dev): pass ``hf_backup_repo=None``.
from src.disease.segment_cache import (
    DEFAULT_MASK_BACKUP_REPO,
    build_mask_cache_from_hf,
    load_flagged_set,
)
print(f"Mask backup HF repo: {DEFAULT_MASK_BACKUP_REPO}")

print()
print("=== PlantVillage (classical) ===")
for split in ("train", "val", "test"):
    stats = build_mask_cache_from_hf(
        dataset_repo="ankit-iiitdmj/iks-plantvillage",
        dataset_id="plantvillage",
        split=split, style="lab", log_every=500,
        hf_push_every_n_rows=5000,
    )
    print(f"  {split:<6} total={stats.total:>5} new={stats.newly_segmented:>5}"
          f" flagged={stats.flagged:>4} ({stats.flagged_fraction:>5.1%})"
          f" failures={stats.failures}")

print()
print("=== PlantDoc (rembg / U2Net) ===")
for split in ("train", "val", "test"):
    stats = build_mask_cache_from_hf(
        dataset_repo="ankit-iiitdmj/iks-plantdoc",
        dataset_id="plantdoc",
        split=split, style="field", log_every=50,
        hf_push_every_n_rows=500,   # PD is small, push more often
    )
    print(f"  {split:<6} total={stats.total:>5} new={stats.newly_segmented:>5}"
          f" flagged={stats.flagged:>4} ({stats.flagged_fraction:>5.1%})"
          f" failures={stats.failures}")

print()
print(f"Flagged keys persisted across splits:"
      f" plantvillage={len(load_flagged_set('plantvillage'))},"
      f" plantdoc={len(load_flagged_set('plantdoc'))}")
"""
)

# ----------------------------- Cell 5 — sanity composites ------------------
code(
    """# Cell 5 — Sanity: render 6 on-the-fly composites from the actual
# randomized dataset wrapper. If anything looks off (mask misses the
# leaf, bg pattern bleeding through where the leaf should be opaque,
# etc.) STOP and re-segment before letting Stage 1 run.
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from src.disease.randomized_dataset import HFRandomizedDiseaseDataset

rng = random.Random(123)

def _preview(dataset_repo, dataset_id, n_rows=3):
    hf_train = load_dataset(dataset_repo, split="train")
    ds = HFRandomizedDiseaseDataset(
        hf_split=hf_train, dataset_id=dataset_id, split="train",
        mode="randomize", bg_pool=pool, transform=None, no_leaf_rows=[],
        seed=123,
    )
    ds.set_epoch(0)
    out = []
    indices = rng.sample(range(len(hf_train)), n_rows)
    for idx in indices:
        composed_arr, label = ds[idx]
        out.append((idx, label, composed_arr))
    return out

previews = _preview("ankit-iiitdmj/iks-plantvillage", "plantvillage") + \\
           _preview("ankit-iiitdmj/iks-plantdoc",     "plantdoc")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (idx, label, arr) in enumerate(previews):
    ax = axes[i // 3, i % 3]
    ax.imshow(arr); ax.set_title(f"idx={idx} label={label}", fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
"""
)

# ----------------------------- Cell 6 — Stage 1 (PlantVillage) -------------
code(
    """# Cell 6 — Stage 1: pretrain B4 on randomized PlantVillage (38 classes).
# Saves to ankit-iiitdmj/iks-disease-r-plantvillage every epoch. If
# this cell is interrupted (Colab timeout), re-running it pulls
# checkpoint_latest.pt and resumes from the saved epoch.
import torch

from src.disease.config import DiseaseConfig
from src.disease.train_cascade_r import train_one_stage_r

config = DiseaseConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

result_pv = train_one_stage_r(
    "pretrain_r", config=config, device=device,
    initial_state_dict=None, resume_from_hub=True,
)
print(f"\\nStage 1 done: best_val_acc={result_pv.best_val_acc:.4f}")
print(f"Checkpoints on HF: {result_pv.model_repo}")
"""
)

# ----------------------------- Cell 7 — Stage 2 (Paddy) -------------------
code(
    """# Cell 7 — Stage 2: finetune on Paddy Doctor as-is (no randomization).
# Warm-starts from Stage 1's best on HF; saves to
# ankit-iiitdmj/iks-disease-r-paddy-doctor.
import torch

from huggingface_hub import hf_hub_download

local = hf_hub_download(
    repo_id="ankit-iiitdmj/iks-disease-r-plantvillage",
    filename="checkpoint_best.pt",
    repo_type="model",
)
prior_state = torch.load(local, map_location="cpu", weights_only=False)
prior_state = prior_state.get("model_state", prior_state)

result_paddy = train_one_stage_r(
    "finetune_paddy_r", config=config, device=device,
    initial_state_dict=prior_state, resume_from_hub=True,
)
print(f"\\nStage 2 done: best_val_acc={result_paddy.best_val_acc:.4f}")
print(f"Checkpoints on HF: {result_paddy.model_repo}")
"""
)

# ----------------------------- Cell 8 — Stage 3 (PlantDoc + no_leaf) -------
code(
    """# Cell 8 — Stage 3: finetune on randomized PlantDoc + 28th no_leaf
# class. Warm-starts from Stage 2; saves to
# ankit-iiitdmj/iks-disease-r-plantdoc.
import torch

from huggingface_hub import hf_hub_download

local = hf_hub_download(
    repo_id="ankit-iiitdmj/iks-disease-r-paddy-doctor",
    filename="checkpoint_best.pt",
    repo_type="model",
)
prior_state = torch.load(local, map_location="cpu", weights_only=False)
prior_state = prior_state.get("model_state", prior_state)

result_pd = train_one_stage_r(
    "finetune_plantdoc_r", config=config, device=device,
    initial_state_dict=prior_state, resume_from_hub=True,
)
print(f"\\nStage 3 done: best_val_acc={result_pd.best_val_acc:.4f}")
print(f"Checkpoints on HF: {result_pd.model_repo}")
"""
)

# ----------------------------- Cell 9 — RAW test eval ---------------------
code(
    """# Cell 9 — Evaluate the NEW cascade on RAW (un-composited) test
# splits — directly comparable to the original Phase 5 numbers.
# Also reports the no_leaf reject precision / recall at the
# PlantDoc stage.
import torch

from datasets import load_dataset

from src.disease.infer import DiseaseInferenceEngine
from src.disease.train_cascade_r import STAGE_INFO_R
from src.disease.transforms import build_disease_eval_aug

mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
eval_aug = build_disease_eval_aug(380, mean, std)


def _eval_stage(stage_name):
    info = STAGE_INFO_R[stage_name]
    repo = info["model_repo"]
    print(f"--- {stage_name}  ({repo}) ---")
    engine = DiseaseInferenceEngine(
        model_source=repo,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    hf_test = load_dataset(info["dataset_repo"], split="test")
    n_total = n_correct = 0
    import numpy as np
    no_leaf_idx = info["num_classes"] - 1 if info.get("add_no_leaf") else None
    tp = pp_gt = pp_pred = 0
    for row in hf_test:
        arr = np.asarray(row["image"].convert("RGB"))
        tensor = eval_aug(image=arr)["image"].unsqueeze(0).to(engine.device)
        with torch.no_grad():
            logits = engine.model(tensor)
        pred = int(logits.argmax(dim=1).item())
        n_total += 1
        if pred == int(row["label_idx"]):
            n_correct += 1
        if no_leaf_idx is not None:
            if int(row["label_idx"]) == no_leaf_idx:
                pp_gt += 1
                if pred == no_leaf_idx:
                    tp += 1
            if pred == no_leaf_idx:
                pp_pred += 1
    acc = n_correct / max(1, n_total)
    print(f"  top-1 acc = {acc:.4f}  n={n_total}")
    if no_leaf_idx is not None:
        prec = tp / max(1, pp_pred); rec = tp / max(1, pp_gt)
        print(f"  no_leaf: precision={prec:.3f}  recall={rec:.3f}"
              f"  (GT={pp_gt}, predicted={pp_pred})")
    return acc

acc_pv = _eval_stage("pretrain_r")
acc_paddy = _eval_stage("finetune_paddy_r")
acc_pd = _eval_stage("finetune_plantdoc_r")
"""
)

# ----------------------------- Cell 10 — Grad-CAM audit -------------------
code(
    """# Cell 10 — Grad-CAM central-attention audit, OLD vs NEW, on the
# same PlantDoc test images. This is the headline metric for the
# keep/revert decision. ~5-7 min on T4.
import json
from pathlib import Path

import torch

from src.disease.gradcam_audit import (
    DEFAULT_OLD_REPO,
    keep_or_revert, print_comparison, run_old_vs_new,
)
from src.disease.infer import DiseaseInferenceEngine

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build the NEW engine from the HF model repo we just pushed to.
new_engine = DiseaseInferenceEngine(
    model_source="ankit-iiitdmj/iks-disease-r-plantdoc",
    device=device,
)
# OLD engine is the existing Phase-5 model on HF.
old_engine = DiseaseInferenceEngine(
    model_source=DEFAULT_OLD_REPO, device=device,
)

old_summary, new_summary = run_old_vs_new(
    old_engine=old_engine, new_engine=new_engine, device=device,
)
print_comparison(old_summary, new_summary)
verdict = keep_or_revert(old_summary, new_summary)
print()
print(f"VERDICT: {verdict}")

Path("docs").mkdir(exist_ok=True)
Path("docs/phase5r_audit.json").write_text(
    json.dumps(
        {"old": old_summary.to_json(),
         "new": new_summary.to_json(),
         "verdict": verdict},
        indent=2,
    ),
    encoding="utf-8",
)
print("Audit JSON saved to docs/phase5r_audit.json")
"""
)

# ----------------------------- Cell 11 — verdict md -----------------------
md(
    """## VERDICT — keep or revert (read the table from Cell 10)

| Metric | OLD (Phase 5) | NEW (Phase 5-R) | Delta | Threshold |
|---|---:|---:|---:|---|
| Central-attention rate (CAM peak in central 60 % box) | filled by Cell 10 | filled by Cell 10 | filled by Cell 10 | gain ≥ +5 pp |
| Top-1 accuracy on RAW PlantDoc test | filled by Cell 10 | filled by Cell 10 | filled by Cell 10 | drop ≤ 3 pp |

`src.disease.gradcam_audit.keep_or_revert(old, new)` encodes the rule
above and printed the decision in Cell 10.

- **KEEP**: both thresholds met. Re-run `notebooks/phase9_explainability.ipynb`
  with `disease_engine` pointing at `ankit-iiitdmj/iks-disease-r-plantdoc`
  to refresh the figures, and wire the no_leaf reject head into the
  Phase 10 Streamlit UI guardrail.
- **REVERT**: keep using `ankit-iiitdmj/iks-disease-plantdoc`; document
  the unchanged bias as a Phase-11 RAGAS / pointing-game follow-up.
  The old checkpoints are untouched (separate namespace) so revert is
  literally "ignore the new ones".

A null result (bias unchanged or accuracy regressed) is a **valid,
documentable finding** — the point of the experiment is to *measure*
the effect, not to force a win.
"""
)

# ----------------------------- Cell 12 — next steps md --------------------
md(
    """## What comes next

### If KEEP

1. Tag a release on the three `ankit-iiitdmj/iks-disease-r-*` repos
   so future loads pin to today's checkpoint.
2. Re-run `notebooks/phase9_explainability.ipynb` after swapping
   `disease_engine = DiseaseInferenceEngine(model_source="ankit-iiitdmj/iks-disease-r-plantdoc", ...)`.
   Cell 7's image-picker (`scripts/find_phase9_demo_images.py`) should
   now find more central-attention samples — expect the "qualified"
   count to rise from 3 / 256.
3. Wire the `no_leaf` reject head into Phase 10's Streamlit UI:
   `if pred_class_name == "no_leaf": st.warning("We don't think this
   is a leaf — please upload a leaf photo.")`.

### If REVERT

1. Keep `ankit-iiitdmj/iks-disease-plantdoc` as-is. The Phase 9 panels
   stay; the bias is now an honest Phase-11 limitation.
2. Add a Phase-11 follow-up: train with stronger leaf segmentation
   (maybe SAM for paddy too) and quantitative pointing-game
   evaluation. Document the bias openly in the thesis.

### In either case

- The Phase 5-R Part 1 segmentation pipeline stays — useful for the
  Phase 10 "upload a leaf" preview pane.
- The `no_leaf` reject concept feeds the UI guardrail either way
  (use a low-confidence threshold as a proxy if we revert).
- Phase 11's RAGAS / pointing-game evaluation is unchanged.
"""
)


# ============================================================================
# Assemble + write
# ============================================================================

NB.cells = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(NB, fh)

print(f"Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)} with {len(cells)} cells.")
