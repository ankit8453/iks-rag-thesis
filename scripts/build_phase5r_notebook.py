"""Generate ``notebooks/phase5r_retrain.ipynb`` (Phase 5-R Part 2).

12 cells per the prompt's locked structure. Same builder pattern as
``build_phase7_notebook.py`` etc.
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
    """# Phase 5-R Part 2 — Background-randomization retrain + no-leaf reject + Grad-CAM verdict

Phase 9 Grad-CAM revealed that the original Phase-5 disease cascade
attends to **image corners / backgrounds / watermarks** rather than the
leaf — only 3 of 256 PlantDoc test images had the CAM peak inside the
central 60 % box. Phase 5-R retrains the cascade with **randomized
backgrounds** so the background can no longer act as a label cue.

## What changes (Phase 5-R)

- **PlantVillage stage** — train on leaves segmented out (Part 1
  classical pipeline) and composited onto **random soil / urban
  backgrounds** each epoch. Same architecture, same hyperparameters,
  same seed — *only* the input pipeline changes.
- **Paddy Doctor stage** — train as-is. Part 1 verdict: paddy is
  full-canopy field photos with no meaningful foreground/background
  split, so randomization would be meaningless.
- **PlantDoc stage** — same randomization (Part 1 rembg pipeline)
  PLUS a **28th `no_leaf` reject class** drawn from Dr. Pandey's
  `Background_without_leaves` folder. This makes the model refuse to
  classify a non-leaf input, which feeds the Phase 10 Streamlit UI
  guardrail.

## Keep or revert

This is a **controlled experiment**. The decision rule (encoded in
`src.disease.gradcam_audit.keep_or_revert`):

| Outcome | Decision |
|---|---|
| Central-attention rate up by **≥ 5 pp** AND test top-1 accuracy not down by **> 3 pp** | **KEEP** the new model; re-run Phase 9 with it. |
| Otherwise | **REVERT** to the old model; document the bias as a Phase-11 follow-up. |

Old (`iks-disease-*`) and new (`iks-disease-r-*`) checkpoints live in
separate namespaces so revert is trivial.

## Hard rules (master plan §16)

- Local commits only — never `git push`.
- Do NOT merge Dr. Pandey's leaf classes (his dataset is a confirmed
  PlantVillage re-pack per `docs/pandey_dataset_inspection.md`).
  Only his `Background_without_leaves` folder enters this pipeline.
- Reuse Phase 5-R Part 1 code (`src/disease/segment.py`,
  `src/disease/backgrounds.py`) — don't rewrite the segmenters here.
"""
)

# ----------------------------- Cell 2 — setup ------------------------------
code(
    f"""# Cell 2 — clone repo + install dependencies + HF auth + GPU check
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

# Phase 5-R adds rembg + onnxruntime on top of Phase 5/7 deps.
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

# HF auth — needed to pull the existing iks-* datasets + the OLD
# baseline model.
from huggingface_hub import HfApi, login
login()
print(f"HF user: {{HfApi().whoami().get('name')}}")

# GPU check — T4 is enough for B4@380; sub-15-min epochs typical.
import torch
print(f"torch: {{torch.__version__}}  cuda: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"device: {{torch.cuda.get_device_name(0)}}")
else:
    print("WARNING: no GPU. The cascade will be infeasibly slow on CPU.")
"""
)

# ----------------------------- Cell 3 — bg pool ----------------------------
code(
    """# Cell 3 — Build the random-background pool + show 8 samples.
# Same pool used by Part 1 QC: Phantom-fs soils + Sirajganj moisture
# variants + Dr. Pandey's Background_without_leaves urban photos.
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.disease.backgrounds import (
    DEFAULT_BACKGROUND_ROOTS,
    build_background_pool,
    pool_size_by_source,
)

# Cap each source at 1k for the cell — full pool is built fresh at training time.
pool = build_background_pool(max_per_source=1000)
sizes = pool_size_by_source(pool)
print("Background pool sizes per source:")
for src, n in sizes.items():
    print(f"  {src:<24} {n}")
print(f"  TOTAL                   {len(pool)}")
assert len(pool) > 0, "Empty background pool — check the configured roots."

# 8-image preview (3 phantomfs, 3 sirajganj, 2 pandey if available).
import random
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
    """# Cell 4 — Segment + cache masks for PlantVillage (classical) +
# PlantDoc (rembg). Paddy Doctor is NOT segmented (Part 1 verdict).
# Idempotent: a second run skips cached masks.
from pathlib import Path

from src.disease.segment_cache import build_mask_cache
from src.utils.data_splits import load_split
from src.utils.paths import PROJECT_ROOT, DATA_PLANT_DISEASE_DIR

SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"


def _build_image_iter(splits_dir, raw_root):
    \"\"\"Concatenate train + val + test relpaths so the cache covers every
    image the trainer might draw at any stage.\"\"\"
    out = []
    seen = set()
    for split in ("train", "val", "test"):
        entries = load_split(splits_dir / f"{split}.json")
        for e in entries:
            rel = str(e.path).replace("\\\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            out.append((rel, raw_root / e.path))
    return out


# PlantVillage — classical HSV+Otsu+GrabCut
pv_root = DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw" / "plantvillage dataset" / "color"
pv_iter = _build_image_iter(SPLITS_ROOT / "plantvillage", pv_root)
print(f"PlantVillage images to segment: {len(pv_iter)}")
pv_stats = build_mask_cache("plantvillage", "lab", pv_iter, log_every=500)

# PlantDoc — rembg (U2Net). First call downloads ~170 MB.
pd_root = DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw"
pd_iter = _build_image_iter(SPLITS_ROOT / "plantdoc", pd_root)
print(f"PlantDoc images to segment: {len(pd_iter)}")
pd_stats = build_mask_cache("plantdoc", "field", pd_iter, log_every=50)

print()
print("=== Segmentation summary ===")
for stats in (pv_stats, pd_stats):
    print(
        f"  {stats.dataset:<14} total={stats.total:>5}  "
        f"new={stats.newly_segmented:>5}  flagged={stats.flagged:>4} "
        f"({stats.flagged_fraction:>5.1%})  failures={stats.failures}"
    )
assert pv_stats.flagged_fraction < 0.10, (
    f"PlantVillage flagged fraction {pv_stats.flagged_fraction:.1%} is high — "
    "review the segmentation before training."
)
assert pd_stats.flagged_fraction < 0.20, (
    f"PlantDoc flagged fraction {pd_stats.flagged_fraction:.1%} is high — "
    "review the segmentation before training."
)
"""
)

# ----------------------------- Cell 5 — sanity composites -----------------
code(
    """# Cell 5 — Sanity: render 6 on-the-fly composites so we can confirm
# the training inputs look right (leaves cleanly cut, bg looks like soil).
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.disease.backgrounds import composite_leaf_on_bg
from src.disease.randomized_dataset import build_samples_from_split
from src.disease.segment_cache import mask_path_for
from src.utils.paths import PROJECT_ROOT, DATA_PLANT_DISEASE_DIR

rng = random.Random(123)

# Pick 3 PlantVillage + 3 PlantDoc samples whose masks were NOT flagged.
def _sample_unflagged(splits_dir, raw_root, dataset_id, n):
    samples = build_samples_from_split(
        splits_dir / "train.json", raw_root, mode="randomize",
    )
    rng.shuffle(samples)
    out = []
    for s in samples:
        mp = mask_path_for(dataset_id, s.rel_path)
        if mp.is_file():
            out.append(s)
            if len(out) >= n:
                break
    return out

pv_samples = _sample_unflagged(
    PROJECT_ROOT / "data" / "splits" / "plantvillage",
    DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw" / "plantvillage dataset" / "color",
    "plantvillage", 3,
)
pd_samples = _sample_unflagged(
    PROJECT_ROOT / "data" / "splits" / "plantdoc",
    DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw",
    "plantdoc", 3,
)

fig, axes = plt.subplots(6, 3, figsize=(12, 18))
for i, s in enumerate(pv_samples + pd_samples):
    dataset = "plantvillage" if i < 3 else "plantdoc"
    img = Image.open(s.abs_path).convert("RGB")
    mask = Image.open(mask_path_for(dataset, s.rel_path)).convert("L")
    bg = rng.choice(pool)
    comp = composite_leaf_on_bg(img, mask, bg.path, out_size=img.size, rng=rng)
    axes[i, 0].imshow(img); axes[i, 0].set_title(f"{dataset}: original", fontsize=9); axes[i, 0].axis("off")
    axes[i, 1].imshow(mask, cmap="gray"); axes[i, 1].set_title("cached mask", fontsize=9); axes[i, 1].axis("off")
    axes[i, 2].imshow(comp); axes[i, 2].set_title(f"composite on {bg.source}", fontsize=9); axes[i, 2].axis("off")
plt.tight_layout(); plt.show()
"""
)

# ----------------------------- Cell 6 — stage 1 (PlantVillage) -------------
code(
    """# Cell 6 — Stage 1: pretrain B4 on randomized PlantVillage (38 classes).
# Same architecture / hyperparameters / seed as the original Phase 5
# trainer. The ONLY difference is the input pipeline (randomized vs raw).
import torch

from src.disease.config import DiseaseConfig
from src.disease.train_cascade_r import train_one_stage_r

config = DiseaseConfig()      # default seed=42, B4@380, AdamW
device = "cuda" if torch.cuda.is_available() else "cpu"

result_pv = train_one_stage_r("pretrain_r", config=config, device=device,
                              initial_state_dict=None)
print(f"Stage 1 done: best_val_acc={result_pv.best_val_acc:.4f}")
print(f"Checkpoint:   {result_pv.final_ckpt}")
"""
)

# ----------------------------- Cell 7 — stage 2 (Paddy) -------------------
code(
    """# Cell 7 — Stage 2: finetune on Paddy Doctor as-is (no segmentation,
# no randomization). Warm-starts from Stage 1's best checkpoint via
# strict=False so the 38-class head can be replaced by a 10-class one.
import torch

prior_state = torch.load(result_pv.final_ckpt, map_location="cpu", weights_only=False)
prior_state = prior_state.get("model_state", prior_state)

result_paddy = train_one_stage_r(
    "finetune_paddy_r", config=config, device=device,
    initial_state_dict=prior_state,
)
print(f"Stage 2 done: best_val_acc={result_paddy.best_val_acc:.4f}")
print(f"Checkpoint:   {result_paddy.final_ckpt}")
"""
)

# ----------------------------- Cell 8 — stage 3 (PlantDoc + no_leaf) ------
code(
    """# Cell 8 — Stage 3: finetune on randomized PlantDoc + the no_leaf
# reject class (28 classes total). Warm-starts from Stage 2.
import torch

prior_state = torch.load(result_paddy.final_ckpt, map_location="cpu", weights_only=False)
prior_state = prior_state.get("model_state", prior_state)

result_pd = train_one_stage_r(
    "finetune_plantdoc_r", config=config, device=device,
    initial_state_dict=prior_state,
)
print(f"Stage 3 done: best_val_acc={result_pd.best_val_acc:.4f}")
print(f"Checkpoint:   {result_pd.final_ckpt}")
"""
)

# ----------------------------- Cell 9 — RAW test eval ---------------------
code(
    """# Cell 9 — Evaluate the new model on RAW (un-composited) original
# test splits — directly comparable to the original Phase 5 numbers.
# Also reports the no_leaf reject head's precision/recall.
import torch

from src.disease.infer import DiseaseInferenceEngine
from src.disease.train_cascade_r import (
    CHECKPOINT_ROOT_R, build_loaders_for_stage,
)
from src.utils.data_splits import load_class_map
from src.utils.paths import PROJECT_ROOT

PV_NS = CHECKPOINT_ROOT_R / "iks-disease-r-plantvillage" / "checkpoint_best.pt"
PADDY_NS = CHECKPOINT_ROOT_R / "iks-disease-r-paddy-doctor" / "checkpoint_best.pt"
PD_NS = CHECKPOINT_ROOT_R / "iks-disease-r-plantdoc" / "checkpoint_best.pt"


def _eval_stage(stage_name, ckpt_path, class_map_path):
    cm = load_class_map(class_map_path) if class_map_path.is_file() else None
    class_names = [k for k, _ in sorted(cm.items(), key=lambda kv: kv[1])] if cm else None
    eng = DiseaseInferenceEngine(
        model_source=str(ckpt_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        class_names=class_names,
    )
    _, _, test_loader, _ = build_loaders_for_stage(
        stage_name, batch_size=16, num_workers=2, seed=42,
    )
    n_total = n_correct = 0
    per_class_tp = {}; per_class_pp = {}; per_class_pred_total = {}
    no_leaf_idx = None
    if class_names and "no_leaf" in class_names:
        no_leaf_idx = class_names.index("no_leaf")
    for images, labels in test_loader:
        images = images.to(eng.device)
        with torch.no_grad():
            logits = eng.model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        for p, y in zip(preds, labels.tolist()):
            n_total += 1
            if p == y: n_correct += 1
            per_class_pp[y] = per_class_pp.get(y, 0) + 1
            per_class_pred_total[p] = per_class_pred_total.get(p, 0) + 1
            if p == y:
                per_class_tp[y] = per_class_tp.get(y, 0) + 1
    acc = n_correct / max(1, n_total)
    print(f"  {stage_name:<22} top-1 acc = {acc:.4f}   n={n_total}")
    if no_leaf_idx is not None:
        tp = per_class_tp.get(no_leaf_idx, 0)
        pp = per_class_pp.get(no_leaf_idx, 0)
        ppt = per_class_pred_total.get(no_leaf_idx, 0)
        prec = tp / max(1, ppt)
        rec = tp / max(1, pp)
        print(f"    no_leaf reject class: precision={prec:.3f}  recall={rec:.3f}  "
              f"(support: {pp} GT, {ppt} predicted)")
    return acc

print("=== NEW model evaluation on RAW test splits ===")
acc_pv = _eval_stage("pretrain_r", PV_NS,
                     PROJECT_ROOT / "data" / "splits" / "plantvillage" / "class_map.json")
acc_paddy = _eval_stage("finetune_paddy_r", PADDY_NS,
                     PROJECT_ROOT / "data" / "splits" / "paddy_doctor" / "class_map.json")
acc_pd = _eval_stage("finetune_plantdoc_r", PD_NS,
                     PROJECT_ROOT / "data" / "splits" / "plantdoc" / "class_map.json")
"""
)

# ----------------------------- Cell 10 — Grad-CAM audit -------------------
code(
    """# Cell 10 — Grad-CAM central-attention audit, OLD vs NEW, on the
# SAME PlantDoc test images. This is the headline number the keep/revert
# decision rule reads. ~5-7 min on T4.
from src.disease.gradcam_audit import (
    keep_or_revert, print_comparison, run_old_vs_new,
)

old_summary, new_summary = run_old_vs_new(device="cuda" if torch.cuda.is_available() else "cpu")
print_comparison(old_summary, new_summary)

verdict = keep_or_revert(old_summary, new_summary)
print()
print(f"VERDICT: {verdict}")

# Persist the audit to docs/ so future re-runs can compare against today.
import json
from pathlib import Path
audit_path = Path("docs/phase5r_audit.json")
audit_path.parent.mkdir(exist_ok=True)
audit_path.write_text(json.dumps({
    "old": old_summary.to_json(),
    "new": new_summary.to_json(),
    "verdict": verdict,
}, indent=2), encoding="utf-8")
print(f"Audit JSON saved to {audit_path}")
"""
)

# ----------------------------- Cell 11 — verdict md -----------------------
md(
    """## VERDICT — keep or revert (read the table above)

| Metric | Old (Phase 5) | New (Phase 5-R) | Delta | Threshold |
|---|---:|---:|---:|---|
| Central-attention rate (CAM peak in central 60% box) | filled in by Cell 10 | filled in by Cell 10 | filled in | **gain >= +5 pp** |
| Top-1 accuracy on RAW PlantDoc test | filled in by Cell 10 | filled in by Cell 10 | filled in | **drop <= 3 pp** |

The decision rule (`src.disease.gradcam_audit.keep_or_revert`):

- **KEEP** the new model iff *both* thresholds are met. Push the new
  checkpoint to `ankit-iiitdmj/iks-disease-r-plantdoc`, re-run the
  Phase 9 notebook with it for honest lesion-focused figures, and the
  `no_leaf` reject class wires into the Phase 10 Streamlit UI guardrail
  ("we don't think this is a leaf — please retry with a leaf image").
- **REVERT** otherwise. The old `iks-disease-*` checkpoints are
  untouched (separate namespace) so revert is just "ignore the new
  ones". Document the unchanged shortcut bias as a Phase-11
  RAGAS / pointing-game follow-up.

A null result (bias unchanged or accuracy regressed) is a **valid,
documentable finding** — not a failure. The point of the experiment
is to *measure* the effect, not to force a win.
"""
)

# ----------------------------- Cell 12 — next steps md --------------------
md(
    """## What comes next (whichever way the verdict lands)

### If KEEP

1. Push the three new checkpoints (`iks-disease-r-*`) to HF. The old
   ones stay live until Phase 9 confirms the new ones produce better
   figures.
2. Re-run `notebooks/phase9_explainability.ipynb` after swapping
   `disease_engine` to load from the new `iks-disease-r-plantdoc` repo.
   The Cell 7 image-picker (`scripts/find_phase9_demo_images.py`)
   should now find more central-attention samples — expect the
   "qualified" count to rise from 3 / 256.
3. Wire the `no_leaf` reject head into Phase 10's Streamlit UI:
   `if pred_class_name == "no_leaf"` -> show "we don't think this is
   a leaf, please upload a leaf photo".

### If REVERT

1. Keep using `iks-disease-plantdoc`. The Phase 9 panels stay as-is.
2. Add a Phase-11 follow-up: train with stronger leaf segmentation +
   pointing-game evaluation. Possibly try SAM for paddy too, and
   re-attempt randomization there.
3. Document the unchanged shortcut bias in the thesis as an honest
   limitation — Grad-CAM-revealed PlantDoc data bias, not a failure
   of the pipeline.

### In either case

- The Phase 5-R Part 1 segmentation pipeline (`src/disease/segment.py`,
  `src/disease/backgrounds.py`) stays — it's useful for Phase 10's
  "upload a leaf" preview pane.
- The `no_leaf` reject class concept feeds the Phase 10 UI guardrail
  even if we revert to the old classifier (use a small confidence
  threshold as a proxy).
- Phase 11's RAGAS / pointing-game evaluation is unchanged either way.
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
