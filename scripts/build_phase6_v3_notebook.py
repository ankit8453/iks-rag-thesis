"""Generate ``notebooks/phase6_soil_training_v3.ipynb`` (Phase 6 V3 §C).

One-shot builder for the V3 sequential-transfer-learning experiment.
Mirrors V1/V2 cell structure; the differences are the 3 staged training
cells (9/10/11) and the Cell-13 V2-vs-V3 comparison + ship/revert
verdict. V1/V2 notebooks stay untouched.

14 cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase6_soil_training_v3.ipynb"

REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

# V2 baseline (TTA) from ankit-iiitdmj/iks-soil-multitask-v2/eval_metrics_test.json
V2_BASELINE = {
    "soil_type": {"top1": 0.8992, "f1": 0.8509},
    "moisture":  {"top1": 0.9576, "f1": 0.9578},
    "texture":   {"top1": 0.6786, "f1": 0.6778},
}

NB = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# --------------------------------------------------------------------- #
# Cell 1 — Title + V2 baseline + 3-stage plan
# --------------------------------------------------------------------- #
md(
    "# Phase 6 V3 — Sequential Transfer Learning (Texture Boost Experiment)\n"
    "\n"
    "**Hypothesis:** warming the EfficientNet-B0 backbone on Indian soil data\n"
    "(Phantom-fs soil_type) → then moisture → then activating texture produces\n"
    "a more soil-aware backbone than ImageNet pretraining alone. Expected\n"
    "texture gain: **+2–5 points**.\n"
    "\n"
    "**This is an EXPERIMENT, not a commitment.** If V3 doesn't clearly beat\n"
    "V2, ship V2 (which stays untouched at `ankit-iiitdmj/iks-soil-multitask-v2`).\n"
    "\n"
    "**3-stage plan (50 epochs total):**\n"
    "- **Stage A** (15 ep): Phantom-fs only — soil_type head. 5 frozen + 10 unfrozen.\n"
    "- **Stage B** (15 ep): + Sirajganj moisture — soil_type + moisture heads. All unfrozen.\n"
    "- **Stage C** (20 ep): + texture-irsid-vit — all 3 heads. All unfrozen. Final model.\n"
    "\n"
    "All stages use V2's strong augmentation + Mixup/CutMix (p=0.3) + label\n"
    "smoothing 0.1. TTA (5 views) is used for the final test eval only.\n"
    "\n"
    "**V2 baseline (TTA test set), the bar to beat:**\n"
    f"- `soil_type`: {V2_BASELINE['soil_type']['top1']*100:.2f}% / F1 {V2_BASELINE['soil_type']['f1']:.3f}\n"
    f"- `moisture`:  {V2_BASELINE['moisture']['top1']*100:.2f}% / F1 {V2_BASELINE['moisture']['f1']:.3f}\n"
    f"- `texture`:   {V2_BASELINE['texture']['top1']*100:.2f}% / F1 {V2_BASELINE['texture']['f1']:.3f}\n"
    "\n"
    "**Success criteria:** texture top-1 up ≥3 pts AND neither soil_type nor\n"
    "moisture drops >2 pts. Otherwise revert to V2.\n"
)

# --------------------------------------------------------------------- #
# Cell 2 — Setup
# --------------------------------------------------------------------- #
code(
    "# Cell 2 — setup: clone repo + install dependencies (defensive)\n"
    f"REPO_URL = \"{REPO_HTTPS_URL}\"\n"
    f"REPO_PATH = \"{REPO_LOCAL_PATH}\"\n"
    "\n"
    "import os, subprocess, sys\n"
    "\n"
    "if os.path.isdir(REPO_PATH) and not os.path.isfile(os.path.join(REPO_PATH, \"requirements.txt\")):\n"
    "    print(f\"Removing partial clone at {REPO_PATH} ...\")\n"
    "    subprocess.run([\"rm\", \"-rf\", REPO_PATH], check=True)\n"
    "if not os.path.isdir(REPO_PATH):\n"
    "    subprocess.run([\"git\", \"clone\", REPO_URL, REPO_PATH], check=True)\n"
    "\n"
    "os.chdir(REPO_PATH)\n"
    "print(\"Repo root contents:\", sorted(os.listdir(\".\")))\n"
    "\n"
    "_pip_packages = [\n"
    "    \"timm>=1.0.0\",\n"
    "    \"albumentations>=1.4\",\n"
    "    \"huggingface_hub>=0.24\",\n"
    "    \"datasets>=2.20\",\n"
    "    \"iterative-stratification>=0.1.7\",\n"
    "    \"pydantic>=2.7\",\n"
    "    \"pyyaml>=6.0\",\n"
    "]\n"
    "proc = subprocess.run(\n"
    "    [sys.executable, \"-m\", \"pip\", \"install\", *_pip_packages],\n"
    "    capture_output=True, text=True,\n"
    ")\n"
    "if proc.returncode != 0:\n"
    "    print(\"PIP STDOUT (tail):\\n\" + proc.stdout[-3000:])\n"
    "    print(\"PIP STDERR (tail):\\n\" + proc.stderr[-3000:])\n"
    "    raise SystemExit(\"pip install failed — see tails above.\")\n"
    "print(\"setup ok\")\n"
)

# --------------------------------------------------------------------- #
# Cell 3 — HF Hub auth
# --------------------------------------------------------------------- #
code(
    "# Cell 3 — HF Hub login\n"
    "from huggingface_hub import login, HfApi\n"
    "login()  # Colab inline widget — paste your Write token\n"
    "\n"
    "_whoami = HfApi().whoami()\n"
    "assert _whoami[\"name\"] == \"ankit-iiitdmj\", (\n"
    "    f\"HF Hub token belongs to {_whoami['name']!r}, expected 'ankit-iiitdmj'.\"\n"
    ")\n"
    "print(f\"HF Hub ok: user={_whoami['name']}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 4 — GPU + auto batch size
# --------------------------------------------------------------------- #
code(
    "# Cell 4 — GPU check + auto batch size\n"
    "import subprocess, torch, sys\n"
    "sys.path.insert(0, REPO_PATH)\n"
    "\n"
    "subprocess.run([\"nvidia-smi\"], check=False)\n"
    "print()\n"
    "if torch.cuda.is_available():\n"
    "    dev = torch.cuda.get_device_properties(0)\n"
    "    print(f\"GPU: {dev.name}, VRAM: {dev.total_memory / 1024**3:.1f} GiB\")\n"
    "else:\n"
    "    print(\"No GPU detected — switch runtime to GPU before continuing.\")\n"
    "\n"
    "from src.soil.train import auto_batch_size\n"
    "batch = auto_batch_size()\n"
)

# --------------------------------------------------------------------- #
# Cell 5 — V3 configuration markdown
# --------------------------------------------------------------------- #
md(
    "## V3 Configuration\n"
    "\n"
    "- Backbone: **EfficientNet-B0 @ 224×224** (UNCHANGED from V1/V2)\n"
    "- 3 sequential stages: A=15 ep, B=15 ep, C=20 ep (50 total)\n"
    "- AdamW `lr=1e-4`, `weight_decay=1e-4`; a fresh CosineAnnealingLR per\n"
    "  stage (`T_max` = that stage's epoch count)\n"
    "- V2 strong augmentation in ALL stages\n"
    "- Mixup/CutMix `p=0.3` in ALL stages\n"
    "- Label smoothing 0.1 in ALL stages\n"
    "- TTA (5 views) for the final test eval (Cell 13) only\n"
    "- Per-epoch checkpoints pushed to `ankit-iiitdmj/iks-soil-multitask-v3`\n"
    "- Stage transitions also save `checkpoint_stage_{a,b,c}.pt`\n"
)

# --------------------------------------------------------------------- #
# Cell 6 — Load 3 datasets
# --------------------------------------------------------------------- #
code(
    "# Cell 6 — Load all 3 HF datasets (used across the 3 stages)\n"
    "from datasets import load_dataset\n"
    "\n"
    "DATASETS = {\n"
    "    \"soil_type\": \"ankit-iiitdmj/iks-soil-phantomfs\",\n"
    "    \"moisture\":  \"ankit-iiitdmj/iks-soil-sirajganj-moisture\",\n"
    "    \"texture\":   \"ankit-iiitdmj/iks-soil-texture-irsid-vit\",\n"
    "}\n"
    "loaded = {head: load_dataset(repo) for head, repo in DATASETS.items()}\n"
    "for head, dsd in loaded.items():\n"
    "    repo = DATASETS[head].split(\"/\")[-1]\n"
    "    print(f\"{repo}: train={len(dsd['train'])} val={len(dsd['val'])} test={len(dsd['test'])}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 7 — V3 transforms + per-stage loaders + eval loaders
# --------------------------------------------------------------------- #
code(
    "# Cell 7 — V2 augmentation + per-stage train loaders + per-task eval loaders\n"
    "import numpy as np\n"
    "import torch\n"
    "import yaml\n"
    "from torch.utils.data import DataLoader\n"
    "\n"
    "from src.soil.train_v3 import build_stage_loader, _MultiTaskHFDataset\n"
    "from src.soil.transforms import build_soil_eval_aug\n"
    "from src.soil.transforms_v2 import build_soil_train_aug_v2\n"
    "\n"
    "with open(\"configs/data/soil_norm.yaml\") as fh:\n"
    "    norm = yaml.safe_load(fh)\n"
    "MEAN = tuple(norm[\"mean\"])\n"
    "STD  = tuple(norm[\"std\"])\n"
    "IMG_SIZE = 224\n"
    "\n"
    "train_aug = build_soil_train_aug_v2(IMG_SIZE, MEAN, STD)\n"
    "eval_aug  = build_soil_eval_aug(IMG_SIZE, MEAN, STD)\n"
    "\n"
    "train_splits = {head: loaded[head][\"train\"] for head in (\"soil_type\", \"moisture\", \"texture\")}\n"
    "\n"
    "# Per-stage train loaders (ConcatDataset of the included sources).\n"
    "stage_a_loader = build_stage_loader(\"A\", train_splits, train_aug, batch)\n"
    "stage_b_loader = build_stage_loader(\"B\", train_splits, train_aug, batch)\n"
    "stage_c_loader = build_stage_loader(\"C\", train_splits, train_aug, batch)\n"
    "\n"
    "# Per-task val loaders (V1 eval transform, no TTA).\n"
    "val_loaders = {\n"
    "    head: DataLoader(\n"
    "        _MultiTaskHFDataset(loaded[head][\"val\"], head, eval_aug),\n"
    "        batch_size=batch, shuffle=False, num_workers=2, pin_memory=True,\n"
    "    )\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "\n"
    "print(\"USING V2 STRONG AUGMENTATION in all 3 stages + Mixup/CutMix p=0.3 + label smoothing 0.1\")\n"
    "print(f\"Stage A loader: {len(stage_a_loader.dataset)} samples\")\n"
    "print(f\"Stage B loader: {len(stage_b_loader.dataset)} samples\")\n"
    "print(f\"Stage C loader: {len(stage_c_loader.dataset)} samples\")\n"
)

# --------------------------------------------------------------------- #
# Cell 8 — Model + optimizer + scheduler + scaler + V3 ckpt mgr
# --------------------------------------------------------------------- #
code(
    "# Cell 8 — build model + optimizer + scaler + V3 checkpoint manager\n"
    "import torch\n"
    "from src.soil.model import SoilMultiTaskClassifier\n"
    "from src.soil.train_v3 import SoilCheckpointManagerV3\n"
    "\n"
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
    "model = SoilMultiTaskClassifier(\n"
    "    backbone_name=\"efficientnet_b0\", pretrained=True, dropout=0.3,\n"
    ").to(device)\n"
    "\n"
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)\n"
    "scaler = torch.amp.GradScaler(\"cuda\")\n"
    "ckpt_mgr = SoilCheckpointManagerV3(repo_id=\"ankit-iiitdmj/iks-soil-multitask-v3\")\n"
    "\n"
    "bb = model.backbone_param_count(); hd = model.head_param_count()\n"
    "print(f\"params: backbone={bb:,} heads={hd:,} total={bb+hd:,}\")\n"
    "print(\"Note: a FRESH CosineAnnealingLR is created per stage (T_max = that stage's epochs).\")\n"
)

# --------------------------------------------------------------------- #
# Cell 9 — Stage A
# --------------------------------------------------------------------- #
code(
    "# Cell 9 — Stage A: Phantom-fs only (15 epochs, soil_type head)\n"
    "import torch\n"
    "from src.soil.train_v3 import train_stage_a\n"
    "\n"
    "STAGE_A_EPOCHS = 15\n"
    "sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE_A_EPOCHS)\n"
    "\n"
    "history = train_stage_a(\n"
    "    model, stage_a_loader, optimizer, scaler, device,\n"
    "    epochs=STAGE_A_EPOCHS, val_loaders=val_loaders,\n"
    "    scheduler=sched_a, ckpt_mgr=ckpt_mgr,\n"
    ")\n"
    "# Snapshot the backbone-warmed checkpoint at the Stage A boundary.\n"
    "ckpt_mgr._upload(  # noqa: SLF001 — intentional: tag the stage boundary\n"
    "    ckpt_mgr.save(\n"
    "        epoch=len(history) - 1, model_state=model.state_dict(),\n"
    "        optimizer_state=optimizer.state_dict(), scheduler_state=sched_a.state_dict(),\n"
    "        scaler_state=scaler.state_dict(), history=history,\n"
    "        val_metrics=history[-1][\"val_metrics\"],\n"
    "    ),\n"
    "    \"checkpoint_stage_a.pt\",\n"
    ")\n"
    "print(f\"\\nStage A complete ({len(history)} epochs). Saved checkpoint_stage_a.pt\")\n"
)

# --------------------------------------------------------------------- #
# Cell 10 — Stage B
# --------------------------------------------------------------------- #
code(
    "# Cell 10 — Stage B: + Sirajganj moisture (15 epochs, soil_type + moisture)\n"
    "import torch\n"
    "from src.soil.train_v3 import train_stage_b\n"
    "\n"
    "STAGE_B_EPOCHS = 15\n"
    "# Fresh scheduler for Stage B.\n"
    "sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE_B_EPOCHS)\n"
    "\n"
    "history = train_stage_b(\n"
    "    model, stage_b_loader, optimizer, scaler, device,\n"
    "    epochs=STAGE_B_EPOCHS, val_loaders=val_loaders,\n"
    "    scheduler=sched_b, ckpt_mgr=ckpt_mgr, history=history,\n"
    ")\n"
    "ckpt_mgr._upload(  # noqa: SLF001\n"
    "    ckpt_mgr.save(\n"
    "        epoch=len(history) - 1, model_state=model.state_dict(),\n"
    "        optimizer_state=optimizer.state_dict(), scheduler_state=sched_b.state_dict(),\n"
    "        scaler_state=scaler.state_dict(), history=history,\n"
    "        val_metrics=history[-1][\"val_metrics\"],\n"
    "    ),\n"
    "    \"checkpoint_stage_b.pt\",\n"
    ")\n"
    "print(f\"\\nStage B complete ({len(history)} total epochs). Saved checkpoint_stage_b.pt\")\n"
)

# --------------------------------------------------------------------- #
# Cell 11 — Stage C
# --------------------------------------------------------------------- #
code(
    "# Cell 11 — Stage C: full multi-task (20 epochs, all 3 heads)\n"
    "import torch\n"
    "from src.soil.train_v3 import train_stage_c\n"
    "\n"
    "STAGE_C_EPOCHS = 20\n"
    "sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE_C_EPOCHS)\n"
    "\n"
    "best_val_sum = 0.0\n"
    "history = train_stage_c(\n"
    "    model, stage_c_loader, optimizer, scaler, device,\n"
    "    epochs=STAGE_C_EPOCHS, val_loaders=val_loaders,\n"
    "    scheduler=sched_c, ckpt_mgr=ckpt_mgr, history=history,\n"
    ")\n"
    "# Track best by sum of the 3 val top-1 accuracies across Stage C epochs.\n"
    "stage_c_hist = [h for h in history if h[\"stage\"] == \"C\" and h[\"val_metrics\"]]\n"
    "if stage_c_hist:\n"
    "    def _vsum(h):\n"
    "        vm = h[\"val_metrics\"]\n"
    "        return sum(vm.get(k, {}).get(\"top1_accuracy\", 0.0) for k in (\"soil_type\",\"moisture\",\"texture\"))\n"
    "    best = max(stage_c_hist, key=_vsum)\n"
    "    ckpt_mgr.save_best(epoch=best[\"epoch\"], model_state=model.state_dict(), val_metrics=best[\"val_metrics\"])\n"
    "    print(f\"\\nStage C complete. best Stage-C epoch={best['epoch']} val_sum={_vsum(best):.4f}\")\n"
    "ckpt_mgr._upload(  # noqa: SLF001\n"
    "    ckpt_mgr.save(\n"
    "        epoch=len(history) - 1, model_state=model.state_dict(),\n"
    "        optimizer_state=optimizer.state_dict(), scheduler_state=sched_c.state_dict(),\n"
    "        scaler_state=scaler.state_dict(), history=history,\n"
    "        val_metrics=history[-1][\"val_metrics\"],\n"
    "    ),\n"
    "    \"checkpoint_stage_c.pt\",\n"
    ")\n"
    "print(\"Saved checkpoint_stage_c.pt (final V3 model).\")\n"
)

# --------------------------------------------------------------------- #
# Cell 12 — Markdown: TTA eval header
# --------------------------------------------------------------------- #
md(
    "## Final test evaluation with TTA\n"
    "\n"
    "Evaluates the V3 best checkpoint on the held-out test splits using\n"
    "Test-Time Augmentation (5 views averaged), exactly as V2 did, so the\n"
    "V2-vs-V3 comparison is apples-to-apples. Pushes `eval_metrics_test.json`\n"
    "to `iks-soil-multitask-v3`.\n"
)

# --------------------------------------------------------------------- #
# Cell 13 — TTA eval + V2-vs-V3 comparison + verdict
# --------------------------------------------------------------------- #
code(
    "# Cell 13 — TTA test eval + V2 vs V3 comparison + ship/revert verdict\n"
    "import json, torch\n"
    "from huggingface_hub import HfApi\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "\n"
    "from src.soil.transforms_v2 import build_tta_views\n"
    "from src.soil.train_v3 import evaluate_per_task_tta\n"
    "\n"
    "_best = ckpt_mgr.try_load_best()\n"
    "if _best is None:\n"
    "    print(\"checkpoint_best.pt missing — falling back to checkpoint_latest.pt.\")\n"
    "    _best = ckpt_mgr.try_load_latest()\n"
    "if _best is None:\n"
    "    raise RuntimeError(\"No V3 checkpoint on HF Hub — run the stage cells first.\")\n"
    "model.load_state_dict(_best[\"model_state\"])\n"
    "model.eval()\n"
    "print(f\"Loaded V3 checkpoint (epoch {_best['epoch']})\")\n"
    "\n"
    "tta_views = build_tta_views(IMG_SIZE, MEAN, STD)\n"
    "\n"
    "\n"
    "class _RawTestDataset(Dataset):\n"
    "    def __init__(self, hf_split):\n"
    "        self.hf_split = hf_split\n"
    "    def __len__(self):\n"
    "        return len(self.hf_split)\n"
    "    def __getitem__(self, idx):\n"
    "        row = self.hf_split[idx]\n"
    "        return np.asarray(row[\"image\"].convert(\"RGB\")), int(row[\"label_idx\"])\n"
    "\n"
    "\n"
    "def _raw_collate(samples):\n"
    "    return [s[0] for s in samples], torch.tensor([s[1] for s in samples], dtype=torch.long)\n"
    "\n"
    "\n"
    "tta_loaders = {\n"
    "    head: DataLoader(\n"
    "        _RawTestDataset(loaded[head][\"test\"]),\n"
    "        batch_size=batch, shuffle=False, num_workers=0, collate_fn=_raw_collate,\n"
    "    )\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "\n"
    "metrics = evaluate_per_task_tta(model, tta_loaders, tta_views, device)\n"
    "\n"
    "display_names = {\"soil_type\": \"soil_type\", \"moisture\": \"moisture_appearance\", \"texture\": \"texture\"}\n"
    "for head in (\"soil_type\", \"moisture\", \"texture\"):\n"
    "    m = metrics[head]\n"
    "    n_classes = {\"soil_type\": 7, \"moisture\": 3, \"texture\": 3}[head]\n"
    "    print(f\"\\n=== {display_names[head]} TEST eval (TTA) ===\")\n"
    "    print(f\"  top1 acc: {m['top1_accuracy']:.4f}\")\n"
    "    if n_classes >= 5:\n"
    "        print(f\"  top5 acc: {m['top5_accuracy']:.4f}\")\n"
    "    else:\n"
    "        print(f\"  top5 acc: 1.0000   ({n_classes} classes → top-5 trivially perfect)\")\n"
    "    print(f\"  macro F1: {m['macro_f1']:.4f}\")\n"
    "    print(f\"  n_samples: {m['n_samples']}\")\n"
    "\n"
    "summary = {\n"
    "    head: {\n"
    "        \"top1_accuracy\": metrics[head][\"top1_accuracy\"],\n"
    "        \"top5_accuracy\": metrics[head][\"top5_accuracy\"],\n"
    "        \"macro_f1\":      metrics[head][\"macro_f1\"],\n"
    "        \"n_samples\":     metrics[head][\"n_samples\"],\n"
    "        \"num_classes\":   {\"soil_type\": 7, \"moisture\": 3, \"texture\": 3}[head],\n"
    "    }\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "tmp_path = \"/tmp/eval_metrics_test.json\"\n"
    "with open(tmp_path, \"w\") as fh:\n"
    "    json.dump(summary, fh, indent=2)\n"
    "HfApi().upload_file(\n"
    "    path_or_fileobj=tmp_path, path_in_repo=\"eval_metrics_test.json\",\n"
    "    repo_id=ckpt_mgr.repo_id, repo_type=\"model\",\n"
    ")\n"
    "print(\"\\nV3 TTA evaluation complete. Metrics pushed to HF Hub.\")\n"
    "\n"
    f"V2_BASELINE = {V2_BASELINE!r}\n"
    "print()\n"
    "print(\"COMPARISON V2 vs V3 (both with TTA):\")\n"
    "print(\"                     V2          V3           Δ\")\n"
    "deltas = {}\n"
    "for head in (\"soil_type\", \"moisture\", \"texture\"):\n"
    "    v2_top1 = V2_BASELINE[head][\"top1\"]; v2_f1 = V2_BASELINE[head][\"f1\"]\n"
    "    v3_top1 = metrics[head][\"top1_accuracy\"]; v3_f1 = metrics[head][\"macro_f1\"]\n"
    "    deltas[head] = (v3_top1 - v2_top1) * 100\n"
    "    label = display_names[head]\n"
    "    print(f\"  {label:8s} top-1:  {v2_top1*100:5.2f}%      {v3_top1*100:5.2f}%      {(v3_top1-v2_top1)*100:+5.2f} pts\")\n"
    "    print(f\"  {label:8s} F1:     {v2_f1:.3f}       {v3_f1:.3f}        {(v3_f1-v2_f1):+.3f}\")\n"
    "\n"
    "# Success criteria: texture +>=3 pts AND soil_type/moisture drop <=2 pts.\n"
    "tex_gain = deltas[\"texture\"]\n"
    "soil_drop = -deltas[\"soil_type\"]\n"
    "moist_drop = -deltas[\"moisture\"]\n"
    "ship = (tex_gain >= 3.0) and (soil_drop <= 2.0) and (moist_drop <= 2.0)\n"
    "print()\n"
    "print(f\"texture gain: {tex_gain:+.2f} pts (need >= +3.00)\")\n"
    "print(f\"soil_type change: {deltas['soil_type']:+.2f} pts (need >= -2.00)\")\n"
    "print(f\"moisture change: {deltas['moisture']:+.2f} pts (need >= -2.00)\")\n"
    "print()\n"
    "print(\"VERDICT:\", \"SHIP V3\" if ship else \"REVERT TO V2\")\n"
)

# --------------------------------------------------------------------- #
# Cell 14 — Closing markdown
# --------------------------------------------------------------------- #
md(
    "## V3 Complete\n"
    "\n"
    "Final model: `ankit-iiitdmj/iks-soil-multitask-v3` (private). Stage-boundary\n"
    "snapshots `checkpoint_stage_{a,b,c}.pt` are kept in the same repo for\n"
    "inspection. V1 (`iks-soil-multitask`) and V2 (`iks-soil-multitask-v2`)\n"
    "are untouched.\n"
    "\n"
    "**Decision tree:**\n"
    "- **SHIP V3** (texture ≥+3 pts, no head drop >2 pts): use V3 as the\n"
    "  production soil model in Phase 8 integration.\n"
    "- **REVERT TO V2**: keep V2 as production. V3 stays on HF Hub as a\n"
    "  documented negative result — useful for the paper's ablation table\n"
    "  (sequential transfer didn't beat single-stage + augmentation on this\n"
    "  small a dataset).\n"
    "\n"
    "Either way, report all three versions' numbers in the thesis so the\n"
    "augmentation and transfer-learning contributions are transparent.\n"
)


# Finalise notebook
NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(NB, fh)
print(f"wrote {NOTEBOOK_PATH} ({len(cells)} cells)")
