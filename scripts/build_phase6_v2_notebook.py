"""Generate ``notebooks/phase6_soil_training_v2.ipynb`` (Phase 6 V2 §C).

One-shot builder for the V2 augmentation-boosted retraining notebook.
Mirrors ``scripts/build_phase6_notebook.py`` (V1) for cell structure so
they read side-by-side; the differences are isolated to:

- Cell 1 markdown (V1 baselines + V2 changes summary)
- Cell 7 (V2 transforms via ``build_soil_train_aug_v2``)
- Cell 9 (``SoilCheckpointManagerV2``, ``T_max=40``)
- Cell 10 (``train_one_epoch_v2`` with Mixup/CutMix; 40 epochs)
- Cell 12 (TTA eval with ``evaluate_per_task_tta`` + side-by-side V1 comparison)
- Cell 13 (V2 closing notes)

V1 notebook stays untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase6_soil_training_v2.ipynb"

REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

# V1 baseline numbers from ankit-iiitdmj/iks-soil-multitask/eval_metrics_test.json
V1_BASELINE = {
    "soil_type": {"top1": 0.8908, "f1": 0.8184},
    "moisture":  {"top1": 0.8898, "f1": 0.8902},
    "texture":   {"top1": 0.6786, "f1": 0.6703},
}

NB = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# --------------------------------------------------------------------- #
# Cell 1 — Title + V1 baseline + V2 changes
# --------------------------------------------------------------------- #
md(
    "# Phase 6 V2 — Soil Multi-Task with Augmentation\n"
    "\n"
    "**Goal:** push texture from 67.86% → ~75-82% via strong augmentation,\n"
    "Mixup/CutMix, label smoothing, and TTA. soil_type and moisture may shift\n"
    "±1–2 points due to the shared backbone — that's expected.\n"
    "\n"
    "**Changes from V1:**\n"
    "- Strong augmentation pipeline (`transforms_v2.py`): wider scale crop,\n"
    "  vertical flip, ±30° rotation, GridDistortion/ElasticTransform,\n"
    "  stronger ColorJitter, GaussNoise/GaussianBlur, CoarseDropout.\n"
    "- Mixup + CutMix at batch level (p=0.3, 50/50 split when triggered).\n"
    "- Label smoothing 0.1 on all three heads' cross-entropy.\n"
    "- Test-Time Augmentation (5 views averaged) in Cell 12.\n"
    "- 40 epochs (V1 was 30) to compensate for heavier augmentation.\n"
    "- New HF Hub repo: `ankit-iiitdmj/iks-soil-multitask-v2` (private).\n"
    "\n"
    "**V1 baseline (held out test set):**\n"
    f"- `soil_type`:  {V1_BASELINE['soil_type']['top1']*100:.2f}% / F1 {V1_BASELINE['soil_type']['f1']:.3f}\n"
    f"- `moisture`:   {V1_BASELINE['moisture']['top1']*100:.2f}% / F1 {V1_BASELINE['moisture']['f1']:.3f}\n"
    f"- `texture`:    {V1_BASELINE['texture']['top1']*100:.2f}% / F1 {V1_BASELINE['texture']['f1']:.3f}\n"
    "\n"
    "V1 model preserved at `ankit-iiitdmj/iks-soil-multitask` for the paper's\n"
    "ablation. Resume-aware via HF Hub checkpoints just like V1.\n"
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
# Cell 5 — V2 configuration markdown
# --------------------------------------------------------------------- #
md(
    "## V2 Configuration\n"
    "\n"
    "- Backbone: **EfficientNet-B0 @ 224×224** (UNCHANGED from V1)\n"
    "- 40 epochs (5 frozen warmup + 35 unfrozen)\n"
    "- AdamW `lr=1e-4`, `weight_decay=1e-4`\n"
    "- Cosine annealing `T_max=40`\n"
    "- **Mixup** `p=0.3 / α=0.2` and **CutMix** `α=1.0` (50/50 when triggered)\n"
    "- **Label smoothing** 0.1 on each head's cross-entropy\n"
    "- **TTA** in Cell 12 — 5 deterministic views averaged before argmax\n"
    "- Per-epoch checkpoint pushed to `ankit-iiitdmj/iks-soil-multitask-v2`\n"
)

# --------------------------------------------------------------------- #
# Cell 6 — Load 3 HF datasets
# --------------------------------------------------------------------- #
code(
    "# Cell 6 — Load 3 HF datasets (train + val + test for each)\n"
    "from datasets import load_dataset\n"
    "\n"
    "DATASETS = {\n"
    "    \"soil_type\": \"ankit-iiitdmj/iks-soil-phantomfs\",\n"
    "    \"moisture\":  \"ankit-iiitdmj/iks-soil-sirajganj-moisture\",\n"
    "    \"texture\":   \"ankit-iiitdmj/iks-soil-texture-irsid-vit\",\n"
    "}\n"
    "\n"
    "loaded = {head: load_dataset(repo) for head, repo in DATASETS.items()}\n"
    "for head, dsd in loaded.items():\n"
    "    repo = DATASETS[head].split(\"/\")[-1]\n"
    "    print(f\"{repo}: train={len(dsd['train'])} val={len(dsd['val'])} test={len(dsd['test'])}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 7 — V2 transforms + loaders (train uses v2 aug; val uses V1 eval)
# --------------------------------------------------------------------- #
code(
    "# Cell 7 — V2 transforms + loaders\n"
    "import numpy as np\n"
    "import torch\n"
    "import yaml\n"
    "from torch.utils.data import ConcatDataset, DataLoader, Dataset\n"
    "\n"
    "from src.soil.dataset import build_multitask_labels\n"
    "# V1's eval transform is reused as-is. V2's training transform is the new one.\n"
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
    "print(\"=\" * 60)\n"
    "print(\"USING V2 STRONG AUGMENTATION (transforms_v2.build_soil_train_aug_v2)\")\n"
    "print(\"  + Mixup p=0.3 alpha=0.2 / CutMix alpha=1.0 (50/50 when triggered)\")\n"
    "print(\"  + Label smoothing 0.1\")\n"
    "print(\"=\" * 60)\n"
    "\n"
    "\n"
    "class _MultiTaskWrapper(Dataset):\n"
    "    \"\"\"Wrap an HF dataset split as multi-task rows (one head valid, others -1).\"\"\"\n"
    "\n"
    "    def __init__(self, hf_split, head: str, transform):\n"
    "        self.hf_split = hf_split\n"
    "        self.head = head\n"
    "        self.transform = transform\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.hf_split)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        row = self.hf_split[idx]\n"
    "        arr = np.asarray(row[\"image\"].convert(\"RGB\"))\n"
    "        tensor = self.transform(image=arr)[\"image\"]\n"
    "        labels = build_multitask_labels(\n"
    "            self.head, int(row[\"label_idx\"]),\n"
    "            head_order=(\"soil_type\", \"moisture\", \"texture\"),\n"
    "        )\n"
    "        return {\n"
    "            \"image\": tensor,\n"
    "            \"soil_type_label\": torch.tensor(labels[\"soil_type\"], dtype=torch.long),\n"
    "            \"moisture_label\":  torch.tensor(labels[\"moisture\"],  dtype=torch.long),\n"
    "            \"texture_label\":   torch.tensor(labels[\"texture\"],   dtype=torch.long),\n"
    "        }\n"
    "\n"
    "\n"
    "train_datasets = [\n"
    "    _MultiTaskWrapper(loaded[\"soil_type\"][\"train\"], \"soil_type\", train_aug),\n"
    "    _MultiTaskWrapper(loaded[\"moisture\"][\"train\"],  \"moisture\",  train_aug),\n"
    "    _MultiTaskWrapper(loaded[\"texture\"][\"train\"],   \"texture\",   train_aug),\n"
    "]\n"
    "train_loader = DataLoader(\n"
    "    ConcatDataset(train_datasets),\n"
    "    batch_size=batch, shuffle=True, num_workers=2, pin_memory=True,\n"
    ")\n"
    "\n"
    "val_loaders = {\n"
    "    head: DataLoader(\n"
    "        _MultiTaskWrapper(loaded[head][\"val\"], head, eval_aug),\n"
    "        batch_size=batch, shuffle=False, num_workers=2, pin_memory=True,\n"
    "    )\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "test_loaders_v1eval = {\n"
    "    head: DataLoader(\n"
    "        _MultiTaskWrapper(loaded[head][\"test\"], head, eval_aug),\n"
    "        batch_size=batch, shuffle=False, num_workers=2, pin_memory=True,\n"
    "    )\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "\n"
    "print(f\"train loader: {len(train_loader.dataset)} samples in {len(train_loader)} batches\")\n"
    "for head in (\"soil_type\", \"moisture\", \"texture\"):\n"
    "    print(f\"  val  {head}: {len(val_loaders[head].dataset)} samples\")\n"
    "    print(f\"  test {head}: {len(test_loaders_v1eval[head].dataset)} samples\")\n"
)

# --------------------------------------------------------------------- #
# Cell 8 — Build model
# --------------------------------------------------------------------- #
code(
    "# Cell 8 — Build SoilMultiTaskClassifier (UNCHANGED from V1)\n"
    "from src.soil.model import SoilMultiTaskClassifier\n"
    "\n"
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
    "model = SoilMultiTaskClassifier(\n"
    "    backbone_name=\"efficientnet_b0\", pretrained=True, dropout=0.3,\n"
    ").to(device)\n"
    "\n"
    "bb = model.backbone_param_count()\n"
    "hd = model.head_param_count()\n"
    "print(f\"backbone params: {bb:,}\")\n"
    "print(f\"head params:     {hd:,}\")\n"
    "print(f\"total params:    {bb + hd:,}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 9 — Optimizer / scheduler / scaler / V2 checkpoint manager + resume
# --------------------------------------------------------------------- #
code(
    "# Cell 9 — optimizer + scheduler + scaler + V2 checkpoint manager\n"
    "import torch\n"
    "from src.soil.train_v2 import SoilCheckpointManagerV2\n"
    "\n"
    "EPOCHS_TOTAL = 40\n"
    "FREEZE_EPOCHS = 5\n"
    "MIX_P = 0.3\n"
    "LABEL_SMOOTHING = 0.1\n"
    "\n"
    "optimizer = torch.optim.AdamW(\n"
    "    model.parameters(), lr=1e-4, weight_decay=1e-4,\n"
    ")\n"
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_TOTAL)\n"
    "scaler = torch.amp.GradScaler(\"cuda\")\n"
    "ckpt_mgr = SoilCheckpointManagerV2(repo_id=\"ankit-iiitdmj/iks-soil-multitask-v2\")\n"
    "\n"
    "start_epoch = 0\n"
    "history: list[dict] = []\n"
    "best_val_sum = 0.0\n"
    "\n"
    "_resume = ckpt_mgr.try_load_latest()\n"
    "if _resume is not None:\n"
    "    print(f\"Resuming V2 from epoch {_resume['epoch']} ...\")\n"
    "    model.load_state_dict(_resume[\"model_state\"])\n"
    "    if _resume.get(\"optimizer_state\") is not None:\n"
    "        optimizer.load_state_dict(_resume[\"optimizer_state\"])\n"
    "    if _resume.get(\"scheduler_state\") is not None:\n"
    "        scheduler.load_state_dict(_resume[\"scheduler_state\"])\n"
    "    if _resume.get(\"scaler_state\") is not None:\n"
    "        scaler.load_state_dict(_resume[\"scaler_state\"])\n"
    "    start_epoch = int(_resume[\"epoch\"]) + 1\n"
    "    history = list(_resume.get(\"history\", []))\n"
    "    if history:\n"
    "        best_val_sum = max(h.get(\"val_sum\", 0.0) for h in history)\n"
    "    print(f\"  history len={len(history)}  best_val_sum={best_val_sum:.4f}\")\n"
    "else:\n"
    "    print(\"No prior V2 checkpoint — starting fresh from ImageNet weights.\")\n"
)

# --------------------------------------------------------------------- #
# Cell 10 — Training loop (V2)
# --------------------------------------------------------------------- #
code(
    "# Cell 10 — V2 training loop with Mixup/CutMix and label smoothing\n"
    "import time, datetime\n"
    "from src.soil.train import evaluate_per_task\n"
    "from src.soil.train_v2 import train_one_epoch_v2\n"
    "\n"
    "for epoch in range(start_epoch, EPOCHS_TOTAL):\n"
    "    is_frozen = epoch < FREEZE_EPOCHS\n"
    "    if is_frozen:\n"
    "        model.freeze_backbone()\n"
    "        stage_tag = \"FROZEN\"\n"
    "    else:\n"
    "        model.unfreeze_backbone()\n"
    "        stage_tag = \"UNFROZEN\"\n"
    "\n"
    "    t0 = time.time()\n"
    "    train_losses = train_one_epoch_v2(\n"
    "        model, train_loader, optimizer, scaler, device,\n"
    "        mix_p=MIX_P, label_smoothing=LABEL_SMOOTHING,\n"
    "    )\n"
    "    # Validation uses V1's non-TTA eval — TTA is reserved for the final\n"
    "    # held-out test eval in Cell 12.\n"
    "    val_metrics = evaluate_per_task(model, val_loaders, device)\n"
    "    scheduler.step()\n"
    "    elapsed = time.time() - t0\n"
    "\n"
    "    val_sum = (\n"
    "        val_metrics[\"soil_type\"][\"top1_accuracy\"]\n"
    "        + val_metrics[\"moisture\"][\"top1_accuracy\"]\n"
    "        + val_metrics[\"texture\"][\"top1_accuracy\"]\n"
    "    )\n"
    "\n"
    "    print(\n"
    "        f\"[V2 epoch {epoch+1}/{EPOCHS_TOTAL}] {stage_tag} | \"\n"
    "        f\"type_loss={train_losses['loss_soil_type']:.4f} \"\n"
    "        f\"moist_loss={train_losses['loss_moisture']:.4f} \"\n"
    "        f\"tex_loss={train_losses['loss_texture']:.4f} \"\n"
    "        f\"total={train_losses['loss_total']:.4f} | \"\n"
    "        f\"val: type_acc={val_metrics['soil_type']['top1_accuracy']:.4f} \"\n"
    "        f\"moist_acc={val_metrics['moisture']['top1_accuracy']:.4f} \"\n"
    "        f\"tex_acc={val_metrics['texture']['top1_accuracy']:.4f} | \"\n"
    "        f\"{elapsed:.0f}s\"\n"
    "    )\n"
    "\n"
    "    history.append({\n"
    "        \"epoch\": int(epoch + 1),\n"
    "        \"stage\": stage_tag,\n"
    "        **train_losses,\n"
    "        \"val_soil_type\": val_metrics[\"soil_type\"],\n"
    "        \"val_moisture\":  val_metrics[\"moisture\"],\n"
    "        \"val_texture\":   val_metrics[\"texture\"],\n"
    "        \"val_sum\": float(val_sum),\n"
    "        \"elapsed_seconds\": float(elapsed),\n"
    "        \"timestamp\": datetime.datetime.utcnow().isoformat() + \"Z\",\n"
    "    })\n"
    "\n"
    "    ckpt_mgr.save(\n"
    "        epoch=epoch,\n"
    "        model_state=model.state_dict(),\n"
    "        optimizer_state=optimizer.state_dict(),\n"
    "        scheduler_state=scheduler.state_dict(),\n"
    "        scaler_state=scaler.state_dict(),\n"
    "        history=history,\n"
    "        val_metrics=val_metrics,\n"
    "    )\n"
    "\n"
    "    if val_sum > best_val_sum:\n"
    "        best_val_sum = float(val_sum)\n"
    "        ckpt_mgr.save_best(\n"
    "            epoch=epoch, model_state=model.state_dict(),\n"
    "            val_metrics=val_metrics,\n"
    "        )\n"
    "        print(f\"  -> new best (val_sum={best_val_sum:.4f}) checkpoint_best.pt pushed\")\n"
    "\n"
    "best_epoch = max(history, key=lambda h: h[\"val_sum\"]) if history else None\n"
    "if best_epoch:\n"
    "    print(\n"
    "        f\"\\nV2 training complete. best epoch={best_epoch['epoch']} \"\n"
    "        f\"val_sum={best_epoch['val_sum']:.4f} \"\n"
    "        f\"type={best_epoch['val_soil_type']['top1_accuracy']:.4f} \"\n"
    "        f\"moist={best_epoch['val_moisture']['top1_accuracy']:.4f} \"\n"
    "        f\"tex={best_epoch['val_texture']['top1_accuracy']:.4f}\"\n"
    "    )\n"
)

# --------------------------------------------------------------------- #
# Cell 11 — Markdown: TTA eval intro
# --------------------------------------------------------------------- #
md(
    "## Test-set Evaluation with TTA\n"
    "\n"
    "Evaluates the V2 best checkpoint on the held-out test splits using\n"
    "Test-Time Augmentation: each image is forward-passed through 5\n"
    "deterministic views (original + HFlip + VFlip + Rot90 + Rot270), and\n"
    "the logits are averaged before argmax. Pushed as `eval_metrics_test.json`\n"
    "to `iks-soil-multitask-v2`.\n"
    "\n"
    "**V1 baseline (no TTA), held out test set:**\n"
    f"- `soil_type`: {V1_BASELINE['soil_type']['top1']*100:.2f}% / F1 {V1_BASELINE['soil_type']['f1']:.3f}\n"
    f"- `moisture`:  {V1_BASELINE['moisture']['top1']*100:.2f}% / F1 {V1_BASELINE['moisture']['f1']:.3f}\n"
    f"- `texture`:   {V1_BASELINE['texture']['top1']*100:.2f}% / F1 {V1_BASELINE['texture']['f1']:.3f}\n"
)

# --------------------------------------------------------------------- #
# Cell 12 — TTA eval + side-by-side V1 comparison
# --------------------------------------------------------------------- #
code(
    "# Cell 12 — held-out test-set evaluation with TTA (5 views averaged)\n"
    "import json, torch\n"
    "from huggingface_hub import HfApi\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "\n"
    "from src.soil.transforms_v2 import build_tta_views\n"
    "from src.soil.train_v2 import evaluate_per_task_tta\n"
    "\n"
    "# Pull the V2 best checkpoint and load it before TTA eval.\n"
    "_best = ckpt_mgr.try_load_best()\n"
    "if _best is None:\n"
    "    print(\"checkpoint_best.pt missing — falling back to checkpoint_latest.pt.\")\n"
    "    _best = ckpt_mgr.try_load_latest()\n"
    "if _best is None:\n"
    "    raise RuntimeError(\"No V2 checkpoint on HF Hub — run Cell 10 first.\")\n"
    "model.load_state_dict(_best[\"model_state\"])\n"
    "model.eval()\n"
    "print(f\"Loaded V2 checkpoint (epoch {_best['epoch']})\")\n"
    "\n"
    "tta_views = build_tta_views(IMG_SIZE, MEAN, STD)\n"
    "print(f\"TTA views: {len(tta_views)} (original + HFlip + VFlip + Rot90 + Rot270)\")\n"
    "\n"
    "\n"
    "class _RawTestDataset(Dataset):\n"
    "    \"\"\"Yields (numpy_HWC_uint8, label_int) — TTA Composes are applied inside\n"
    "    evaluate_per_task_tta so each image goes through all 5 views per call.\"\"\"\n"
    "\n"
    "    def __init__(self, hf_split):\n"
    "        self.hf_split = hf_split\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.hf_split)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        row = self.hf_split[idx]\n"
    "        arr = np.asarray(row[\"image\"].convert(\"RGB\"))\n"
    "        return arr, int(row[\"label_idx\"])\n"
    "\n"
    "\n"
    "def _raw_collate(samples):\n"
    "    images = [s[0] for s in samples]\n"
    "    labels = torch.tensor([s[1] for s in samples], dtype=torch.long)\n"
    "    return images, labels\n"
    "\n"
    "\n"
    "tta_loaders = {\n"
    "    head: DataLoader(\n"
    "        _RawTestDataset(loaded[head][\"test\"]),\n"
    "        batch_size=batch, shuffle=False, num_workers=0,\n"
    "        collate_fn=_raw_collate,\n"
    "    )\n"
    "    for head in (\"soil_type\", \"moisture\", \"texture\")\n"
    "}\n"
    "\n"
    "metrics = evaluate_per_task_tta(model, tta_loaders, tta_views, device)\n"
    "\n"
    "display_names = {\n"
    "    \"soil_type\": \"soil_type\",\n"
    "    \"moisture\": \"moisture_appearance\",\n"
    "    \"texture\": \"texture\",\n"
    "}\n"
    "for head in (\"soil_type\", \"moisture\", \"texture\"):\n"
    "    m = metrics[head]\n"
    "    print(f\"\\n=== {display_names[head]} TEST eval (TTA) ===\")\n"
    "    print(f\"  top1 acc: {m['top1_accuracy']:.4f}\")\n"
    "    n_classes = {\"soil_type\": 7, \"moisture\": 3, \"texture\": 3}[head]\n"
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
    "\n"
    "tmp_path = \"/tmp/eval_metrics_test.json\"\n"
    "with open(tmp_path, \"w\") as fh:\n"
    "    json.dump(summary, fh, indent=2)\n"
    "HfApi().upload_file(\n"
    "    path_or_fileobj=tmp_path,\n"
    "    path_in_repo=\"eval_metrics_test.json\",\n"
    "    repo_id=ckpt_mgr.repo_id,\n"
    "    repo_type=\"model\",\n"
    ")\n"
    "print(\"\\nV2 TTA evaluation complete. Metrics pushed to HF Hub.\")\n"
    "\n"
    f"V1_BASELINE = {V1_BASELINE!r}\n"
    "print()\n"
    "print(\"COMPARISON V1 vs V2 (TTA):\")\n"
    "print(f\"                       V1          V2           Δ\")\n"
    "for head in (\"soil_type\", \"moisture\", \"texture\"):\n"
    "    v1_top1 = V1_BASELINE[head][\"top1\"]\n"
    "    v1_f1   = V1_BASELINE[head][\"f1\"]\n"
    "    v2_top1 = metrics[head][\"top1_accuracy\"]\n"
    "    v2_f1   = metrics[head][\"macro_f1\"]\n"
    "    label = display_names[head]\n"
    "    print(f\"  {label:8s} top-1:  {v1_top1*100:5.2f}%      {v2_top1*100:5.2f}%      {(v2_top1-v1_top1)*100:+5.2f} pts\")\n"
    "    print(f\"  {label:8s} F1:     {v1_f1:.3f}       {v2_f1:.3f}        {(v2_f1-v1_f1):+.3f}\")\n"
)

# --------------------------------------------------------------------- #
# Cell 13 — Closing markdown
# --------------------------------------------------------------------- #
md(
    "## V2 Complete\n"
    "\n"
    "Model checkpoint: `ankit-iiitdmj/iks-soil-multitask-v2`\n"
    "V1 model preserved: `ankit-iiitdmj/iks-soil-multitask`\n"
    "\n"
    "For the paper, the V1 vs V2 comparison serves as an ablation showing the\n"
    "contribution of strong augmentation + Mixup/CutMix + TTA on a small\n"
    "dataset (279 texture images merged from IRSID + VIT).\n"
    "\n"
    "Honest expectation: texture should move from 67.86% to ~75–82%.\n"
    "`soil_type` and `moisture` may shift ±1–2 points due to the shared\n"
    "backbone. If any head regressed significantly (more than 3 points), open\n"
    "an issue before reporting — it usually means an over-aggressive\n"
    "augmentation hyperparameter that needs tempering.\n"
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
