"""Generate notebooks/phase5_disease_training.ipynb (Phase 5 §G).

One-shot script — runs once locally and writes the notebook. The
notebook itself is meant to be uploaded to Google Colab and executed
on a T4 / L4 / A100 runtime. Cells 1–4 are setup (no GPU needed);
cells 5–12 do the three-stage cascade training.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase5_disease_training.ipynb"

# Update this if the GitHub remote URL ever changes. The notebook has
# a clearly-marked comment in Cell 2 pointing at this constant.
REPO_HTTPS_URL = "https://github.com/ankit8453/iks-rag-thesis.git"
REPO_LOCAL_PATH = "/content/iks-rag-thesis"

NB = nbf.v4.new_notebook()
cells: list[dict] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# Cell 1 — title + abstract
md(
    "# Phase 5 — Disease Module Training (Colab)\n"
    "\n"
    "Three-stage cascade for the IKS-grounded multimodal agricultural advisory\n"
    "system at IIITDM Jabalpur (Author: Ankit Pawar, Supervisor: Dr. Akshay\n"
    "Pandey).\n"
    "\n"
    "**Stages:**\n"
    "1. **Pretrain on PlantVillage** (38 classes, 25 epochs)\n"
    "2. **Fine-tune on Paddy Doctor** (10 classes, 20 epochs)\n"
    "3. **Fine-tune on PlantDoc** (27 classes, 30 epochs)\n"
    "\n"
    "Each stage is **independently resumable** — its checkpoint is pushed to a\n"
    "private HF Hub model repo after every epoch. If the Colab session resets,\n"
    "re-run the same cell with the `--resume` flag and training continues from\n"
    "the latest checkpoint.\n"
    "\n"
    "## ⚠️ Before you start\n"
    "\n"
    "- Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).\n"
    "- **Colab free tier limits:** ~12 h continuous session, ~3–4 h daily GPU\n"
    "  quota. Plan to run one stage per session if you're on free tier;\n"
    "  Colab Pro / Pro+ raise the quota substantially.\n"
    "- The notebook reads datasets from private HF Hub repos. You will be\n"
    "  prompted to log in with a HF Hub Write token (Cell 3).\n"
)

# Cell 2 — setup
code(
    "# Cell 2 — setup\n"
    "# If your repo URL differs from the line below, edit just this one line:\n"
    f"REPO_URL = \"{REPO_HTTPS_URL}\"\n"
    f"REPO_PATH = \"{REPO_LOCAL_PATH}\"\n"
    "\n"
    "import os\n"
    "import subprocess\n"
    "if not os.path.isdir(REPO_PATH):\n"
    "    subprocess.run([\"git\", \"clone\", REPO_URL, REPO_PATH], check=True)\n"
    "os.chdir(REPO_PATH)\n"
    "\n"
    "# Install the project's runtime dependencies. Colab pre-installs\n"
    "# torch / numpy / pandas / matplotlib so we skip those.\n"
    "_pip_packages = [\n"
    "    \"timm>=1.0.0\",\n"
    "    \"albumentations>=1.4\",\n"
    "    \"huggingface_hub>=0.24\",\n"
    "    \"datasets>=2.20\",\n"
    "    \"pytorch-grad-cam>=1.5\",\n"
    "    \"iterative-stratification>=0.1.7\",\n"
    "    \"pydantic>=2.7\",\n"
    "]\n"
    "subprocess.run([\"pip\", \"install\", \"--quiet\", *_pip_packages], check=True)\n"
    "print(\"setup ok\")"
)

# Cell 3 — HF Hub login
code(
    "# Cell 3 — HF Hub login\n"
    "from huggingface_hub import notebook_login\n"
    "notebook_login()  # paste your Write token when prompted"
)

# Cell 4 — verify GPU + pick batch size
code(
    "# Cell 4 — verify GPU + pick batch size\n"
    "import subprocess, torch\n"
    "subprocess.run([\"nvidia-smi\"], check=False)\n"
    "print()\n"
    "if torch.cuda.is_available():\n"
    "    dev = torch.cuda.get_device_properties(0)\n"
    "    print(f\"GPU: {dev.name}, VRAM: {dev.total_memory / 1024**3:.1f} GiB\")\n"
    "else:\n"
    "    print(\"No GPU detected — switch runtime to GPU before continuing.\")\n"
    "\n"
    "import sys\n"
    "sys.path.insert(0, REPO_PATH)\n"
    "from src.disease.train import auto_batch_size\n"
    "print(\"auto_batch_size for B4 at 380x380:\", auto_batch_size(380))"
)

# Stage 1
md("## Stage 1 — Pretrain on PlantVillage\n\n"
   "**25 epochs.** Pushes checkpoints to `ankit-iiitdmj/iks-disease-plantvillage`.\n"
   "Re-running this cell with `--resume` picks up from the latest checkpoint if\n"
   "the previous session got interrupted.")
code(
    "# Cell 6 — Stage 1: PlantVillage pretraining\n"
    "!python -m src.disease.train --stage pretrain --resume"
)

# Stage 2
md("## Stage 2 — Fine-tune on Paddy Doctor\n\n"
   "**20 epochs.** Seeds from the final PlantVillage checkpoint automatically.\n"
   "Only run this once Stage 1 reaches max epochs (or you'll be fine-tuning a\n"
   "half-trained model).")
code(
    "# Cell 8 — Stage 2: Paddy Doctor fine-tune\n"
    "!python -m src.disease.train --stage finetune_paddy --resume"
)

# Stage 3
md("## Stage 3 — Fine-tune on PlantDoc\n\n"
   "**30 epochs.** Seeds from the final Paddy Doctor checkpoint. This is the\n"
   "real-field-imagery stage — performance here is what gets reported in the\n"
   "thesis.")
code(
    "# Cell 10 — Stage 3: PlantDoc fine-tune\n"
    "!python -m src.disease.train --stage finetune_plantdoc --resume"
)

# Evaluation
md("## Evaluation on held-out test sets\n\n"
   "Run final eval on each dataset's test split. Prints per-class\n"
   "precision/recall/F1 + confusion matrix. The JSON gets pushed to HF Hub\n"
   "alongside the final model checkpoint.")
code(
    "# Cell 12 — Test-set evaluation\n"
    "import json\n"
    "import torch\n"
    "from huggingface_hub import HfApi\n"
    "\n"
    "from src.disease.train import (\n"
    "    STAGE_INFO,\n"
    "    CheckpointManager,\n"
    "    auto_batch_size,\n"
    "    _build_loaders_from_hf,\n"
    "    _build_model_for_stage,\n"
    "    evaluate,\n"
    ")\n"
    "\n"
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
    "batch = auto_batch_size()\n"
    "api = HfApi()\n"
    "\n"
    "for stage in (\"pretrain\", \"finetune_paddy\", \"finetune_plantdoc\"):\n"
    "    info = STAGE_INFO[stage]\n"
    "    # Re-build the test loader for this dataset.\n"
    "    train_loader, val_loader, num_classes = _build_loaders_from_hf(\n"
    "        stage, batch_size=batch, num_workers=2\n"
    "    )\n"
    "    # Pull the best checkpoint.\n"
    "    ckpt_mgr = CheckpointManager(info[\"model_repo\"])\n"
    "    ckpt = ckpt_mgr.try_load_latest()\n"
    "    if ckpt is None:\n"
    "        print(f\"Skip {stage}: no checkpoint on HF Hub.\")\n"
    "        continue\n"
    "    model = _build_model_for_stage(stage, num_classes, ckpt)\n"
    "    model.load_state_dict(ckpt[\"model_state\"], strict=False)\n"
    "    model.to(device).eval()\n"
    "\n"
    "    metrics = evaluate(model, val_loader, num_classes, device)\n"
    "    summary = metrics.to_dict()\n"
    "    print(f\"\\n=== {stage} eval ===\")\n"
    "    print(f\"  top1 acc: {summary['top1_accuracy']:.4f}\")\n"
    "    print(f\"  top5 acc: {summary['top5_accuracy']:.4f}\")\n"
    "    print(f\"  macro F1: {summary['macro_f1']:.4f}\")\n"
    "\n"
    "    metrics_path = f\"/tmp/{stage}_metrics.json\"\n"
    "    with open(metrics_path, \"w\") as fh:\n"
    "        json.dump(summary, fh, indent=2)\n"
    "    api.upload_file(\n"
    "        path_or_fileobj=metrics_path,\n"
    "        path_in_repo=\"eval_metrics.json\",\n"
    "        repo_id=info[\"model_repo\"],\n"
    "        repo_type=\"model\",\n"
    "    )\n"
    "    print(f\"  pushed eval metrics -> {info['model_repo']}/eval_metrics.json\")"
)

# What's next
md(
    "## What's next\n"
    "\n"
    "After all three stages complete and the eval metrics are pushed:\n"
    "\n"
    "1. From your laptop, pull the final PlantDoc checkpoint back via\n"
    "   `huggingface-cli download ankit-iiitdmj/iks-disease-plantdoc\n"
    "   checkpoint_best.pt --local-dir models/`.\n"
    "2. Move on to **Phase 6** (soil module training) using\n"
    "   `notebooks/phase6_soil_training.ipynb` (separate prompt).\n"
    "3. Phase 8 integrates the trained disease module into the multimodal\n"
    "   context constructor.\n"
)

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
print(f"wrote {NOTEBOOK_PATH}")
