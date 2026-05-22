"""Generate notebooks/dataset_eda.ipynb.

One-shot script — produces a notebook with per-dataset EDA cells:
class-distribution bar chart, 4×4 sample-image grid, image-size
histogram, plus a cross-region side-by-side and an OLID label table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf  # noqa: E402

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "dataset_eda.ipynb"

NB = nbf.v4.new_notebook()
cells: list[dict] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    "# Phase 4 Dataset EDA\n"
    "\n"
    "Interactive exploration of the six Phase 4 datasets. Each dataset "
    "section shows:\n"
    "1. Class-distribution bar chart (per-class counts across train + "
    "val + test).\n"
    "2. 4×4 grid of sample images from the first four classes.\n"
    "3. Image-size histogram (sampled).\n"
    "\n"
    "Plus a side-by-side cross-region soil bar chart and an OLID I "
    "label-vocabulary table.\n"
    "\n"
    "Run top-to-bottom with `jupyter nbconvert --to notebook --execute "
    "notebooks/dataset_eda.ipynb` to verify end-to-end."
)

code(
    "from __future__ import annotations\n"
    "import sys, json\n"
    "from pathlib import Path\n"
    "from collections import Counter\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "from PIL import Image\n"
    "\n"
    "# Make `from src...` imports work without `pip install -e .`.\n"
    "PROJECT_ROOT = Path.cwd()\n"
    "while not (PROJECT_ROOT / 'requirements.txt').is_file():\n"
    "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n"
    "\n"
    "from scripts._dataset_specs import DATASET_SPECS\n"
    "from src.utils.data_splits import load_class_map, load_split\n"
    "\n"
    "SPLITS = PROJECT_ROOT / 'data' / 'splits'\n"
    "print('PROJECT_ROOT =', PROJECT_ROOT)"
)

code(
    "def load_all_entries(dataset_name):\n"
    "    out = {}\n"
    "    for split_name in ('train', 'val', 'test'):\n"
    "        p = SPLITS / dataset_name / f'{split_name}.json'\n"
    "        if p.is_file():\n"
    "            out[split_name] = load_split(p)\n"
    "    return out\n"
    "\n"
    "def class_counts(entries_by_split):\n"
    "    total = Counter()\n"
    "    for entries in entries_by_split.values():\n"
    "        for e in entries:\n"
    "            total[e.label] += 1\n"
    "    return total\n"
    "\n"
    "def plot_class_distribution(name, counts):\n"
    "    labels = sorted(counts)\n"
    "    sizes = [counts[lab] for lab in labels]\n"
    "    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), 4))\n"
    "    ax.bar(range(len(labels)), sizes)\n"
    "    ax.set_xticks(range(len(labels)))\n"
    "    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=8)\n"
    "    ax.set_title(f'{name} — class distribution (total={sum(sizes)})')\n"
    "    ax.set_ylabel('image count')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "def show_sample_images(name, entries, raw_root, resolver=None, n_classes=4):\n"
    "    by_class = {}\n"
    "    for e in entries:\n"
    "        by_class.setdefault(e.label, []).append(e)\n"
    "    classes = sorted(by_class)[:n_classes]\n"
    "    if not classes:\n"
    "        print(f'{name}: no entries to display.')\n"
    "        return\n"
    "    fig, axes = plt.subplots(n_classes, 4, figsize=(12, 3 * n_classes))\n"
    "    if n_classes == 1:\n"
    "        axes = np.array([axes])\n"
    "    for row, cls in enumerate(classes):\n"
    "        for col, e in enumerate(by_class[cls][:4]):\n"
    "            path = resolver(e) if resolver else (raw_root / e.path)\n"
    "            try:\n"
    "                img = Image.open(path).convert('RGB')\n"
    "                axes[row, col].imshow(img)\n"
    "            except Exception as exc:\n"
    "                axes[row, col].text(0.5, 0.5, f'load fail: {exc}',\n"
    "                                    ha='center', va='center')\n"
    "            axes[row, col].set_axis_off()\n"
    "            if col == 0:\n"
    "                axes[row, col].set_ylabel(cls)\n"
    "    fig.suptitle(f'{name} — sample images (first {n_classes} classes)')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "\n"
    "def plot_image_size_histogram(name, entries, raw_root, resolver=None, sample=200):\n"
    "    sizes = []\n"
    "    for e in entries[:sample]:\n"
    "        path = resolver(e) if resolver else (raw_root / e.path)\n"
    "        try:\n"
    "            with Image.open(path) as im:\n"
    "                sizes.append(im.size)\n"
    "        except Exception:\n"
    "            continue\n"
    "    if not sizes:\n"
    "        print(f'{name}: nothing to plot.')\n"
    "        return\n"
    "    widths = [w for w, _ in sizes]\n"
    "    heights = [h for _, h in sizes]\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
    "    axes[0].hist(widths, bins=30); axes[0].set_title(f'{name} widths (n={len(widths)})')\n"
    "    axes[1].hist(heights, bins=30); axes[1].set_title(f'{name} heights')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
)

for spec in [
    "plantvillage",
    "plantdoc",
    "paddy_doctor",
    "phantomfs",
    "olid_i",
]:
    md(f"## {spec}")
    code(
        f"spec = next(s for s in DATASET_SPECS if s.name == '{spec}')\n"
        f"entries_by_split = load_all_entries(spec.name)\n"
        f"counts = class_counts(entries_by_split)\n"
        f"print(spec.name, 'classes=', len(counts), 'total=', sum(counts.values()))\n"
        f"plot_class_distribution(spec.name, counts)"
    )
    code(
        f"spec = next(s for s in DATASET_SPECS if s.name == '{spec}')\n"
        f"entries = load_all_entries(spec.name).get('train', [])\n"
        f"show_sample_images(spec.name, entries, spec.raw_root)"
    )
    code(
        f"spec = next(s for s in DATASET_SPECS if s.name == '{spec}')\n"
        f"entries = load_all_entries(spec.name).get('train', [])\n"
        f"plot_image_size_histogram(spec.name, entries, spec.raw_root)"
    )

md(
    "## Cross-region soil (§14)\n"
    "Side-by-side bar charts of label distributions in Phantom-fs (train+val) "
    "vs IRSID (test). Note the label spaces differ — deposit names vs "
    "texture names — which is the §14 transfer-stress signal."
)
code(
    "cr_train = load_split(SPLITS / 'soil_cross_region' / 'train.json')\n"
    "cr_val = load_split(SPLITS / 'soil_cross_region' / 'val.json')\n"
    "cr_test = load_split(SPLITS / 'soil_cross_region' / 'test.json')\n"
    "phantom_counts = Counter(e.label for e in cr_train + cr_val)\n"
    "irsid_counts = Counter(e.label for e in cr_test)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for ax, (title, counts) in zip(axes, [('Phantom-fs (train+val)', phantom_counts),\n"
    "                                       ('IRSID (test)', irsid_counts)]):\n"
    "    labels = sorted(counts)\n"
    "    ax.bar(range(len(labels)), [counts[l] for l in labels])\n"
    "    ax.set_xticks(range(len(labels)))\n"
    "    ax.set_xticklabels(labels, rotation=45, ha='right')\n"
    "    ax.set_title(title)\n"
    "    ax.set_ylabel('count')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

md(
    "## OLID I labels (smoke sample)\n"
    "Full multi-label co-occurrence heatmap needs the Phase-11 ~14 GB "
    "Zenodo download. For now we show the smoke-sample vocabulary."
)
code(
    "olid_cm_path = SPLITS / 'olid_i' / 'class_map.json'\n"
    "if olid_cm_path.is_file():\n"
    "    olid_cm = load_class_map(olid_cm_path)\n"
    "    for k, v in sorted(olid_cm.items(), key=lambda kv: kv[1]):\n"
    "        print(f'  {v}: {k}')\n"
    "else:\n"
    "    print('No OLID class map yet — run scripts/build_splits.py first.')"
)

NB["cells"] = cells
NB["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
    nbf.write(NB, fh)
print(f"wrote {NOTEBOOK_PATH}")
