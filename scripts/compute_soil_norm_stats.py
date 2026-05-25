"""Compute channel-norm stats over the COMBINED soil train splits.

Loads the ``train`` split of each of the three Phase 6 HF Hub dataset
repos via ``datasets.load_dataset`` (which caches locally on first run)
and computes per-channel mean and std at 224x224 after RGB conversion.
Writes results to ``configs/data/soil_norm.yaml``.

The schema produced by ``scripts/prepare_soil_hf_datasets.py`` stores
images in an HF ``Image()`` column that lazily decodes to PIL on access
— so we stream through the dataset without ever materialising all
bytes in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import CONFIGS_DIR  # noqa: E402

_LOGGER = get_logger(__name__)

IMAGE_SIZE = 224
OUTPUT_PATH = CONFIGS_DIR / "data" / "soil_norm.yaml"

EXPECTED_HF_USERNAME = "ankit-iiitdmj"
HF_REPOS: tuple[str, ...] = (
    f"{EXPECTED_HF_USERNAME}/iks-soil-phantomfs",
    f"{EXPECTED_HF_USERNAME}/iks-soil-sirajganj-moisture",
    f"{EXPECTED_HF_USERNAME}/iks-soil-texture-irsid-vit",
)


def _iter_train_images():
    from datasets import load_dataset  # noqa: PLC0415

    for repo in HF_REPOS:
        _LOGGER.info("Streaming %s/train ...", repo)
        ds = load_dataset(repo, split="train")
        for row in ds:
            yield row["image"]


def _running_mean_std(image_iter) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel mean + std over images resized to ``IMAGE_SIZE``.

    Streams one image at a time so RAM stays bounded for thousand-image
    datasets.
    """
    n = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)

    for img in image_iter:
        rgb = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float64) / 255.0  # H, W, 3
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sum_sq += (flat ** 2).sum(axis=0)
        n += 1
        if n % 500 == 0:
            _LOGGER.info("  ... %d images processed", n)

    if n == 0:
        raise RuntimeError("No images found in train splits.")

    pixels = n * IMAGE_SIZE * IMAGE_SIZE
    mean = channel_sum / pixels
    var = np.maximum(channel_sum_sq / pixels - mean ** 2, 0.0)
    return mean, np.sqrt(var), n


def main() -> int:
    mean, std, n_images = _running_mean_std(_iter_train_images())
    payload = {
        "mean": [round(float(x), 6) for x in mean.tolist()],
        "std": [round(float(x), 6) for x in std.tolist()],
        "n_images": int(n_images),
        "image_size": IMAGE_SIZE,
        "source": "union of train splits of " + ", ".join(HF_REPOS),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Channel norm stats for soil module training input.\n"
        "# Computed from the union of TRAIN splits across the three\n"
        "# Phase 6 HF Hub dataset repos. Image resize: 224x224.\n"
        "# Re-run via ``python scripts/compute_soil_norm_stats.py`` after\n"
        "# the dataset repos are rebuilt.\n"
    )
    OUTPUT_PATH.write_text(
        header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
    )
    print()
    print(f"Soil channel stats over {n_images} images written to {OUTPUT_PATH}:")
    print(f"  mean = {payload['mean']}")
    print(f"  std  = {payload['std']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
