"""Per-dataset RGB channel statistics via Welford's online algorithm.

Computes per-channel mean and standard deviation in a single streaming
pass over the training split — no need to load every pixel into memory
at once. Saved to ``configs/data/<dataset>_norm.yaml`` per master
reference §22 (we use dataset-computed stats, not blanket ImageNet
defaults, because agriculture imagery has different distributions).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from src.utils.data_splits import load_split
from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

_LOGGER = get_logger(__name__)


class ChannelStats(BaseModel):
    """Per-channel mean and std plus provenance metadata.

    Attributes
    ----------
    mean : tuple[float, float, float]
        Per-channel means in [0, 1] (RGB order).
    std : tuple[float, float, float]
        Per-channel population standard deviations.
    n_images_sampled : int
        How many images contributed to the statistic.
    image_size : int
        Square resize size applied before accumulation.
    """

    model_config = ConfigDict(extra="forbid")

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    n_images_sampled: int
    image_size: int


def compute_channel_stats_from_paths(
    image_paths: list[Path],
    image_size: int,
    *,
    max_images: int | None = None,
    seed: int = 42,
) -> ChannelStats:
    """Compute channel stats directly from a list of image paths.

    Equivalent to :func:`compute_channel_stats` but bypasses the split-
    JSON resolution step. Useful for datasets that have no train.json
    (e.g. IRSID is test-only under the §14 cross-region setup).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = list(image_paths)
    if max_images is not None and max_images < len(paths):
        set_global_seed(seed)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx.tolist())]

    total_pixels = 0
    sum_per_channel = np.zeros(3, dtype=np.float64)
    sum_sq_per_channel = np.zeros(3, dtype=np.float64)
    used_images = 0
    for path in paths:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Skipping %s during channel stats: %s", path, exc)
            continue
        flat = arr.reshape(-1, 3).astype(np.float64)
        total_pixels += flat.shape[0]
        sum_per_channel += flat.sum(axis=0)
        sum_sq_per_channel += (flat**2).sum(axis=0)
        used_images += 1

    if total_pixels < 2:
        raise RuntimeError(
            f"Refusing to report channel stats from fewer than 2 pixels "
            f"(got {total_pixels})."
        )
    mean = sum_per_channel / total_pixels
    variance = np.maximum(sum_sq_per_channel / total_pixels - mean**2, 0.0)
    std = variance**0.5
    return ChannelStats(
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        std=(float(std[0]), float(std[1]), float(std[2])),
        n_images_sampled=used_images,
        image_size=image_size,
    )


def compute_channel_stats(
    split_path: Path,
    raw_root: Path,
    image_size: int,
    *,
    max_images: int | None = None,
    seed: int = 42,
) -> ChannelStats:
    """Stream the training split and accumulate per-channel mean/std.

    Parameters
    ----------
    split_path : Path
        Path to a ``train.json`` produced by :func:`save_split`.
    raw_root : Path
        Dataset's raw root; ``SplitEntry.path`` is relative to this.
    image_size : int
        Resize each image to ``(image_size, image_size)`` before
        accumulating.
    max_images : int, optional
        If set, randomly subsample to this many images. Deterministic
        via ``seed``. Useful for very large training sets where the
        marginal information per extra image is small.
    seed : int
        Seed for the subsample shuffle. Default 42.

    Returns
    -------
    ChannelStats
        Mean and std in ``[0, 1]`` (i.e. the inputs you'd hand to
        :class:`albumentations.Normalize` directly).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    entries = load_split(split_path)
    if max_images is not None and max_images < len(entries):
        set_global_seed(seed)
        # NumPy's RNG is deterministic given the same seed.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(entries), size=max_images, replace=False)
        entries = [entries[i] for i in sorted(idx.tolist())]

    # Running sums in float64 (enough precision for billions of pixels).
    total_pixels = 0
    sum_per_channel = np.zeros(3, dtype=np.float64)
    sum_sq_per_channel = np.zeros(3, dtype=np.float64)
    used_images = 0

    for entry in entries:
        path = raw_root / entry.path
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Skipping %s during channel stats: %s", path, exc)
            continue

        # Vectorised batched update — pixels of one image contribute as
        # independent samples to the running mean/variance.
        flat = arr.reshape(-1, 3).astype(np.float64)
        total_pixels += flat.shape[0]
        sum_per_channel += flat.sum(axis=0)
        sum_sq_per_channel += (flat**2).sum(axis=0)
        used_images += 1

    if total_pixels < 2:
        raise RuntimeError(
            f"Refusing to report channel stats from fewer than 2 pixels "
            f"(got {total_pixels}). Check the split JSON at {split_path}."
        )

    mean = sum_per_channel / total_pixels
    variance = sum_sq_per_channel / total_pixels - mean**2
    # Numerical-safety floor (variance can be ~-1e-15 from accumulation).
    variance = np.maximum(variance, 0.0)
    std = variance**0.5
    return ChannelStats(
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        std=(float(std[0]), float(std[1]), float(std[2])),
        n_images_sampled=used_images,
        image_size=image_size,
    )


def save_channel_stats(stats: ChannelStats, output_path: Path) -> None:
    """Write a :class:`ChannelStats` to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(stats.model_dump(), fh, sort_keys=False)
    _LOGGER.info("Saved %s", output_path)


def load_channel_stats(path: Path) -> ChannelStats:
    """Load a YAML produced by :func:`save_channel_stats`."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ChannelStats.model_validate(raw)


__all__ = [
    "ChannelStats",
    "compute_channel_stats",
    "compute_channel_stats_from_paths",
    "load_channel_stats",
    "save_channel_stats",
]
