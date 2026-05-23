"""Build OLID I splits + class_map + norm stats from the full Kaggle archive.

Run after ``scripts/download_olid_i.py`` has finished. The OLID I
folder structure (per the upstream Kaggle archive) is
``<crop>__<symptom>/<image>.JPG``. The multi-label vocabulary is the
union of all crop tags and all symptom tags as returned by
:func:`src.integration.causation_dataset._labels_for`.

Outputs:
- ``data/splits/olid_i/train.json``
- ``data/splits/olid_i/val.json``
- ``data/splits/olid_i/test.json``
- ``data/splits/olid_i/class_map.json`` — full multi-label vocabulary
  (``label -> index``).
- ``configs/data/olid_i_norm.yaml`` — per-channel mean/std at 380×380.

Idempotent: re-runs overwrite the prior JSON / YAML.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.integration.causation_dataset import OLID_DEFAULT_ROOT, _labels_for  # noqa: E402
from src.utils.data_splits import SplitEntry  # noqa: E402
from src.utils.data_stats import (  # noqa: E402
    compute_channel_stats_from_paths,
    save_channel_stats,
)
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

_LOGGER = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "olid_i"
NORM_PATH = CONFIGS_DIR / "data" / "olid_i_norm.yaml"
NORM_IMAGE_SIZE = 380

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _discover_items(raw_root: Path) -> list[tuple[Path, str]]:
    """Walk ``raw_root`` for ``<class_folder>/<image>`` pairs.

    Returns ``[(absolute_path, folder_name)]``. Images can be nested
    arbitrarily deep underneath the class folder — we use
    ``rglob('*')`` and take the immediate child of ``raw_root`` as the
    label.
    """
    items: list[tuple[Path, str]] = []
    for top in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = top.name
        for path in sorted(top.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            items.append((path, label))
    return items


def _build_multihot(
    items: list[tuple[Path, str]],
) -> tuple[list[str], np.ndarray]:
    """Build the global label vocabulary and the per-item multi-hot matrix."""
    label_vocab: set[str] = set()
    per_item: list[list[str]] = []
    for _, folder in items:
        labels = _labels_for(folder)
        per_item.append(labels)
        label_vocab.update(labels)
    sorted_vocab = sorted(label_vocab)
    label_to_idx = {lab: i for i, lab in enumerate(sorted_vocab)}

    y = np.zeros((len(items), len(sorted_vocab)), dtype=np.int8)
    for i, labels in enumerate(per_item):
        for lab in labels:
            y[i, label_to_idx[lab]] = 1
    return sorted_vocab, y


def _stratified_multilabel_split(
    items: list[tuple[Path, str]],
    y: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Two-stage stratified split via iterstrat.

    Returns ``(train_idx, val_idx, test_idx)`` — lists of integer
    indices into ``items``.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # noqa: PLC0415

    n = len(items)
    indices = np.arange(n)

    # Stage 1: 10% test.
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=seed
    )
    rest_pos, test_pos = next(msss1.split(np.zeros(n), y))

    # Stage 2: split the remaining 90% into train (~88.89% of rest) +
    # val (~11.11% of rest) so the overall ratios end up 80/10/10.
    val_fraction_of_rest = 0.10 / 0.90
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction_of_rest, random_state=seed
    )
    train_pos_local, val_pos_local = next(
        msss2.split(np.zeros(len(rest_pos)), y[rest_pos])
    )

    train_idx = indices[rest_pos][train_pos_local].tolist()
    val_idx = indices[rest_pos][val_pos_local].tolist()
    test_idx = indices[test_pos].tolist()
    return train_idx, val_idx, test_idx


def _save_split_indices(
    items: list[tuple[Path, str]],
    indices: list[int],
    raw_root: Path,
    output_path: Path,
) -> None:
    """Write a split JSON for the given index subset.

    OLID is multi-label, so the single-int ``label_idx`` field in
    :class:`SplitEntry` cannot honestly represent the 2-3 active tags
    per image. Convention:

    - ``label`` carries the folder name (``<crop>__<symptom>``), which
      the multi-label dataset class expands at load time via
      :func:`src.integration.causation_dataset._labels_for`.
    - ``label_idx`` is set to **0 as a fixed placeholder**. The value
      is meaningless and must NOT be read as a class identity (in
      particular, ``DM`` is index 0 in ``class_map.json`` but that
      does not mean every row is DM).
    - The runtime ``MultiLabelImageDataset.__getitem__`` reads only
      ``entry.label`` and constructs the multi-hot vector against the
      vocabulary; it never touches ``entry.label_idx``.

    See PHASE4_SUMMARY.md → "OLID split JSON convention" for the
    full rationale.
    """
    entries = []
    for i in indices:
        path, folder = items[i]
        rel = path.resolve().relative_to(raw_root.resolve())
        entries.append(
            SplitEntry(path=rel.as_posix(), label=folder, label_idx=0).model_dump()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    _LOGGER.info("Saved %s (%d entries)", output_path, len(entries))


def main() -> int:
    raw_root = OLID_DEFAULT_ROOT
    if not raw_root.is_dir():
        _LOGGER.error("OLID raw root missing at %s — run download_olid_i.py first.", raw_root)
        return 1

    _LOGGER.info("Discovering OLID items under %s ...", raw_root)
    items = _discover_items(raw_root)
    if not items:
        _LOGGER.error("No images found under %s.", raw_root)
        return 1

    label_vocab, y = _build_multihot(items)
    _LOGGER.info(
        "OLID I: %d images, %d folders, %d unique multi-labels.",
        len(items),
        len({lab for _, lab in items}),
        len(label_vocab),
    )

    set_global_seed(42)
    train_idx, val_idx, test_idx = _stratified_multilabel_split(items, y, seed=42)
    _LOGGER.info(
        "Multi-label split: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    _save_split_indices(items, train_idx, raw_root, SPLITS_DIR / "train.json")
    _save_split_indices(items, val_idx, raw_root, SPLITS_DIR / "val.json")
    _save_split_indices(items, test_idx, raw_root, SPLITS_DIR / "test.json")

    class_map = {lab: idx for idx, lab in enumerate(label_vocab)}
    with (SPLITS_DIR / "class_map.json").open("w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)
    _LOGGER.info(
        "Saved class_map.json -> %d multi-label classes (%s)",
        len(class_map),
        SPLITS_DIR / "class_map.json",
    )

    # Channel stats from the training split.
    train_paths = [items[i][0] for i in train_idx]
    stats = compute_channel_stats_from_paths(
        image_paths=train_paths,
        image_size=NORM_IMAGE_SIZE,
        max_images=4000,  # converge well before the full ~3800 train images
        seed=42,
    )
    save_channel_stats(stats, NORM_PATH)
    _LOGGER.info(
        "OLID I norm stats: n=%d, mean=%s, std=%s",
        stats.n_images_sampled,
        stats.mean,
        stats.std,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
