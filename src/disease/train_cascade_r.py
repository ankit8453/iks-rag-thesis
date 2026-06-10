"""Phase 5-R cascade trainer — randomized-background variant.

Mirrors :mod:`src.disease.train`'s 3-stage cascade (PlantVillage pretrain
→ Paddy finetune → PlantDoc finetune) but routes data through
:class:`~src.disease.randomized_dataset.RandomizedDiseaseDataset`:

- Stage 1 (PlantVillage) — ``mode="randomize"``. The pretrain stage
  benefits the most from de-correlating background from class because
  PlantVillage is uniformly studio-shot — the shortcut is the
  strongest there.
- Stage 2 (Paddy Doctor) — ``mode="raw"``. Phase 5-R Part 1 verdict:
  paddy is full-canopy, no meaningful foreground/background split, so
  randomization would just paste foliage onto soil and look weird.
  Train conventionally.
- Stage 3 (PlantDoc) — ``mode="randomize"`` for the 27 PlantDoc
  classes, PLUS a 28th ``no_leaf`` reject class drawn from Pandey's
  Background_without_leaves and a bare-soil hold-out from the
  background pool. The final classifier therefore has **28 classes**.

Same architecture / hyperparameters / seed as the original Phase-5
trainer — every difference between this run and the old run can then
be attributed to the background randomization rather than confounded
with a config drift.

Checkpoints land in a NEW namespace ``iks-disease-r-*`` so the old
:mod:`iks-disease-*` models stay intact and revert is trivial.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.disease.backgrounds import build_background_pool
from src.disease.randomized_dataset import (
    Mode,
    SampleSpec,
    build_no_leaf_samples,
    build_samples_from_split,
    make_randomized_dataset,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import (
    DATA_PLANT_DISEASE_DIR,
    PROJECT_ROOT,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

# Stage routing table — same shape as :data:`src.disease.train.STAGE_INFO`
# but adds the ``mode`` field (randomize vs raw vs +no_leaf) and a NEW
# checkpoint namespace so the original Phase-5 models stay untouched.
STAGE_INFO_R: dict[str, dict[str, Any]] = {
    "pretrain_r": {
        "dataset_id": "plantvillage",
        "raw_root": DATA_PLANT_DISEASE_DIR / "plantvillage" / "raw"
                    / "plantvillage dataset" / "color",
        "splits_dir": PROJECT_ROOT / "data" / "splits" / "plantvillage",
        "num_classes": 38,
        "epochs_field": "pretrain_epochs",
        "ckpt_namespace": "iks-disease-r-plantvillage",
        "mode": "randomize",
        "start_from_stage": None,
    },
    "finetune_paddy_r": {
        "dataset_id": "paddy_doctor",
        "raw_root": DATA_PLANT_DISEASE_DIR / "paddy_doctor" / "raw",
        "splits_dir": PROJECT_ROOT / "data" / "splits" / "paddy_doctor",
        "num_classes": 10,
        "epochs_field": "finetune_paddy_epochs",
        "ckpt_namespace": "iks-disease-r-paddy-doctor",
        "mode": "raw",
        "start_from_stage": "pretrain_r",
    },
    "finetune_plantdoc_r": {
        "dataset_id": "plantdoc",
        "raw_root": DATA_PLANT_DISEASE_DIR / "plantdoc" / "raw",
        "splits_dir": PROJECT_ROOT / "data" / "splits" / "plantdoc",
        "num_classes": 28,                                    # 27 + no_leaf
        "epochs_field": "finetune_plantdoc_epochs",
        "ckpt_namespace": "iks-disease-r-plantdoc",
        "mode": "randomize",
        "start_from_stage": "finetune_paddy_r",
        "add_no_leaf": True,
    },
}

# Per-checkpoint root. Trainer writes ``models/disease_r/<namespace>/...``
CHECKPOINT_ROOT_R: Path = PROJECT_ROOT / "models" / "disease_r"


# --------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------- #


@dataclass
class StageResult:
    stage_name: str
    best_val_acc: float
    history_path: Path
    final_ckpt: Path


def _resolve_no_leaf_sources() -> list[Path]:
    """Path to the Pandey Background_without_leaves folder (only piece
    of his dataset we use; the rest is a confirmed PlantVillage re-pack).

    Falls back to an empty list when the folder is not present; the
    randomized dataset then has only the 27 PlantDoc classes."""
    candidates = [
        Path(
            r"C:\Users\HP\Downloads\Plant_leaf_diseases_dataset"
            r"\Plant_leave_diseases_dataset_with_augmentation"
            r"\Background_without_leaves"
        ),
        # Linux/Colab fallback if the folder gets staged under data/.
        PROJECT_ROOT / "data" / "_aux" / "background_without_leaves",
    ]
    return [p for p in candidates if p.is_dir()]


def build_loaders_for_stage(
    stage_name: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
) -> tuple["DataLoader", "DataLoader", "DataLoader", int]:
    """Build train/val/test loaders for one cascade stage.

    Returns ``(train_loader, val_loader, test_loader, num_classes)``.

    For stages with ``add_no_leaf=True`` the train + val splits gain
    no-leaf samples (raw-mode) drawn from Pandey + a hold-out slice of
    the background pool. The TEST split intentionally does NOT get
    no-leaf samples — accuracy is reported per the prompt on the
    "RAW (un-composited) original test splits" so it is directly
    comparable to the original 71 % Phase-5 number.
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.disease.transforms import (  # noqa: PLC0415
        build_disease_eval_aug,
        build_disease_train_aug,
    )

    info = STAGE_INFO_R[stage_name]
    splits_dir: Path = info["splits_dir"]
    raw_root: Path = info["raw_root"]
    dataset_id: str = info["dataset_id"]
    mode: Mode = info["mode"]
    num_classes: int = info["num_classes"]
    add_no_leaf: bool = bool(info.get("add_no_leaf", False))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = build_disease_train_aug(380, mean, std)
    eval_aug = build_disease_eval_aug(380, mean, std)

    # Shared background pool across train/val so the no-leaf hold-out
    # slice can be drawn from a deterministic part of the pool.
    bg_pool = build_background_pool() if mode == "randomize" or add_no_leaf else []

    # ----- no-leaf training samples (PlantDoc stage only) ----- #
    train_extra: list[SampleSpec] = []
    val_extra: list[SampleSpec] = []
    if add_no_leaf:
        sources = _resolve_no_leaf_sources()
        no_leaf_idx = num_classes - 1   # 27 in the 28-class layout
        all_no_leaf = build_no_leaf_samples(sources, label_idx=no_leaf_idx)
        # Conservative 80/20 split for the no_leaf class so val carries some.
        cut = int(0.8 * len(all_no_leaf))
        train_extra = all_no_leaf[:cut]
        val_extra = all_no_leaf[cut:]
        _LOGGER.info(
            "[%s] no_leaf samples: train=%d, val=%d",
            stage_name, len(train_extra), len(val_extra),
        )

    train_ds = make_randomized_dataset(
        split_path=splits_dir / "train.json",
        raw_root=raw_root, dataset_id=dataset_id,
        transform=train_aug, seed=seed, mode=mode,
        extra_samples=train_extra, bg_pool=bg_pool,
    )
    val_ds = make_randomized_dataset(
        split_path=splits_dir / "val.json",
        raw_root=raw_root, dataset_id=dataset_id,
        transform=eval_aug, seed=seed, mode=mode,
        extra_samples=val_extra, bg_pool=bg_pool,
    )
    # The test loader is RAW for every stage so we can compare apples
    # to apples against the original Phase-5 numbers.
    test_samples = build_samples_from_split(
        split_path=splits_dir / "test.json",
        raw_root=raw_root, mode="raw",
    )
    from src.disease.randomized_dataset import RandomizedDiseaseDataset  # noqa: PLC0415

    test_ds = RandomizedDiseaseDataset(
        samples=test_samples, dataset_id=dataset_id, bg_pool=[],
        transform=eval_aug, seed=seed, flagged_rel_paths=set(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, num_classes


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def train_one_stage_r(
    stage_name: str,
    *,
    config: Any,
    device: str = "cuda",
    initial_state_dict: dict[str, Any] | None = None,
) -> StageResult:
    """Train one cascade stage end-to-end. Reuses the Phase-5
    :func:`~src.disease.train.train_one_stage` core so behaviour and
    metrics match.

    Parameters
    ----------
    stage_name
        ``"pretrain_r"`` / ``"finetune_paddy_r"`` / ``"finetune_plantdoc_r"``.
    config
        :class:`~src.disease.config.DiseaseConfig`. Same field names as
        the original trainer reads.
    device
        ``"cuda"`` or ``"cpu"``.
    initial_state_dict
        State dict from the previous stage's best checkpoint, used to
        warm-start the cascade. The head is re-initialised if the
        number of classes changes (PlantVillage 38 → Paddy 10 → PlantDoc
        28).
    """
    import torch  # noqa: PLC0415

    from src.disease.model import DiseaseClassifier  # noqa: PLC0415
    from src.disease.train import (  # noqa: PLC0415
        CheckpointManager,
        train_one_stage,
    )

    info = STAGE_INFO_R[stage_name]
    num_classes = int(info["num_classes"])
    namespace = info["ckpt_namespace"]

    train_loader, val_loader, _test_loader, _ = build_loaders_for_stage(
        stage_name,
        batch_size=int(getattr(config, "batch_size", 16)),
        num_workers=int(getattr(config, "num_workers", 2)),
        seed=int(getattr(config, "seed", 42)),
    )

    model = DiseaseClassifier(
        num_classes=num_classes, pretrained=initial_state_dict is None,
        dropout_rate=getattr(config, "dropout_rate", 0.3),
    )
    if initial_state_dict is not None:
        _LOGGER.info(
            "[%s] warm-starting from prior stage's best checkpoint "
            "(strict=False so a num_classes change does not break the head load).",
            stage_name,
        )
        model.load_state_dict(initial_state_dict, strict=False)

    ckpt_dir = CHECKPOINT_ROOT_R / namespace
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = CheckpointManager(stage_name=stage_name, output_dir=ckpt_dir)

    t0 = time.monotonic()
    result = train_one_stage(
        stage_name=stage_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=config,
        ckpt_manager=ckpt_manager,
        device=device,
    )
    elapsed = time.monotonic() - t0

    history_path = ckpt_dir / "history.json"
    history_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    _LOGGER.info(
        "[%s] stage complete: best_val_acc=%.4f, elapsed=%.0fs, ckpts=%s",
        stage_name, result["best_val_acc"], elapsed, ckpt_dir.relative_to(PROJECT_ROOT),
    )
    final_ckpt = ckpt_dir / "checkpoint_best.pt"
    return StageResult(
        stage_name=stage_name,
        best_val_acc=float(result["best_val_acc"]),
        history_path=history_path,
        final_ckpt=final_ckpt,
    )


def run_cascade(config: Any, device: str = "cuda") -> list[StageResult]:
    """Run all three Phase 5-R stages sequentially.

    Each stage warm-starts from the previous stage's ``checkpoint_best.pt``.
    """
    import torch  # noqa: PLC0415

    results: list[StageResult] = []
    prior_state: dict[str, Any] | None = None
    for stage_name in (
        "pretrain_r", "finetune_paddy_r", "finetune_plantdoc_r",
    ):
        result = train_one_stage_r(
            stage_name=stage_name, config=config, device=device,
            initial_state_dict=prior_state,
        )
        results.append(result)
        # Load the best checkpoint to warm-start the next stage.
        if result.final_ckpt.is_file():
            ckpt = torch.load(result.final_ckpt, map_location="cpu", weights_only=False)
            prior_state = ckpt.get("model_state", ckpt)
    return results


__all__ = [
    "CHECKPOINT_ROOT_R",
    "STAGE_INFO_R",
    "StageResult",
    "build_loaders_for_stage",
    "run_cascade",
    "train_one_stage_r",
]
