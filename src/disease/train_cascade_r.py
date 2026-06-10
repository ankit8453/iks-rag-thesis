"""Phase 5-R cascade trainer — randomized-background variant (HF-first).

Mirrors :mod:`src.disease.train` exactly:

- Images are pulled from HF datasets (``ankit-iiitdmj/iks-plantvillage``
  etc.) — NO local file paths, so a fresh Colab runtime works.
- Checkpoints save AND resume via HF Hub through
  :class:`src.disease.train.CheckpointManager` — every epoch pushes
  ``checkpoint_latest.pt`` (and ``checkpoint_best.pt`` when val acc
  improves) to a new namespace ``ankit-iiitdmj/iks-disease-r-*`` so
  the OLD models stay untouched and revert is trivial.
- Same architecture (EfficientNet-B4 @ 380), same hyperparameters
  (AdamW, lr_head/lr_backbone, weight_decay, gradient_clip), same seed,
  same loss, same scheduler as the original Phase 5 trainer. The ONLY
  difference is the input pipeline: train + val pull through
  :class:`~src.disease.randomized_dataset.HFRandomizedDiseaseDataset`
  in the chosen ``mode``. Test loaders are ALWAYS raw so the top-1
  number is directly comparable to the original 71 %.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.disease.backgrounds import build_background_pool
from src.disease.randomized_dataset import (
    HFRandomizedDiseaseDataset,
    Mode,
    NoLeafRow,
    build_no_leaf_rows,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

StageNameR = Literal["pretrain_r", "finetune_paddy_r", "finetune_plantdoc_r"]

# Per-stage routing. ``dataset_repo`` = HF dataset, ``model_repo`` = HF
# checkpoint namespace (NEW — leaves old ``iks-disease-*`` untouched).
# ``mode`` chooses the randomization treatment for that stage.
STAGE_INFO_R: dict[str, dict[str, Any]] = {
    "pretrain_r": {
        "dataset_repo": "ankit-iiitdmj/iks-plantvillage",
        "dataset_id": "plantvillage",
        "model_repo": "ankit-iiitdmj/iks-disease-r-plantvillage",
        "num_classes": 38,
        "epochs_field": "pretrain_epochs",
        "mode": "randomize",
        "start_from_stage": None,
    },
    "finetune_paddy_r": {
        "dataset_repo": "ankit-iiitdmj/iks-paddy-doctor",
        "dataset_id": "paddy_doctor",
        "model_repo": "ankit-iiitdmj/iks-disease-r-paddy-doctor",
        "num_classes": 10,
        "epochs_field": "finetune_paddy_epochs",
        "mode": "raw",
        "start_from_stage": "pretrain_r",
    },
    "finetune_plantdoc_r": {
        "dataset_repo": "ankit-iiitdmj/iks-plantdoc",
        "dataset_id": "plantdoc",
        "model_repo": "ankit-iiitdmj/iks-disease-r-plantdoc",
        # 27 PlantDoc classes + 1 no_leaf reject = 28.
        "num_classes": 28,
        "epochs_field": "finetune_plantdoc_epochs",
        "mode": "randomize",
        "start_from_stage": "finetune_paddy_r",
        "add_no_leaf": True,
    },
}


@dataclass
class StageResultR:
    stage_name: str
    best_val_acc: float
    history: list[dict[str, Any]]
    model_repo: str


# --------------------------------------------------------------------- #
# Loader factory
# --------------------------------------------------------------------- #


def _build_loaders_r(
    stage_name: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple["DataLoader", "DataLoader", "DataLoader", int]:
    """Build train / val / test loaders for one cascade stage.

    Always returns ``(train, val, test, num_classes)``. The test loader
    is RAW for every stage (no compositing) so the top-1 number is
    comparable to the original Phase 5's evaluation.

    Heavy imports (torch + datasets + albumentations) happen inside the
    function so module-import for tests stays cheap.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.disease.transforms import (  # noqa: PLC0415
        build_disease_eval_aug,
        build_disease_train_aug,
    )

    info = STAGE_INFO_R[stage_name]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = build_disease_train_aug(380, mean, std)
    eval_aug = build_disease_eval_aug(380, mean, std)

    mode: Mode = info["mode"]
    dataset_id = info["dataset_id"]
    dataset_repo = info["dataset_repo"]
    num_classes: int = int(info["num_classes"])
    add_no_leaf: bool = bool(info.get("add_no_leaf", False))

    # Background pool — only built when randomization OR no_leaf is on.
    bg_pool = []
    if mode == "randomize" or add_no_leaf:
        bg_pool = build_background_pool()
        _LOGGER.info("[%s] background pool: %d images", stage_name, len(bg_pool))

    # no_leaf rows (PlantDoc stage only). 80/20 train/val split so val
    # carries the reject class too; test deliberately does NOT include
    # no_leaf so test accuracy stays comparable to the original.
    train_no_leaf: list[NoLeafRow] = []
    val_no_leaf: list[NoLeafRow] = []
    if add_no_leaf:
        no_leaf_idx = num_classes - 1
        # Prefer Pandey (truly "not a leaf"); on Colab fall back to all
        # bg pool sources (still non-leaf urban / soil images).
        sources_present = {e.source for e in bg_pool}
        chosen_sources = (
            ("pandey_background",)
            if "pandey_background" in sources_present
            else tuple(sorted(sources_present))
        )
        all_rows = build_no_leaf_rows(
            bg_pool, label_idx=no_leaf_idx, sources=chosen_sources,
        )
        cut = int(0.8 * len(all_rows))
        train_no_leaf = all_rows[:cut]
        val_no_leaf = all_rows[cut:]
        _LOGGER.info(
            "[%s] no_leaf rows: train=%d val=%d (sources=%s)",
            stage_name, len(train_no_leaf), len(val_no_leaf), chosen_sources,
        )

    from src.disease.randomized_dataset import load_hf_randomized  # noqa: PLC0415

    train_ds = load_hf_randomized(
        dataset_repo=dataset_repo, dataset_id=dataset_id,
        split="train", mode=mode, transform=train_aug,
        bg_pool=bg_pool, no_leaf_rows=train_no_leaf, seed=seed,
    )
    val_ds = load_hf_randomized(
        dataset_repo=dataset_repo, dataset_id=dataset_id,
        split="val", mode=mode, transform=eval_aug,
        bg_pool=bg_pool, no_leaf_rows=val_no_leaf, seed=seed,
    )
    # Test = RAW. No bg pool, no no_leaf rows.
    test_ds = load_hf_randomized(
        dataset_repo=dataset_repo, dataset_id=dataset_id,
        split="test", mode="raw", transform=eval_aug,
        bg_pool=[], no_leaf_rows=[], seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, num_classes


# --------------------------------------------------------------------- #
# Stage runner — reuses the original train_one_stage + CheckpointManager
# --------------------------------------------------------------------- #


def train_one_stage_r(
    stage_name: str,
    *,
    config: Any,
    device: str = "cuda",
    initial_state_dict: dict[str, Any] | None = None,
    resume_from_hub: bool = True,
) -> StageResultR:
    """Train one cascade stage with HF-Hub checkpointing.

    Resume semantics:

    - If a ``checkpoint_latest.pt`` exists in the stage's HF model repo
      AND ``resume_from_hub=True``, the trainer pulls it, loads the
      model + optimizer + history, and resumes from the saved epoch.
      This makes free-Colab session timeouts harmless.
    - Otherwise, the model is initialized from ``initial_state_dict``
      (the previous cascade stage's best) with ``strict=False`` so the
      head re-init is automatic when class counts change.

    Returns
    -------
    StageResultR
        Carries the best val acc and a JSON-serialisable history that
        :func:`run_cascade` reads to warm-start the next stage.
    """
    import torch  # noqa: PLC0415

    from src.disease.model import DiseaseClassifier  # noqa: PLC0415
    from src.disease.train import (  # noqa: PLC0415
        CheckpointManager,
        train_one_stage,
    )

    info = STAGE_INFO_R[stage_name]
    num_classes = int(info["num_classes"])
    model_repo = info["model_repo"]

    train_loader, val_loader, _test, _ = _build_loaders_r(
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
            "[%s] warm-starting from prior stage (strict=False).", stage_name,
        )
        model.load_state_dict(initial_state_dict, strict=False)

    ckpt_manager = CheckpointManager(
        hub_repo_id=model_repo,
        work_dir=PROJECT_ROOT / "_checkpoints" / model_repo.split("/")[-1],
    )
    ckpt_manager.ensure_repo(private=True)

    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_from_hub:
        latest = ckpt_manager.try_load_latest()
        if latest is not None:
            _LOGGER.info(
                "[%s] RESUMING from HF Hub: epoch=%d best_val_acc=%.4f",
                stage_name, latest["epoch"], latest["best_val_acc"],
            )
            model.load_state_dict(latest["model_state"], strict=False)
            start_epoch = int(latest["epoch"])
            history = list(latest.get("history") or [])

    t0 = time.monotonic()
    result = train_one_stage(
        stage_name=stage_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=config,
        ckpt_manager=ckpt_manager,
        start_epoch=start_epoch,
        history=history,
        device=device,
    )
    elapsed = time.monotonic() - t0
    _LOGGER.info(
        "[%s] stage complete: best_val_acc=%.4f elapsed=%.0fs",
        stage_name, result["best_val_acc"], elapsed,
    )
    return StageResultR(
        stage_name=stage_name,
        best_val_acc=float(result["best_val_acc"]),
        history=list(result["history"]),
        model_repo=model_repo,
    )


# --------------------------------------------------------------------- #
# Cascade driver
# --------------------------------------------------------------------- #


def run_cascade(
    config: Any,
    device: str = "cuda",
    *,
    resume_from_hub: bool = True,
) -> list[StageResultR]:
    """Run all three Phase 5-R stages sequentially.

    Each stage warm-starts from the prior stage's best checkpoint on
    HF (pulled via :class:`CheckpointManager`); if a stage was
    interrupted by a Colab session timeout, calling :func:`run_cascade`
    again resumes that stage from ``checkpoint_latest.pt`` on HF and
    rejoins the cascade in place.
    """
    import torch  # noqa: PLC0415

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    results: list[StageResultR] = []
    prior_state: dict[str, Any] | None = None

    stage_order: list[StageNameR] = [
        "pretrain_r", "finetune_paddy_r", "finetune_plantdoc_r",
    ]
    for stage_name in stage_order:
        result = train_one_stage_r(
            stage_name=stage_name, config=config, device=device,
            initial_state_dict=prior_state,
            resume_from_hub=resume_from_hub,
        )
        results.append(result)
        # Pull the *best* checkpoint from this stage's HF repo to warm
        # the next stage. ``checkpoint_best.pt`` was already pushed by
        # the CheckpointManager whenever val acc improved.
        try:
            local = hf_hub_download(
                repo_id=result.model_repo,
                filename="checkpoint_best.pt",
                repo_type="model",
            )
            ckpt = torch.load(local, map_location="cpu", weights_only=False)
            prior_state = ckpt.get("model_state", ckpt)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "[%s] could not pull checkpoint_best.pt from %s: %s — "
                "next stage will train from ImageNet init.",
                stage_name, result.model_repo, exc,
            )
            prior_state = None
    return results


__all__ = [
    "STAGE_INFO_R",
    "StageNameR",
    "StageResultR",
    "run_cascade",
    "train_one_stage_r",
]
