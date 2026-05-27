"""Phase 6 V3: sequential transfer-learning experiment (texture boost).

Hypothesis: warming the EfficientNet-B0 backbone on Indian soil
supervision (Phantom-fs soil_type), then on moisture, before activating
the texture head produces a more soil-aware backbone than ImageNet
pretraining alone — expected texture gain +2-5 points.

Three sequential stages (50 epochs total):

- **Stage A** (15 epochs): only ``iks-soil-phantomfs`` is loaded, so only
  the soil_type head sees a non-zero loss (moisture/texture labels are
  ``-1`` for every sample → masked to a graph-consistent zero by V2's
  ``compute_multitask_loss_smoothed``). 5 frozen-backbone warmup epochs,
  then 10 unfrozen.
- **Stage B** (15 epochs): Phantom-fs + Sirajganj. soil_type + moisture
  heads train; texture stays at random init (still all ``-1``). All
  epochs unfrozen (Stage A already warmed the backbone).
- **Stage C** (20 epochs): all three datasets. The texture head finally
  receives gradient on a soil-aware backbone. All epochs unfrozen. These
  checkpoints become the final V3 model.

Everything reuses V1/V2 building blocks unchanged — this module only
composes them into the staged schedule. V1 and V2 source files are NOT
modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# --- V1 building blocks -------------------------------------------------
from src.soil.dataset import build_multitask_labels
from src.soil.model import SoilMultiTaskClassifier  # noqa: F401  (re-exported for the notebook)
from src.soil.train import TASK_WEIGHTS, evaluate_per_task  # noqa: F401
from src.soil.transforms import build_soil_eval_aug  # noqa: F401

# --- V2 building blocks -------------------------------------------------
from src.soil.mixup import (  # noqa: F401
    cutmix_data,
    maybe_apply_mix,
    mixed_loss,
    mixup_data,
)
from src.soil.train_v2 import (
    SoilCheckpointManagerV2,
    compute_multitask_loss_smoothed,  # noqa: F401
    evaluate_per_task_tta,  # noqa: F401
    train_one_epoch_v2,
)
from src.soil.transforms_v2 import build_soil_train_aug_v2, build_tta_views  # noqa: F401
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader

_LOGGER = get_logger(__name__)


DEFAULT_MODEL_REPO_V3: str = "ankit-iiitdmj/iks-soil-multitask-v3"

# Which heads carry valid labels in each stage (the rest stay at -1 and
# are masked to zero loss). Purely informational — the masking itself is
# data-driven via which datasets ``build_stage_loader`` includes.
STAGE_ACTIVE_HEADS: dict[str, tuple[str, ...]] = {
    "A": ("soil_type",),
    "B": ("soil_type", "moisture"),
    "C": ("soil_type", "moisture", "texture"),
}

# Datasets included per stage. Keys are the head names used by
# build_multitask_labels; values are the HF Hub repo "source" head.
STAGE_DATASETS: dict[str, tuple[str, ...]] = {
    "A": ("soil_type",),
    "B": ("soil_type", "moisture"),
    "C": ("soil_type", "moisture", "texture"),
}


# --------------------------------------------------------------------- #
# Checkpoint manager
# --------------------------------------------------------------------- #


class SoilCheckpointManagerV3(SoilCheckpointManagerV2):
    """Same contract as V2's manager, pinned to the V3 model repo.

    Inherits ``save`` / ``save_best`` / ``try_load_latest`` /
    ``try_load_best`` unchanged so the V3 notebook reads naturally
    side-by-side with V2.
    """

    def __init__(self, repo_id: str = DEFAULT_MODEL_REPO_V3, *args, **kwargs) -> None:
        super().__init__(repo_id=repo_id, *args, **kwargs)


# --------------------------------------------------------------------- #
# Per-stage data loader
# --------------------------------------------------------------------- #


class _MultiTaskHFDataset:
    """Wrap one HF dataset split as multi-task rows (one head valid, rest -1).

    Defined here (not in the notebook) so ``build_stage_loader`` is fully
    self-contained. Returns the same dict shape V2's training loop
    expects: ``{image, soil_type_label, moisture_label, texture_label}``.
    """

    def __init__(self, hf_split: Any, head: str, transform: Any) -> None:
        self.hf_split = hf_split
        self.head = head
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        row = self.hf_split[idx]
        arr = np.asarray(row["image"].convert("RGB"))
        tensor = self.transform(image=arr)["image"]
        labels = build_multitask_labels(
            self.head, int(row["label_idx"]),
            head_order=("soil_type", "moisture", "texture"),
        )
        return {
            "image": tensor,
            "soil_type_label": torch.tensor(labels["soil_type"], dtype=torch.long),
            "moisture_label": torch.tensor(labels["moisture"], dtype=torch.long),
            "texture_label": torch.tensor(labels["texture"], dtype=torch.long),
        }


def build_stage_loader(
    stage_name: str,
    hf_datasets: dict[str, Any],
    transform: Any,
    batch_size: int,
    *,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> "DataLoader":
    """Build the combined train loader for a stage.

    Parameters
    ----------
    stage_name
        ``"A"`` / ``"B"`` / ``"C"``.
    hf_datasets
        ``{"soil_type": phantomfs_split, "moisture": sirajganj_split,
        "texture": texture_split}`` — the **train** splits.
    transform
        The (V2 strong) train augmentation pipeline applied to every
        source in the stage.
    batch_size
        DataLoader batch size.

    Returns
    -------
    DataLoader
        A ``ConcatDataset`` over the stage's included sources, so each
        batch naturally mixes the sources present in this stage.
    """
    from torch.utils.data import ConcatDataset, DataLoader  # noqa: PLC0415

    stage = stage_name.upper()
    if stage not in STAGE_DATASETS:
        raise ValueError(f"Unknown stage {stage_name!r}; expected one of A/B/C.")

    heads = STAGE_DATASETS[stage]
    parts = [
        _MultiTaskHFDataset(hf_datasets[head], head, transform) for head in heads
    ]
    combined = ConcatDataset(parts)
    _LOGGER.info(
        "Stage %s loader: %d samples from %s", stage, len(combined), list(heads),
    )
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# --------------------------------------------------------------------- #
# Stage training drivers
# --------------------------------------------------------------------- #


def _run_stage(
    stage_name: str,
    model: Any,
    train_loader: "DataLoader",
    optimizer: Any,
    scaler: Any,
    device: str,
    *,
    epochs: int,
    freeze_epochs: int,
    val_loaders: dict[str, "DataLoader"] | None = None,
    scheduler: Any | None = None,
    ckpt_mgr: Any | None = None,
    mix_p: float = 0.3,
    label_smoothing: float = 0.1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Shared multi-epoch driver for the three stages.

    Returns the per-epoch ``history`` list. Each entry carries the train
    losses, the per-task val metrics (restricted to the stage's active
    heads for display, though ``evaluate_per_task`` always computes all
    three), elapsed seconds, and the stage tag.
    """
    import time  # noqa: PLC0415

    stage = stage_name.upper()
    active = STAGE_ACTIVE_HEADS[stage]
    history = list(history or [])

    for epoch in range(epochs):
        if epoch < freeze_epochs:
            model.freeze_backbone()
            stage_phase = "FROZEN"
        else:
            model.unfreeze_backbone()
            stage_phase = "UNFROZEN"

        t0 = time.time()
        losses = train_one_epoch_v2(
            model, train_loader, optimizer, scaler, device,
            mix_p=mix_p, label_smoothing=label_smoothing,
        )
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0

        val_metrics: dict[str, Any] = {}
        if val_loaders is not None:
            # Only evaluate the heads active in this stage — the others
            # are still at random init and their numbers are meaningless.
            stage_val_loaders = {h: val_loaders[h] for h in active if h in val_loaders}
            val_metrics = evaluate_per_task(model, stage_val_loaders, device)

        val_str = " ".join(
            f"{h}_acc={val_metrics[h]['top1_accuracy']:.4f}"
            for h in active if h in val_metrics
        )
        _LOGGER.info(
            "[V3 stage %s epoch %d/%d] %s | total=%.4f | %s | %.0fs",
            stage, epoch + 1, epochs, stage_phase,
            losses["loss_total"], val_str or "(no val)", elapsed,
        )

        entry = {
            "stage": stage,
            "epoch": epoch + 1,
            "phase": stage_phase,
            **losses,
            "val_metrics": val_metrics,
            "elapsed_seconds": float(elapsed),
        }
        history.append(entry)

        if ckpt_mgr is not None:
            ckpt_mgr.save(
                epoch=len(history) - 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                history=history,
                val_metrics=val_metrics,
            )

    return history


def train_stage_a(
    model: Any,
    train_loader: "DataLoader",
    optimizer: Any,
    scaler: Any,
    device: str,
    epochs: int = 15,
    *,
    val_loaders: dict[str, "DataLoader"] | None = None,
    scheduler: Any | None = None,
    ckpt_mgr: Any | None = None,
    freeze_epochs: int = 5,
    mix_p: float = 0.3,
    label_smoothing: float = 0.1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Stage A — Phantom-fs only. 5 frozen-backbone epochs then 10 unfrozen.

    Only the soil_type head gets gradient (moisture/texture labels are all
    ``-1`` because only Phantom-fs is in ``train_loader``).
    """
    return _run_stage(
        "A", model, train_loader, optimizer, scaler, device,
        epochs=epochs, freeze_epochs=freeze_epochs,
        val_loaders=val_loaders, scheduler=scheduler, ckpt_mgr=ckpt_mgr,
        mix_p=mix_p, label_smoothing=label_smoothing, history=history,
    )


def train_stage_b(
    model: Any,
    train_loader: "DataLoader",
    optimizer: Any,
    scaler: Any,
    device: str,
    epochs: int = 15,
    *,
    val_loaders: dict[str, "DataLoader"] | None = None,
    scheduler: Any | None = None,
    ckpt_mgr: Any | None = None,
    mix_p: float = 0.3,
    label_smoothing: float = 0.1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Stage B — Phantom-fs + Sirajganj. All epochs unfrozen.

    soil_type + moisture heads train; texture stays at random init.
    """
    return _run_stage(
        "B", model, train_loader, optimizer, scaler, device,
        epochs=epochs, freeze_epochs=0,
        val_loaders=val_loaders, scheduler=scheduler, ckpt_mgr=ckpt_mgr,
        mix_p=mix_p, label_smoothing=label_smoothing, history=history,
    )


def train_stage_c(
    model: Any,
    train_loader: "DataLoader",
    optimizer: Any,
    scaler: Any,
    device: str,
    epochs: int = 20,
    *,
    val_loaders: dict[str, "DataLoader"] | None = None,
    scheduler: Any | None = None,
    ckpt_mgr: Any | None = None,
    mix_p: float = 0.3,
    label_smoothing: float = 0.1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Stage C — full multi-task on all 3 datasets. All epochs unfrozen.

    Texture head now trains on the soil-aware backbone. Per-epoch
    checkpoints (via ``ckpt_mgr``) become the final V3 model.
    """
    return _run_stage(
        "C", model, train_loader, optimizer, scaler, device,
        epochs=epochs, freeze_epochs=0,
        val_loaders=val_loaders, scheduler=scheduler, ckpt_mgr=ckpt_mgr,
        mix_p=mix_p, label_smoothing=label_smoothing, history=history,
    )


__all__ = [
    "DEFAULT_MODEL_REPO_V3",
    "STAGE_ACTIVE_HEADS",
    "STAGE_DATASETS",
    "SoilCheckpointManagerV3",
    "build_stage_loader",
    "train_stage_a",
    "train_stage_b",
    "train_stage_c",
]
