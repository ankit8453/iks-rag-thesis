"""Linear-probe fine-tuning (LP-FT) for the PlantDoc stage.

Phase 5 audit (step-wise Grad-CAM diagnosis) established:

- Stage 1 (PlantVillage): 99.8% acc, attends to the LEAF — healthy.
- Stage 2 (Paddy Doctor): 97.0% acc, attends to lesions — healthy.
- Stage 3 (PlantDoc full fine-tune): acc collapses 97% -> 72% AND
  Grad-CAM drifts to background corners.

Root cause: full fine-tuning of the whole B4 on the small, cluttered
PlantDoc set **distorts the good pretrained features** (Kumar et al.,
ICLR 2022 — "Fine-Tuning can Distort Pretrained Features"). The
literature's answer when the pretrained features are good and the
target set is small/shifted is **LP-FT**: freeze the backbone, train
only the classifier head (a linear probe). Frontiers-2026 reports
frozen probes beating fully fine-tuned CNNs by 11-15pp on field
PlantDoc.

This module freezes the **OLD Paddy-stage backbone** (the healthy 97%
one — NOT the failed background-randomization R model) and trains a
fresh 27-class PlantDoc head on top. It pushes to a NEW HF repo so the
original `iks-disease-plantdoc` checkpoint is left untouched for
comparison.

Everything heavy (CheckpointManager, HF loaders, metrics, evaluate) is
re-used from :mod:`src.disease.train` — this module only adds the
freeze-backbone-train-head-only loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)


#: HF repo the LP-FT result is pushed to. Deliberately distinct from the
#: original `iks-disease-plantdoc` so the full-fine-tune baseline stays
#: available for the old-vs-LP-FT comparison.
DEFAULT_LPFT_REPO: str = "ankit-iiitdmj/iks-disease-plantdoc-lpft"

#: Backbone we freeze. The OLD Paddy-stage checkpoint — the healthy 97%
#: model. NOT the R (background-randomized) model, which failed.
DEFAULT_PADDY_BACKBONE_REPO: str = "ankit-iiitdmj/iks-disease-paddy-doctor"

#: PlantDoc stage class count.
PLANTDOC_NUM_CLASSES: int = 27


def build_lpft_model(
    num_classes: int = PLANTDOC_NUM_CLASSES,
    paddy_backbone_repo: str = DEFAULT_PADDY_BACKBONE_REPO,
) -> Any:
    """Build a DiseaseClassifier with the Paddy backbone frozen + fresh head.

    Pulls the OLD Paddy-stage checkpoint, transfers ONLY its backbone
    weights (keys under ``0.``) into a new ``num_classes``-head model,
    then freezes the backbone. The head is fresh random init (the Paddy
    head was 10-class, so it's dropped by the shape-mismatch filter in
    ``_build_model_for_stage``).

    Returns
    -------
    DiseaseClassifier
        Backbone = Paddy weights (frozen). Head = fresh, trainable.
    """
    from src.disease.train import CheckpointManager, _build_model_for_stage

    paddy_state = CheckpointManager(paddy_backbone_repo).try_load_latest()
    if paddy_state is None:
        raise RuntimeError(
            f"No checkpoint found at {paddy_backbone_repo!r}. LP-FT needs the "
            "healthy Paddy-stage backbone to freeze. Check HF auth + repo name."
        )
    # Re-uses the existing prior-stage backbone transfer (keys under "0.").
    model = _build_model_for_stage("finetune_plantdoc", num_classes, paddy_state)
    trainable = model.freeze_backbone()
    _LOGGER.info(
        "LP-FT model ready: Paddy backbone FROZEN, fresh %d-class head "
        "(trainable params = %d, head only).",
        num_classes, trainable,
    )
    return model


def train_lpft(
    model: Any,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    ckpt_manager: Any,
    *,
    num_classes: int = PLANTDOC_NUM_CLASSES,
    epochs: int = 25,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    device: str = "cuda",
    seed: int = 42,
    start_epoch: int = 0,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train ONLY the classifier head on PlantDoc (backbone stays frozen).

    Key LP-FT details vs the full-fine-tune ``train_one_stage``:

    - The optimizer is built over ``model.head.parameters()`` ONLY — the
      backbone never receives an update.
    - The backbone is kept in ``eval()`` mode during training so its
      BatchNorm uses the good running statistics inherited from
      PlantVillage+Paddy, instead of recomputing them on small/cluttered
      PlantDoc batches. (Standard LP-FT practice.)
    - Checkpoints push to ``ckpt_manager``'s repo every epoch (HF Hub
      backed) so a Colab timeout is harmless — same resume contract as
      the rest of Phase 5.

    Returns ``{history, best_val_acc}``.
    """
    import torch
    from torch import nn, optim

    from src.disease.train import TrainingMetrics, evaluate

    set_global_seed(seed)
    model.to(device)
    model.freeze_backbone()  # idempotent — guarantee the invariant

    optimizer = optim.AdamW(
        model.head.parameters(), lr=lr_head, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - start_epoch), eta_min=1e-6,
    )
    use_amp = mixed_precision and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    history = list(history or [])
    best_val_acc = max((h.get("val_acc", 0.0) for h in history), default=0.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        # Frozen backbone in eval mode → use inherited BN running stats.
        model.get_feature_extractor().eval()

        train_metrics = TrainingMetrics(num_classes=num_classes)
        epoch_start = time.monotonic()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(logits, labels, float(loss.item()), labels.shape[0])

        scheduler.step()
        val_metrics = evaluate(model, val_loader, num_classes, device)
        elapsed = time.monotonic() - epoch_start
        _LOGGER.info(
            "[lpft] epoch %d/%d | train acc=%.4f loss=%.4f | val acc=%.4f "
            "macroF1=%.4f | %.1fs",
            epoch + 1, epochs,
            train_metrics.top1_accuracy, train_metrics.mean_loss,
            val_metrics.top1_accuracy, val_metrics.macro_f1(), elapsed,
        )

        is_best = val_metrics.top1_accuracy > best_val_acc
        if is_best:
            best_val_acc = val_metrics.top1_accuracy

        history.append({
            "stage": "finetune_plantdoc_lpft",
            "epoch": epoch + 1,
            "train_acc": train_metrics.top1_accuracy,
            "train_loss": train_metrics.mean_loss,
            "val_acc": val_metrics.top1_accuracy,
            "val_macro_f1": val_metrics.macro_f1(),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
        })
        ckpt_manager.save_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_val_acc=best_val_acc,
            history=history,
            is_best=is_best,
        )

    _LOGGER.info("[lpft] done — best val acc = %.4f", best_val_acc)
    return {"history": history, "best_val_acc": best_val_acc}


__all__ = [
    "DEFAULT_LPFT_REPO",
    "DEFAULT_PADDY_BACKBONE_REPO",
    "PLANTDOC_NUM_CLASSES",
    "build_lpft_model",
    "train_lpft",
]
