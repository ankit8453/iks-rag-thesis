"""Phase 5 disease-module training loop.

Designed for Google Colab T4 (free tier): mixed precision via
``torch.cuda.amp``, gradient clipping, cosine-annealed LR, early
stopping on val-acc plateau (patience=5). Checkpoints are written
**to the Hugging Face Hub** rather than Google Drive so the run
survives Colab session resets.

Three CLI stages (each independently resumable):

  python -m src.disease.train --stage pretrain          --resume
  python -m src.disease.train --stage finetune_paddy    --resume
  python -m src.disease.train --stage finetune_plantdoc --resume

Stage 2 (Paddy) seeds from the final PlantVillage checkpoint; Stage 3
(PlantDoc) seeds from the final Paddy checkpoint. Handled inside this
script — no manual copy step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

if TYPE_CHECKING:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader  # noqa: F401

_LOGGER = get_logger(__name__)

STAGE_INFO: dict[str, dict[str, Any]] = {
    "pretrain": {
        "dataset_repo": "ankit-iiitdmj/iks-plantvillage",
        "model_repo": "ankit-iiitdmj/iks-disease-plantvillage",
        "num_classes": 38,
        "epochs_field": "pretrain_epochs",
        "start_from_stage": None,
    },
    "finetune_paddy": {
        "dataset_repo": "ankit-iiitdmj/iks-paddy-doctor",
        "model_repo": "ankit-iiitdmj/iks-disease-paddy-doctor",
        "num_classes": 10,
        "epochs_field": "finetune_paddy_epochs",
        "start_from_stage": "pretrain",
    },
    "finetune_plantdoc": {
        "dataset_repo": "ankit-iiitdmj/iks-plantdoc",
        "model_repo": "ankit-iiitdmj/iks-disease-plantdoc",
        "num_classes": 27,
        "epochs_field": "finetune_plantdoc_epochs",
        "start_from_stage": "finetune_paddy",
    },
}


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #


@dataclass
class TrainingMetrics:
    """Per-epoch metrics accumulator.

    Guardrail #3 (PDF §37): keeps the per-class confusion matrix so
    precision/recall/F1 are computable; mean accuracy is reported but
    never in isolation.
    """

    num_classes: int
    confusion: list[list[int]] = field(default_factory=list)
    top1_correct: int = 0
    top5_correct: int = 0
    total: int = 0
    loss_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.confusion:
            self.confusion = [
                [0] * self.num_classes for _ in range(self.num_classes)
            ]

    def update(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        loss_value: float,
        batch_size: int,
    ) -> None:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            k = min(5, self.num_classes)
            top5 = logits.topk(k, dim=1).indices
            for pred, gold in zip(preds.tolist(), labels.tolist(), strict=True):
                if 0 <= gold < self.num_classes and 0 <= pred < self.num_classes:
                    self.confusion[gold][pred] += 1
            self.top1_correct += int((preds == labels).sum().item())
            self.top5_correct += int(
                (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            )
            self.total += int(batch_size)
            self.loss_sum += float(loss_value) * int(batch_size)

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / max(1, self.total)

    @property
    def top5_accuracy(self) -> float:
        return self.top5_correct / max(1, self.total)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(1, self.total)

    def per_class(self) -> list[dict[str, float]]:
        """Return precision/recall/F1/support per class."""
        out: list[dict[str, float]] = []
        for c in range(self.num_classes):
            tp = self.confusion[c][c]
            fp = sum(self.confusion[r][c] for r in range(self.num_classes)) - tp
            fn = sum(self.confusion[c]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            support = sum(self.confusion[c])
            out.append({
                "class_idx": float(c),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(support),
            })
        return out

    def macro_f1(self) -> float:
        per = self.per_class()
        return sum(p["f1"] for p in per) / max(1, len(per))

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "mean_loss": self.mean_loss,
            "macro_f1": self.macro_f1(),
            "per_class": self.per_class(),
            "confusion_matrix": self.confusion,
        }


# --------------------------------------------------------------------- #
# Auto batch size
# --------------------------------------------------------------------- #


def auto_batch_size(image_size: int = 380) -> int:
    """Return a Colab-safe batch size for EfficientNet-B4 + amp.

    Bucketed by GPU VRAM: T4 (15 GB) → 16, L4 (24 GB) → 32, A100
    (40+ GB) → 48. Falls back to 4 on CPU or an unknown small GPU.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return 4

    if not torch.cuda.is_available():
        return 4

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gib = total_bytes / (1024**3)
    if total_gib >= 38:
        return 48
    if total_gib >= 22:
        return 32
    if total_gib >= 14:
        return 16
    return 8


# --------------------------------------------------------------------- #
# Checkpoint manager (HF Hub backed)
# --------------------------------------------------------------------- #


class CheckpointManager:
    """Save/load training checkpoints to/from an HF Hub model repo."""

    def __init__(self, hub_repo_id: str, work_dir: Path | None = None) -> None:
        self.hub_repo_id = hub_repo_id
        self.work_dir = (
            Path(work_dir) if work_dir is not None else Path(".") / "_checkpoints"
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _api(self):
        from huggingface_hub import HfApi  # noqa: PLC0415

        return HfApi()

    def ensure_repo(self, private: bool = True) -> None:
        api = self._api()
        api.create_repo(
            repo_id=self.hub_repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

    def save_epoch(
        self,
        model: Any,
        optimizer: Any | None,
        scheduler: Any,
        epoch: int,
        best_val_acc: float,
        history: list[dict[str, Any]],
        is_best: bool,
    ) -> Path:
        import torch  # noqa: PLC0415

        local_path = self.work_dir / "checkpoint_latest.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "epoch": int(epoch),
                "best_val_acc": float(best_val_acc),
                "history": history,
            },
            local_path,
        )
        self._push_file(local_path, "checkpoint_latest.pt")

        if is_best:
            import shutil  # noqa: PLC0415

            best_path = self.work_dir / "checkpoint_best.pt"
            shutil.copy2(local_path, best_path)
            self._push_file(best_path, "checkpoint_best.pt")

        history_path = self.work_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        self._push_file(history_path, "history.json")
        return local_path

    def _push_file(self, local_path: Path, repo_path: str) -> None:
        api = self._api()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=self.hub_repo_id,
            repo_type="model",
        )
        _LOGGER.info("Pushed %s -> %s/%s", local_path.name, self.hub_repo_id, repo_path)

    def try_load_latest(self) -> dict[str, Any] | None:
        """Pull ``checkpoint_latest.pt`` if present."""
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from huggingface_hub.errors import (  # noqa: PLC0415
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
        import torch  # noqa: PLC0415

        try:
            local_path = hf_hub_download(
                repo_id=self.hub_repo_id,
                filename="checkpoint_latest.pt",
                repo_type="model",
                cache_dir=str(self.work_dir / ".hf_cache"),
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return None
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("HF Hub load failed: %s", exc)
            return None
        return torch.load(local_path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


def evaluate(
    model: Any,
    loader: "DataLoader",
    num_classes: int,
    device: str,
) -> TrainingMetrics:
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    model.eval()
    metrics = TrainingMetrics(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            metrics.update(logits, labels, float(loss.item()), labels.shape[0])
    return metrics


def train_one_stage(
    stage_name: str,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    model: Any,
    config: Any,
    ckpt_manager: CheckpointManager,
    *,
    start_epoch: int = 0,
    total_epochs: int | None = None,
    history: list[dict[str, Any]] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one Phase-5 training stage. Returns ``{history, best_val_acc}``."""
    import torch  # noqa: PLC0415
    from torch import nn, optim  # noqa: PLC0415

    set_global_seed(getattr(config, "seed", 42))
    num_classes = STAGE_INFO[stage_name]["num_classes"]
    epochs = (
        total_epochs
        if total_epochs is not None
        else getattr(config, STAGE_INFO[stage_name]["epochs_field"])
    )

    model.to(device)
    optimizer = optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": config.lr_head},
            {
                "params": model.get_feature_extractor().parameters(),
                "lr": config.lr_backbone,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - start_epoch),
        eta_min=1e-6,
    )
    use_amp = config.mixed_precision and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    history = list(history or [])
    best_val_acc = max((h.get("val_acc", 0.0) for h in history), default=0.0)
    patience = 5
    plateau = 0

    for epoch in range(start_epoch, epochs):
        if epoch < config.freeze_backbone_epochs:
            model.freeze_backbone()
        elif epoch == config.freeze_backbone_epochs:
            model.unfreeze_backbone()

        model.train()
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
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(logits, labels, float(loss.item()), labels.shape[0])

        scheduler.step()
        val_metrics = evaluate(model, val_loader, num_classes, device)
        elapsed = time.monotonic() - epoch_start
        _LOGGER.info(
            "[%s] epoch %d/%d | train acc=%.4f loss=%.4f | val acc=%.4f loss=%.4f | %.1fs",
            stage_name,
            epoch + 1,
            epochs,
            train_metrics.top1_accuracy,
            train_metrics.mean_loss,
            val_metrics.top1_accuracy,
            val_metrics.mean_loss,
            elapsed,
        )

        is_best = val_metrics.top1_accuracy > best_val_acc
        if is_best:
            best_val_acc = val_metrics.top1_accuracy
            plateau = 0
        else:
            plateau += 1

        history.append({
            "stage": stage_name,
            "epoch": epoch + 1,
            "train_acc": train_metrics.top1_accuracy,
            "train_loss": train_metrics.mean_loss,
            "val_acc": val_metrics.top1_accuracy,
            "val_loss": val_metrics.mean_loss,
            "val_macro_f1": val_metrics.macro_f1(),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "lr_backbone": float(optimizer.param_groups[1]["lr"]),
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

        if plateau >= patience:
            _LOGGER.info(
                "[%s] early stopping at epoch %d (plateau=%d)",
                stage_name, epoch + 1, plateau,
            )
            break

    return {"history": history, "best_val_acc": best_val_acc}


# --------------------------------------------------------------------- #
# Loaders from HF Hub
# --------------------------------------------------------------------- #


def _build_loaders_from_hf(
    stage_name: str, batch_size: int, num_workers: int
) -> tuple[Any, Any, int]:
    from datasets import load_dataset  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    info = STAGE_INFO[stage_name]
    _LOGGER.info("Loading HF dataset %s ...", info["dataset_repo"])
    raw = load_dataset(info["dataset_repo"])

    from src.disease.transforms import (  # noqa: PLC0415
        build_disease_eval_aug,
        build_disease_train_aug,
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = build_disease_train_aug(380, mean, std)
    eval_aug = build_disease_eval_aug(380, mean, std)

    class _HFImageDataset(Dataset):
        def __init__(self, hf_split: Any, transform: Any) -> None:
            self.hf_split = hf_split
            self.transform = transform

        def __len__(self) -> int:
            return len(self.hf_split)

        def __getitem__(self, idx: int) -> tuple[Any, int]:
            row = self.hf_split[idx]
            img = np.asarray(row["image"].convert("RGB"))
            tensor = self.transform(image=img)["image"]
            return tensor, int(row["label_idx"])

    train_loader = DataLoader(
        _HFImageDataset(raw["train"], train_aug),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        _HFImageDataset(raw["val"], eval_aug),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, info["num_classes"]


# --------------------------------------------------------------------- #
# CLI / dispatch
# --------------------------------------------------------------------- #


def _build_model_for_stage(
    stage_name: str, num_classes: int, prior_state: dict[str, Any] | None,
) -> Any:
    """Construct DiseaseClassifier; transfer backbone weights from prior stage if any."""
    from src.disease.model import DiseaseClassifier  # noqa: PLC0415

    model = DiseaseClassifier(num_classes=num_classes, pretrained=True, dropout_rate=0.3)
    if prior_state is not None:
        full = prior_state.get("model_state", prior_state)
        backbone_state = {k: v for k, v in full.items() if k.startswith("0.")}
        if backbone_state:
            try:
                model._module.load_state_dict(backbone_state, strict=False)  # type: ignore[arg-type]
                _LOGGER.info("Loaded backbone state from prior stage.")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Could not transfer prior backbone state: %s", exc)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 disease module training.")
    parser.add_argument(
        "--stage", required=True, choices=list(STAGE_INFO.keys()),
        help="Which training stage to run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Pull latest checkpoint from this stage's HF model repo and continue.",
    )
    parser.add_argument(
        "--config", default="configs/disease/default.yaml",
        help="Path to the DiseaseConfig YAML.",
    )
    args = parser.parse_args(argv)

    from src.disease.config import DiseaseConfig  # noqa: PLC0415
    from src.utils.config import load_config  # noqa: PLC0415
    import torch  # noqa: PLC0415

    config = load_config(Path(args.config), DiseaseConfig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = auto_batch_size(image_size=config.image_size)
    _LOGGER.info("Stage=%s device=%s batch_size=%d", args.stage, device, batch_size)

    train_loader, val_loader, num_classes = _build_loaders_from_hf(
        args.stage, batch_size=batch_size, num_workers=config.num_workers,
    )

    stage_info = STAGE_INFO[args.stage]
    ckpt_manager = CheckpointManager(stage_info["model_repo"])
    ckpt_manager.ensure_repo(private=True)

    start_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        ckpt = ckpt_manager.try_load_latest()
        if ckpt is not None:
            start_epoch = int(ckpt["epoch"])
            history = list(ckpt.get("history", []))
            total = getattr(config, stage_info["epochs_field"])
            if start_epoch >= total:
                _LOGGER.info(
                    "Stage %s already complete at epoch %d/%d — exiting cleanly.",
                    args.stage, start_epoch, total,
                )
                return 0
            model = _build_model_for_stage(args.stage, num_classes, ckpt)
            model.load_state_dict(ckpt["model_state"], strict=False)
            _LOGGER.info("Resumed %s from epoch %d", args.stage, start_epoch)
        else:
            prior = stage_info.get("start_from_stage")
            prior_state = None
            if prior is not None:
                prior_state = CheckpointManager(STAGE_INFO[prior]["model_repo"]).try_load_latest()
                if prior_state is None:
                    _LOGGER.warning(
                        "No prior-stage (%s) checkpoint available; starting from ImageNet weights.",
                        prior,
                    )
            model = _build_model_for_stage(args.stage, num_classes, prior_state)
    else:
        model = _build_model_for_stage(args.stage, num_classes, None)

    train_one_stage(
        stage_name=args.stage,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=config,
        ckpt_manager=ckpt_manager,
        start_epoch=start_epoch,
        history=history,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "CheckpointManager",
    "STAGE_INFO",
    "TrainingMetrics",
    "auto_batch_size",
    "evaluate",
    "main",
    "train_one_stage",
]
