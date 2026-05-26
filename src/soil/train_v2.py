"""Phase 6 V2 training utilities: label smoothing + Mixup/CutMix + TTA eval.

V2 keeps V1's joint multi-task design (3 heads, ``ignore_index=-1``,
per-sample heterogeneous supervision) and adds three things:

1. **Label smoothing**: each head's cross-entropy uses
   ``label_smoothing=0.1``.
2. **Mixup + CutMix collation**: applied to ~30% of training batches
   via :mod:`src.soil.mixup`.
3. **Test-Time Augmentation (TTA)**: 5 deterministic views per test
   image, logits averaged before argmax. See
   :func:`evaluate_per_task_tta`.

V1's :mod:`src.soil.train` is NOT modified — V2 imports ``TASK_WEIGHTS``
and ``SoilCheckpointManager`` from it and adds the new pieces here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.soil.mixup import maybe_apply_mix, mixed_loss
from src.soil.train import TASK_WEIGHTS, SoilCheckpointManager
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

_LOGGER = get_logger(__name__)


# Locked V2 model repo. Distinct from V1's ``ankit-iiitdmj/iks-soil-multitask``
# so the V1 results stay intact for the paper's ablation.
DEFAULT_MODEL_REPO_V2: str = "ankit-iiitdmj/iks-soil-multitask-v2"


# --------------------------------------------------------------------- #
# Label-smoothed multi-task loss
# --------------------------------------------------------------------- #


def compute_multitask_loss_smoothed(
    predictions: dict[str, "torch.Tensor"],
    batch: dict[str, Any],
    smoothing: float = 0.1,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    """Like ``src.soil.train.compute_multitask_loss`` plus ``label_smoothing``.

    Returns ``(total_loss, per_head_loss_tensors)``. ``per_head`` values
    are **tensors** (so :func:`mixed_loss` can blend them), not detached
    floats — the training loop converts to floats only at logging time.

    NaN guard: if a head has zero valid (non-``-1``) samples in the
    batch, ``F.cross_entropy(ignore_index=-1)`` returns NaN. We
    substitute a graph-consistent zero (``logits.sum() * 0.0``) so the
    backward pass produces a zero gradient through that head without
    poisoning the total loss with NaN.
    """
    import torch  # noqa: PLC0415
    from torch.nn import functional as F  # noqa: PLC0415

    head_to_label_key = {
        "soil_type": "soil_type_label",
        "moisture": "moisture_label",
        "texture": "texture_label",
    }

    per_head: dict[str, torch.Tensor] = {}
    head_loss_terms: list[torch.Tensor] = []

    for head, weight in TASK_WEIGHTS.items():
        logits = predictions[head]
        labels = batch[head_to_label_key[head]]
        if labels.dtype != torch.long:
            labels = labels.long()

        head_loss = F.cross_entropy(
            logits, labels,
            ignore_index=-1,
            label_smoothing=smoothing,
            reduction="mean",
        )
        if torch.isnan(head_loss):
            # Graph-consistent zero: keeps the autograd connection to
            # model parameters so backward yields a clean 0 gradient
            # instead of producing a fresh leaf tensor disconnected from
            # the rest of the graph.
            head_loss = logits.sum() * 0.0

        per_head[head] = head_loss
        head_loss_terms.append(weight * head_loss)

    total_loss = torch.stack(head_loss_terms).sum()
    return total_loss, per_head


# --------------------------------------------------------------------- #
# Training loop with Mixup/CutMix
# --------------------------------------------------------------------- #


def train_one_epoch_v2(
    model: Any,
    loader: "DataLoader",
    optimizer: Any,
    scaler: Any,
    device: str,
    mix_p: float = 0.3,
    label_smoothing: float = 0.1,
) -> dict[str, float]:
    """One epoch over the combined multi-task train loader with Mixup/CutMix.

    Returns ``{loss_soil_type, loss_moisture, loss_texture, loss_total}``
    (Python floats, averaged over the epoch's batches). Per-head losses
    are recorded against the **un-mixed** labels of side A so the values
    stay interpretable; the actual backward pass uses
    :func:`src.soil.mixup.mixed_loss` blending sides A and B.
    """
    import torch  # noqa: PLC0415

    model.train()
    use_cuda = device.startswith("cuda")

    n_batches = 0
    sum_per_head = {head: 0.0 for head in TASK_WEIGHTS}
    sum_total = 0.0

    def _loss_fn(preds, labels):
        return compute_multitask_loss_smoothed(preds, labels, smoothing=label_smoothing)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_labels = {
            key: batch[key].to(device, non_blocking=True)
            for key in ("soil_type_label", "moisture_label", "texture_label")
        }

        images, labels_a, labels_b, lam = maybe_apply_mix(
            images, batch_labels, p=mix_p,
        )

        optimizer.zero_grad(set_to_none=True)
        if use_cuda:
            with torch.amp.autocast("cuda"):
                predictions = model(images)
                total_loss = mixed_loss(_loss_fn, predictions, labels_a, labels_b, lam)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            total_loss = mixed_loss(_loss_fn, predictions, labels_a, labels_b, lam)
            total_loss.backward()
            optimizer.step()

        # Per-head logging — recompute on side A (cheap, no autocast).
        with torch.no_grad():
            _, per_head_log = compute_multitask_loss_smoothed(
                predictions, labels_a, smoothing=label_smoothing,
            )
            if labels_b is not None:
                _, per_head_b = compute_multitask_loss_smoothed(
                    predictions, labels_b, smoothing=label_smoothing,
                )
                per_head_log = {
                    k: float(lam * per_head_log[k].detach().item()
                             + (1.0 - lam) * per_head_b[k].detach().item())
                    for k in per_head_log
                }
            else:
                per_head_log = {
                    k: float(v.detach().item()) for k, v in per_head_log.items()
                }

        n_batches += 1
        sum_total += float(total_loss.detach().item())
        for head, val in per_head_log.items():
            sum_per_head[head] = sum_per_head.get(head, 0.0) + float(val)

    n = max(1, n_batches)
    out = {f"loss_{head}": sum_per_head[head] / n for head in TASK_WEIGHTS}
    out["loss_total"] = sum_total / n
    return out


# --------------------------------------------------------------------- #
# Test-Time Augmentation evaluation
# --------------------------------------------------------------------- #


def evaluate_per_task_tta(
    model: Any,
    val_loaders: dict[str, "DataLoader"],
    tta_views: list[Any],
    device: str,
) -> dict[str, dict[str, float]]:
    """Per-task accuracy + macro F1 with logits averaged over TTA views.

    Each ``val_loaders[head]`` is expected to yield batches in the form
    ``(list_of_numpy_HWC_uint8_images, labels_LongTensor)`` — the
    notebook builds these via a custom collate_fn so the TTA Composes
    here can be applied per-image without colliding with the loader's
    own transform.

    For each batch:

    - Apply each of ``len(tta_views)`` Composes to every image to get a
      tensor of shape ``(B*K, C, H, W)`` after stacking.
    - Forward through the model in one pass; pick out this head's
      logits.
    - Reshape to ``(K, B, num_classes)``, average across the K view
      axis, then argmax for the prediction.

    Returns
    -------
    dict
        ``{head: {"top1_accuracy", "top5_accuracy", "macro_f1", "n_samples"}}``.
        ``top5_accuracy`` is set to ``1.0`` for heads with ``<5`` classes
        (trivially perfect), mirroring the V1 notebook's display.
    """
    import torch  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415

    model.eval()
    n_views = len(tta_views)
    results: dict[str, dict[str, float]] = {}
    head_num_classes = {"soil_type": 7, "moisture": 3, "texture": 3}

    for head, loader in val_loaders.items():
        n_classes = head_num_classes[head]
        y_true: list[int] = []
        y_pred: list[int] = []
        top5_correct = 0

        with torch.no_grad():
            for batch_images, batch_labels in loader:
                # batch_images is a list of HWC uint8 numpy arrays.
                # batch_labels is a 1-D LongTensor of length B.
                tensors_per_view: list[torch.Tensor] = []
                for view in tta_views:
                    view_tensors = [
                        view(image=img)["image"] for img in batch_images
                    ]
                    tensors_per_view.append(torch.stack(view_tensors))
                # (K, B, C, H, W) -> (K*B, C, H, W) so we forward in one shot.
                stacked = torch.stack(tensors_per_view, dim=0).to(
                    device, non_blocking=True,
                )
                K, B, C, H, W = stacked.shape
                flat = stacked.view(K * B, C, H, W)
                logits = model(flat)[head]                  # (K*B, n_classes)
                logits = logits.view(K, B, n_classes).mean(dim=0)   # (B, n_classes)

                preds = logits.argmax(dim=1)
                y_true.extend(int(v) for v in batch_labels.tolist())
                y_pred.extend(int(v) for v in preds.tolist())

                if n_classes >= 5:
                    k_top5 = min(5, n_classes)
                    top5 = logits.topk(k_top5, dim=1).indices
                    labels_dev = batch_labels.to(device, non_blocking=True)
                    top5_correct += int(
                        (top5 == labels_dev.unsqueeze(1)).any(dim=1).sum().item()
                    )

        n = len(y_true)
        if n == 0:
            results[head] = {
                "top1_accuracy": 0.0, "top5_accuracy": 1.0 if n_classes < 5 else 0.0,
                "macro_f1": 0.0, "n_samples": 0,
            }
            continue
        top1 = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / n
        if n_classes < 5:
            top5 = 1.0
        else:
            top5 = top5_correct / n
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
        results[head] = {
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "macro_f1": macro,
            "n_samples": n,
        }
    return results


# --------------------------------------------------------------------- #
# V2 checkpoint manager
# --------------------------------------------------------------------- #


class SoilCheckpointManagerV2(SoilCheckpointManager):
    """Same contract as V1's manager, just pinned to the V2 model repo.

    Inheriting keeps the save / save_best / try_load_latest behaviour
    identical so the V2 notebook reads naturally side-by-side with V1.
    """

    def __init__(self, repo_id: str = DEFAULT_MODEL_REPO_V2, *args, **kwargs) -> None:
        super().__init__(repo_id=repo_id, *args, **kwargs)

    def try_load_best(self) -> dict | None:
        """Pull ``checkpoint_best.pt`` from HF Hub. ``None`` on 404."""
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from huggingface_hub.errors import (  # noqa: PLC0415
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
        import torch  # noqa: PLC0415

        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="checkpoint_best.pt",
                repo_type="model",
                cache_dir=str(self.work_dir / ".hf_cache"),
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return None
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("HF Hub load failed: %s", exc)
            return None
        return torch.load(local_path, map_location="cpu", weights_only=False)


__all__ = [
    "DEFAULT_MODEL_REPO_V2",
    "SoilCheckpointManagerV2",
    "compute_multitask_loss_smoothed",
    "evaluate_per_task_tta",
    "train_one_epoch_v2",
]
