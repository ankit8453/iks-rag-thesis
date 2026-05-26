"""Mixup and CutMix batch-level augmentation for Phase 6 V2 training.

V2 applies one of Mixup or CutMix to each minibatch with probability
``p=0.3``. The per-sample labels are dicts of three ``LongTensor``s
(one per head, some entries ``-1`` for the heterogeneous multi-task
supervision); both mix functions reorder/blend the WHOLE label dict
together so the soft-label semantics line up across the three heads.

The loss combiner :func:`mixed_loss` accepts a callable returning
``(total_loss, per_head_dict)`` so it composes with
:func:`src.soil.train_v2.compute_multitask_loss_smoothed` without
either side needing to know about the other.

References
----------
- Zhang et al. 2018, "mixup: Beyond Empirical Risk Minimization"
- Yun et al. 2019, "CutMix: Regularization Strategy to Train Strong
  Classifiers with Localizable Features"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch


def mixup_data(
    images: "torch.Tensor",
    batch_labels: dict[str, "torch.Tensor"],
    alpha: float = 0.2,
) -> tuple["torch.Tensor", dict, dict, float]:
    """Mixup. ``images`` is ``(B, C, H, W)``, ``batch_labels`` is a label dict.

    Returns ``(mixed_images, labels_a, labels_b, lam)``. The caller
    blends per-head losses with :func:`mixed_loss`.
    """
    import torch  # noqa: PLC0415

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index, :]
    labels_a = batch_labels
    labels_b = {k: v[index] for k, v in batch_labels.items()}
    return mixed_images, labels_a, labels_b, lam


def cutmix_data(
    images: "torch.Tensor",
    batch_labels: dict[str, "torch.Tensor"],
    alpha: float = 1.0,
) -> tuple["torch.Tensor", dict, dict, float]:
    """CutMix. Pastes a random rect from a permuted batch over ``images``.

    Returns ``(images, labels_a, labels_b, lam_adjusted)``. The returned
    ``lam_adjusted`` is the true area ratio after rectangle clipping —
    the caller should use this, not the raw Beta-sampled ``lam``, so the
    loss weighting matches the actual pixel mixture.
    """
    import torch  # noqa: PLC0415

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size, _, height, width = images.size()
    index = torch.randperm(batch_size, device=images.device)

    cut_rat = float(np.sqrt(1.0 - lam))
    cut_h = int(height * cut_rat)
    cut_w = int(width * cut_rat)

    cx = int(np.random.randint(width))
    cy = int(np.random.randint(height))
    bbx1 = int(np.clip(cx - cut_w // 2, 0, width))
    bby1 = int(np.clip(cy - cut_h // 2, 0, height))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, width))
    bby2 = int(np.clip(cy + cut_h // 2, 0, height))

    # CutMix paste — in-place modification of ``images`` is intentional;
    # the upstream caller passes a fresh batch every step.
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))

    labels_a = batch_labels
    labels_b = {k: v[index] for k, v in batch_labels.items()}
    return images, labels_a, labels_b, float(lam_adjusted)


def maybe_apply_mix(
    images: "torch.Tensor",
    batch_labels: dict[str, "torch.Tensor"],
    p: float = 0.3,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
) -> tuple["torch.Tensor", dict, dict | None, float]:
    """With probability ``p`` apply Mixup or CutMix (50/50) to the batch.

    Returns ``(images, labels_a, labels_b, lam)``. When no mix is
    applied, ``labels_b`` is ``None`` and ``lam == 1.0`` — the caller
    treats this as a standard single-pass loss.
    """
    if float(np.random.rand()) > p:
        return images, batch_labels, None, 1.0
    if float(np.random.rand()) < 0.5:
        return mixup_data(images, batch_labels, mixup_alpha)
    return cutmix_data(images, batch_labels, cutmix_alpha)


def mixed_loss(
    loss_fn: Callable,
    predictions: dict[str, "torch.Tensor"],
    labels_a: dict[str, "torch.Tensor"],
    labels_b: dict[str, "torch.Tensor"] | None,
    lam: float,
) -> "torch.Tensor":
    """Blend the multi-task loss between ``labels_a`` and ``labels_b``.

    ``loss_fn(predictions, labels)`` must return ``(total_loss,
    per_head_dict)``. We only need the total here; per-head logging
    happens separately in the training loop.

    When ``labels_b`` is ``None`` (no mix applied this step), this
    degrades to the standard single-pass loss.
    """
    total_a, _ = loss_fn(predictions, labels_a)
    if labels_b is None:
        return total_a
    total_b, _ = loss_fn(predictions, labels_b)
    return lam * total_a + (1.0 - lam) * total_b


__all__ = [
    "cutmix_data",
    "maybe_apply_mix",
    "mixed_loss",
    "mixup_data",
]
