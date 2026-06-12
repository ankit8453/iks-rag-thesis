"""LP-FT trainer invariants — backbone frozen, only head updates.

The whole premise of LP-FT (Phase 5 fix) is that the backbone NEVER
moves — only the classifier head trains. If a refactor ever lets a
backbone gradient through, the "preserve the good PlantVillage/Paddy
features" guarantee silently breaks. This test runs one real training
step on a tiny stand-in model (CPU, no network, no timm) and asserts:

1. backbone weights are byte-identical before/after the step;
2. head weights actually changed.

It stubs the checkpoint manager so nothing touches HF Hub.
"""

from __future__ import annotations

from typing import Any

import pytest


class _TinyModel:
    """Minimal DiseaseClassifier-shaped stand-in for ``train_lpft``.

    Exposes exactly the surface ``train_lpft`` touches: ``__call__``,
    ``to`` / ``train`` / ``eval``, ``parameters``, ``head`` property,
    ``get_feature_extractor()``, and ``freeze_backbone()``.
    """

    def __init__(self) -> None:
        from torch import nn

        self._bb = nn.Linear(4, 6)        # "backbone"
        self._head = nn.Linear(6, 3)      # "head"
        self._module = nn.Sequential(self._bb, self._head)

    def __call__(self, x: Any) -> Any:
        return self._module(x)

    def to(self, *a: Any, **k: Any) -> "_TinyModel":
        self._module.to(*a, **k)
        return self

    def train(self, mode: bool = True) -> "_TinyModel":
        self._module.train(mode)
        return self

    def eval(self) -> "_TinyModel":
        self._module.eval()
        return self

    def parameters(self) -> Any:
        return self._module.parameters()

    @property
    def head(self) -> Any:
        return self._head

    def get_feature_extractor(self) -> Any:
        return self._bb

    def freeze_backbone(self) -> int:
        for p in self._bb.parameters():
            p.requires_grad = False
        for p in self._head.parameters():
            p.requires_grad = True
        return sum(p.numel() for p in self._head.parameters())


class _NoOpCkpt:
    """CheckpointManager stub — save_epoch does nothing (no HF Hub)."""

    def __init__(self) -> None:
        self.calls = 0

    def save_epoch(self, **_kwargs: Any) -> None:
        self.calls += 1


def test_lpft_freezes_backbone_and_trains_head_only() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.disease.train_lpft import train_lpft

    torch.manual_seed(0)
    model = _TinyModel()

    # Snapshot backbone + head weights BEFORE training.
    bb_before = model.get_feature_extractor().weight.detach().clone()
    head_before = model.head.weight.detach().clone()

    # Tiny random dataset: 24 samples, 3 classes, feature dim 4.
    x = torch.randn(24, 4)
    y = torch.randint(0, 3, (24,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    ckpt = _NoOpCkpt()
    out = train_lpft(
        model,
        train_loader=loader,
        val_loader=loader,
        ckpt_manager=ckpt,
        num_classes=3,
        epochs=2,
        lr_head=0.1,           # large LR so the head visibly moves
        mixed_precision=False,  # CPU
        device="cpu",
    )

    bb_after = model.get_feature_extractor().weight.detach()
    head_after = model.head.weight.detach()

    # 1. Backbone must be byte-identical — it was frozen.
    assert torch.equal(bb_before, bb_after), (
        "Backbone weights CHANGED during LP-FT — the freeze leaked. "
        "The whole point of LP-FT is the backbone stays fixed."
    )
    # 2. Head must have moved.
    assert not torch.equal(head_before, head_after), (
        "Head weights did NOT change — LP-FT trained nothing."
    )
    # 3. Bookkeeping sanity.
    assert ckpt.calls == 2, "save_epoch should fire once per epoch."
    assert "best_val_acc" in out and "history" in out
    assert len(out["history"]) == 2


def test_lpft_backbone_param_requires_grad_false() -> None:
    """After build, every backbone param must have requires_grad=False
    and every head param requires_grad=True."""
    model = _TinyModel()
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.get_feature_extractor().parameters())
    assert all(p.requires_grad for p in model.head.parameters())
