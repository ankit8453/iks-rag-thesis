"""Tests for :func:`src.soil.train.compute_multitask_loss`.

The loss helper must:

- Return a finite scalar tensor + per-head loss dict for a typical batch.
- Be NaN-safe when **every** sample in the batch has ``-1`` for a head's
  label (cross-entropy with ignore_index returns NaN over an empty
  effective batch; we substitute zero).
- When every sample has a valid label for every head, all three heads
  contribute (no zero-out).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.soil.train import TASK_WEIGHTS, compute_multitask_loss  # noqa: E402


def _fake_logits(batch_size: int = 3) -> dict[str, "torch.Tensor"]:
    return {
        "soil_type": torch.randn(batch_size, 7, requires_grad=True),
        "moisture": torch.randn(batch_size, 3, requires_grad=True),
        "texture": torch.randn(batch_size, 3, requires_grad=True),
    }


def test_returns_finite_scalar_plus_per_head_dict() -> None:
    # One supervising-row per head — the spec example: 3 samples, each
    # has a real label for exactly one head, -1 for the others.
    predictions = _fake_logits(batch_size=3)
    batch = {
        "soil_type_label": torch.tensor([4, -1, -1], dtype=torch.long),
        "moisture_label":  torch.tensor([-1, 1, -1], dtype=torch.long),
        "texture_label":   torch.tensor([-1, -1, 2], dtype=torch.long),
    }
    total, per_head = compute_multitask_loss(predictions, batch)
    assert total.shape == ()
    assert torch.isfinite(total)
    assert set(per_head.keys()) == set(TASK_WEIGHTS.keys())
    for v in per_head.values():
        assert isinstance(v, float)


def test_all_minus_one_labels_for_one_head_yields_zero_contrib_no_nan() -> None:
    # Every sample is unsupervised for soil_type. Cross-entropy over zero
    # valid elements returns NaN — the helper must substitute 0.
    predictions = _fake_logits(batch_size=4)
    batch = {
        "soil_type_label": torch.tensor([-1, -1, -1, -1], dtype=torch.long),
        "moisture_label":  torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "texture_label":   torch.tensor([-1, -1, -1, -1], dtype=torch.long),
    }
    total, per_head = compute_multitask_loss(predictions, batch)
    assert torch.isfinite(total)
    # soil_type and texture had no valid samples — their per-head logged
    # loss should be exactly 0.
    assert per_head["soil_type"] == 0.0
    assert per_head["texture"] == 0.0
    # moisture should be > 0 (random logits vs real labels).
    assert per_head["moisture"] > 0.0
    # Backward pass works without NaN poisoning.
    total.backward()


def test_every_head_supervised_for_every_sample_all_three_contribute() -> None:
    predictions = _fake_logits(batch_size=2)
    batch = {
        "soil_type_label": torch.tensor([0, 1], dtype=torch.long),
        "moisture_label":  torch.tensor([1, 2], dtype=torch.long),
        "texture_label":   torch.tensor([2, 0], dtype=torch.long),
    }
    total, per_head = compute_multitask_loss(predictions, batch)
    assert torch.isfinite(total)
    assert per_head["soil_type"] > 0.0
    assert per_head["moisture"] > 0.0
    assert per_head["texture"] > 0.0
    # Total should approximately equal the weighted sum of per-head losses.
    expected = sum(TASK_WEIGHTS[h] * per_head[h] for h in TASK_WEIGHTS)
    assert total.item() == pytest.approx(expected, rel=1e-4)
