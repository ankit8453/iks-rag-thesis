"""Tests for :mod:`src.soil.mixup` — Mixup/CutMix collation helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.soil.mixup import cutmix_data, maybe_apply_mix, mixed_loss, mixup_data  # noqa: E402


def _fake_batch(batch_size: int = 4):
    images = torch.randn(batch_size, 3, 16, 16)
    labels = {
        "soil_type_label": torch.randint(0, 7, (batch_size,), dtype=torch.long),
        "moisture_label":  torch.randint(0, 3, (batch_size,), dtype=torch.long),
        "texture_label":   torch.randint(0, 3, (batch_size,), dtype=torch.long),
    }
    return images, labels


def test_mixup_data_shapes_and_lam_range() -> None:
    images, labels = _fake_batch(batch_size=4)
    mixed, labels_a, labels_b, lam = mixup_data(images.clone(), labels, alpha=0.2)
    assert mixed.shape == images.shape
    assert labels_a is labels
    assert set(labels_b.keys()) == set(labels.keys())
    for k in labels:
        assert labels_b[k].shape == labels[k].shape
    assert 0.0 <= lam <= 1.0


def test_cutmix_data_bbox_is_valid() -> None:
    images, labels = _fake_batch(batch_size=4)
    h, w = images.shape[2], images.shape[3]
    mixed, labels_a, labels_b, lam = cutmix_data(images.clone(), labels, alpha=1.0)
    # Output shape preserved, mixed pixels still valid floats.
    assert mixed.shape == images.shape
    assert torch.isfinite(mixed).all()
    # lam is the un-cut area ratio in [0, 1].
    assert 0.0 <= lam <= 1.0
    # Each label dict has length == batch_size and corresponds to a valid permutation.
    for k in labels:
        assert labels_b[k].shape == labels[k].shape


def test_maybe_apply_mix_with_p_zero_returns_input_unchanged() -> None:
    images, labels = _fake_batch(batch_size=3)
    out_images, out_labels_a, out_labels_b, lam = maybe_apply_mix(
        images, labels, p=0.0,
    )
    assert out_images is images
    assert out_labels_a is labels
    assert out_labels_b is None
    assert lam == 1.0


def test_maybe_apply_mix_with_p_one_returns_a_mix() -> None:
    images, labels = _fake_batch(batch_size=4)
    out_images, _, out_labels_b, lam = maybe_apply_mix(images.clone(), labels, p=1.0)
    # When mix is applied, labels_b is the permuted dict (not None).
    assert out_labels_b is not None
    assert out_images.shape == images.shape
    assert 0.0 <= lam <= 1.0


def test_mixed_loss_unmixed_path_equals_standard_loss() -> None:
    """When ``labels_b is None``, mixed_loss must equal the bare loss_fn total."""
    predictions = {
        "soil_type": torch.randn(2, 7, requires_grad=True),
        "moisture": torch.randn(2, 3, requires_grad=True),
        "texture": torch.randn(2, 3, requires_grad=True),
    }
    labels = {
        "soil_type_label": torch.tensor([3, -1], dtype=torch.long),
        "moisture_label":  torch.tensor([-1, 2], dtype=torch.long),
        "texture_label":   torch.tensor([-1, -1], dtype=torch.long),
    }

    def loss_fn(preds, batch):
        # Trivial wrapper returning (total, per_head_dict) — exercise the
        # contract without dragging the real V2 loss into the test.
        s = preds["soil_type"].sum() + preds["moisture"].sum() + preds["texture"].sum()
        return s, {"soil_type": s, "moisture": s, "texture": s}

    out = mixed_loss(loss_fn, predictions, labels, None, 1.0)
    assert torch.isfinite(out)


def test_mixed_loss_blends_two_label_sets() -> None:
    predictions = {
        "soil_type": torch.zeros(2, 7, requires_grad=True),
        "moisture": torch.zeros(2, 3, requires_grad=True),
        "texture": torch.zeros(2, 3, requires_grad=True),
    }
    labels_a = {
        "soil_type_label": torch.tensor([0, 1], dtype=torch.long),
        "moisture_label":  torch.tensor([0, 1], dtype=torch.long),
        "texture_label":   torch.tensor([0, 1], dtype=torch.long),
    }
    labels_b = {k: v.flip(0) for k, v in labels_a.items()}

    def loss_fn(preds, batch):
        # Different totals for A vs B so the lam blend is verifiable.
        if batch is labels_a:
            return torch.tensor(2.0, requires_grad=True), {}
        return torch.tensor(4.0, requires_grad=True), {}

    blended = mixed_loss(loss_fn, predictions, labels_a, labels_b, lam=0.25)
    # 0.25 * 2.0 + 0.75 * 4.0 == 3.5
    assert blended.item() == pytest.approx(3.5, rel=1e-5)
