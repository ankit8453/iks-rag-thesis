"""End-to-end CPU smoke test for V2 training utilities.

Builds a tiny in-memory DataLoader of random tensors that look like
the real multi-task batch (image + 3 per-head labels), runs one step
of :func:`train_one_epoch_v2` (which internally applies Mixup/CutMix
some of the time + label-smoothed multi-task loss), and asserts no
exceptions, no NaN, and the expected return-dict shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.soil.model import SoilMultiTaskClassifier  # noqa: E402
from src.soil.train_v2 import (  # noqa: E402
    compute_multitask_loss_smoothed,
    train_one_epoch_v2,
)


class _MultiTaskRandomDataset(Dataset):
    """4 random samples, each supervising exactly one head."""

    HEADS = ("soil_type", "moisture", "texture")
    LABEL_COUNTS = {"soil_type": 7, "moisture": 3, "texture": 3}

    def __init__(self, n: int = 4) -> None:
        torch.manual_seed(0)
        self.images = torch.randn(n, 3, 224, 224)
        self.rows: list[dict] = []
        for i in range(n):
            head = self.HEADS[i % len(self.HEADS)]
            labels = {
                "soil_type_label": -1,
                "moisture_label": -1,
                "texture_label": -1,
            }
            labels[f"{head}_label"] = int(
                torch.randint(0, self.LABEL_COUNTS[head], (1,)).item()
            )
            self.rows.append(labels)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            **{k: torch.tensor(v, dtype=torch.long) for k, v in self.rows[idx].items()},
        }


def test_compute_multitask_loss_smoothed_basic_shape_and_finiteness() -> None:
    predictions = {
        "soil_type": torch.randn(2, 7, requires_grad=True),
        "moisture": torch.randn(2, 3, requires_grad=True),
        "texture": torch.randn(2, 3, requires_grad=True),
    }
    batch = {
        "soil_type_label": torch.tensor([3, -1], dtype=torch.long),
        "moisture_label":  torch.tensor([-1, 1], dtype=torch.long),
        "texture_label":   torch.tensor([-1, -1], dtype=torch.long),
    }
    total, per_head = compute_multitask_loss_smoothed(predictions, batch, smoothing=0.1)
    assert torch.isfinite(total)
    assert set(per_head.keys()) == {"soil_type", "moisture", "texture"}
    for v in per_head.values():
        assert isinstance(v, torch.Tensor)


def test_compute_multitask_loss_smoothed_all_minus_one_for_one_head_is_safe() -> None:
    predictions = {
        "soil_type": torch.randn(3, 7, requires_grad=True),
        "moisture": torch.randn(3, 3, requires_grad=True),
        "texture": torch.randn(3, 3, requires_grad=True),
    }
    batch = {
        "soil_type_label": torch.tensor([-1, -1, -1], dtype=torch.long),
        "moisture_label":  torch.tensor([0, 1, 2], dtype=torch.long),
        "texture_label":   torch.tensor([-1, -1, -1], dtype=torch.long),
    }
    total, per_head = compute_multitask_loss_smoothed(predictions, batch, smoothing=0.1)
    assert torch.isfinite(total)
    # Backward must run without NaN poisoning.
    total.backward()


def test_train_one_epoch_v2_smoke_runs_without_nan_and_returns_loss_dict() -> None:
    torch.manual_seed(42)
    model = SoilMultiTaskClassifier(
        backbone_name="efficientnet_b0", pretrained=False, dropout=0.3,
    )
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    losses = train_one_epoch_v2(
        model, loader, optimizer, scaler, device="cpu",
        mix_p=1.0,  # force a mix on every batch to exercise the Mixup/CutMix path
        label_smoothing=0.1,
    )
    assert set(losses.keys()) == {
        "loss_soil_type", "loss_moisture", "loss_texture", "loss_total",
    }
    for v in losses.values():
        assert isinstance(v, float)
        assert v == v, "NaN loss detected"  # NaN != NaN
