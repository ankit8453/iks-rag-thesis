"""End-to-end CPU smoke test for the Phase 6 training utilities.

Builds a tiny in-memory DataLoader of random tensors that look like
the real multi-task batch (image + 3 per-head labels), runs one step
of :func:`train_one_epoch` and one pass of :func:`evaluate_per_task`,
and asserts no exceptions, no NaN, and the expected dict shapes.

Uses ``pretrained=False`` so the test stays offline. The forward shape
contract is tested separately in :mod:`tests.soil.test_model`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.soil.model import SoilMultiTaskClassifier  # noqa: E402
from src.soil.train import evaluate_per_task, train_one_epoch  # noqa: E402


class _MultiTaskRandomDataset(Dataset):
    """4 random samples, each supervising exactly one head."""

    HEADS = ("soil_type", "moisture", "texture")
    LABEL_COUNTS = {"soil_type": 7, "moisture": 3, "texture": 3}

    def __init__(self, n: int = 4) -> None:
        torch.manual_seed(0)
        self.images = torch.randn(n, 3, 224, 224)
        # Cycle which head each sample supervises so a 4-sample batch
        # covers all three heads at least once.
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


class _SingleTaskRandomDataset(Dataset):
    """A single-head val/test loader — labels are always valid (no -1)."""

    def __init__(self, head: str, n: int = 4) -> None:
        torch.manual_seed(1)
        self.head = head
        self.images = torch.randn(n, 3, 224, 224)
        counts = {"soil_type": 7, "moisture": 3, "texture": 3}[head]
        self.labels = torch.randint(0, counts, (n,), dtype=torch.long)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            f"{self.head}_label": self.labels[idx],
        }


def test_train_one_epoch_and_evaluate_per_task_smoke() -> None:
    model = SoilMultiTaskClassifier(
        backbone_name="efficientnet_b0", pretrained=False, dropout=0.3,
    )
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # CPU path — scaler is unused but train_one_epoch still accepts it.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    losses = train_one_epoch(model, loader, optimizer, scaler, device="cpu")
    assert set(losses.keys()) == {
        "loss_soil_type", "loss_moisture", "loss_texture", "loss_total",
    }
    for v in losses.values():
        assert isinstance(v, float)
        assert v == v, "NaN loss detected"  # NaN != NaN

    val_loaders = {
        head: DataLoader(_SingleTaskRandomDataset(head, n=4), batch_size=2)
        for head in ("soil_type", "moisture", "texture")
    }
    metrics = evaluate_per_task(model, val_loaders, device="cpu")
    assert set(metrics.keys()) == {"soil_type", "moisture", "texture"}
    for head_metrics in metrics.values():
        assert {"top1_accuracy", "macro_f1", "n_samples"} <= set(head_metrics.keys())
        assert head_metrics["n_samples"] == 4
        assert 0.0 <= head_metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= head_metrics["macro_f1"] <= 1.0
