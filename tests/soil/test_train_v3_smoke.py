"""CPU smoke test for the Phase 6 V3 sequential-transfer stage drivers.

Runs one epoch of each of ``train_stage_a`` / ``train_stage_b`` /
``train_stage_c`` against a tiny random multi-task DataLoader and asserts
no exceptions, no NaN, and the expected per-epoch history shape. Uses
``pretrained=False`` so the test is offline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.soil.model import SoilMultiTaskClassifier  # noqa: E402
from src.soil.train_v3 import (  # noqa: E402
    SoilCheckpointManagerV3,
    train_stage_a,
    train_stage_b,
    train_stage_c,
)


class _MultiTaskRandomDataset(Dataset):
    """Random samples cycling which single head each supervises."""

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


def _make_model_opt_scaler():
    model = SoilMultiTaskClassifier(
        backbone_name="efficientnet_b0", pretrained=False, dropout=0.3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def _assert_history_ok(history, expected_stage: str) -> None:
    assert isinstance(history, list)
    assert len(history) == 1
    entry = history[0]
    assert entry["stage"] == expected_stage
    assert entry["epoch"] == 1
    for key in ("loss_soil_type", "loss_moisture", "loss_texture", "loss_total"):
        assert key in entry
        assert entry[key] == entry[key], "NaN loss detected"  # NaN != NaN


def test_stage_a_one_epoch_smoke() -> None:
    model, optimizer, scaler = _make_model_opt_scaler()
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    history = train_stage_a(
        model, loader, optimizer, scaler, device="cpu",
        epochs=1, freeze_epochs=1,  # exercise the frozen-backbone branch
    )
    _assert_history_ok(history, "A")


def test_stage_b_one_epoch_smoke() -> None:
    model, optimizer, scaler = _make_model_opt_scaler()
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    history = train_stage_b(
        model, loader, optimizer, scaler, device="cpu", epochs=1,
    )
    _assert_history_ok(history, "B")


def test_stage_c_one_epoch_smoke() -> None:
    model, optimizer, scaler = _make_model_opt_scaler()
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    history = train_stage_c(
        model, loader, optimizer, scaler, device="cpu", epochs=1,
    )
    _assert_history_ok(history, "C")


def test_stages_chain_and_accumulate_history() -> None:
    """A → B → C should append to a shared history list."""
    model, optimizer, scaler = _make_model_opt_scaler()
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    h = train_stage_a(model, loader, optimizer, scaler, "cpu", epochs=1, freeze_epochs=0)
    h = train_stage_b(model, loader, optimizer, scaler, "cpu", epochs=1, history=h)
    h = train_stage_c(model, loader, optimizer, scaler, "cpu", epochs=1, history=h)
    assert [e["stage"] for e in h] == ["A", "B", "C"]
    assert [e["epoch"] for e in h] == [1, 1, 1]


def test_v3_checkpoint_manager_repo_id_is_v3() -> None:
    mgr = SoilCheckpointManagerV3()
    assert mgr.repo_id == "ankit-iiitdmj/iks-soil-multitask-v3"
