"""CPU smoke test for V3-tiling helpers — imports, health_check, train-step.

Verifies the V3-tiling module wires V2's pieces together without breaking
anything: imports clean, ``health_check`` raises for sub-threshold heads
and passes for healthy ones, and one step of ``train_one_epoch_v2`` runs
through the SoilMultiTaskClassifier without NaN.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.soil.model import SoilMultiTaskClassifier  # noqa: E402
from src.soil.train_v2 import train_one_epoch_v2  # noqa: E402
from src.soil.train_v3_tiling import (  # noqa: E402
    HEALTH_THRESHOLD,
    HF_DATASETS_V3_TILING,
    SoilCheckpointManagerV3Tiling,
    health_check,
)


class _MultiTaskRandomDataset(Dataset):
    HEADS = ("soil_type", "moisture", "texture")
    LABEL_COUNTS = {"soil_type": 7, "moisture": 3, "texture": 3}

    def __init__(self, n: int = 4) -> None:
        torch.manual_seed(0)
        self.images = torch.randn(n, 3, 224, 224)
        self.rows: list[dict] = []
        for i in range(n):
            head = self.HEADS[i % len(self.HEADS)]
            row = {
                "soil_type_label": -1,
                "moisture_label": -1,
                "texture_label": -1,
            }
            row[f"{head}_label"] = int(
                torch.randint(0, self.LABEL_COUNTS[head], (1,)).item()
            )
            self.rows.append(row)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            **{k: torch.tensor(v, dtype=torch.long) for k, v in self.rows[idx].items()},
        }


def test_v3_tiling_checkpoint_manager_repo_id() -> None:
    mgr = SoilCheckpointManagerV3Tiling()
    assert mgr.repo_id == "ankit-iiitdmj/iks-soil-multitask-v3-tiling"


def test_hf_datasets_v3_tiling_points_texture_at_tiled_repo() -> None:
    assert HF_DATASETS_V3_TILING["texture"] == "ankit-iiitdmj/iks-soil-texture-tiled"
    # soil_type and moisture must be unchanged from V2 — same dataset repos.
    assert HF_DATASETS_V3_TILING["soil_type"] == "ankit-iiitdmj/iks-soil-phantomfs"
    assert HF_DATASETS_V3_TILING["moisture"] == "ankit-iiitdmj/iks-soil-sirajganj-moisture"


def test_health_check_passes_when_all_heads_above_threshold() -> None:
    val_metrics = {
        "soil_type": {"top1_accuracy": 0.55, "macro_f1": 0.50, "n_samples": 100},
        "moisture": {"top1_accuracy": 0.62, "macro_f1": 0.60, "n_samples": 100},
        "texture": {"top1_accuracy": 0.45, "macro_f1": 0.40, "n_samples": 100},
    }
    health_check(val_metrics)  # must not raise


def test_health_check_raises_when_one_head_below_threshold() -> None:
    val_metrics = {
        "soil_type": {"top1_accuracy": 0.55, "macro_f1": 0.50, "n_samples": 100},
        "moisture": {"top1_accuracy": 0.30, "macro_f1": 0.20, "n_samples": 100},
        "texture": {"top1_accuracy": 0.20, "macro_f1": 0.15, "n_samples": 100},  # below 0.40
    }
    with pytest.raises(RuntimeError, match="health check FAILED"):
        health_check(val_metrics)


def test_health_check_custom_threshold() -> None:
    val_metrics = {
        "soil_type": {"top1_accuracy": 0.55},
        "moisture": {"top1_accuracy": 0.62},
        "texture": {"top1_accuracy": 0.45},
    }
    # Passes at default 0.40; raises at threshold=0.50 because texture=0.45.
    health_check(val_metrics, threshold=0.40)
    with pytest.raises(RuntimeError):
        health_check(val_metrics, threshold=0.50)


def test_health_threshold_constant() -> None:
    assert HEALTH_THRESHOLD == 0.40


def test_v3_tiling_train_one_epoch_smoke() -> None:
    """One step of V2's train_one_epoch_v2 through SoilMultiTaskClassifier."""
    torch.manual_seed(42)
    model = SoilMultiTaskClassifier(
        backbone_name="efficientnet_b0", pretrained=False, dropout=0.3,
    )
    loader = DataLoader(_MultiTaskRandomDataset(n=4), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    losses = train_one_epoch_v2(
        model, loader, optimizer, scaler, device="cpu",
        mix_p=0.5, label_smoothing=0.1,
    )
    assert set(losses.keys()) == {
        "loss_soil_type", "loss_moisture", "loss_texture", "loss_total",
    }
    for v in losses.values():
        assert isinstance(v, float)
        assert v == v, "NaN loss"  # NaN != NaN
