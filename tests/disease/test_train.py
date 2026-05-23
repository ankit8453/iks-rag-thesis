"""Tests for src.disease.train (Phase 5 §E).

Real HF Hub is mocked — no network calls during pytest. Real training
on the disease datasets does NOT happen here; that's Colab work. We
exercise:
- auto_batch_size returns sensible numbers across GPU classes
- TrainingMetrics computes per-class precision/recall/F1 correctly on
  a tiny synthetic example
- CheckpointManager.save → load round-trips correctly (mocked HF API)
- One training epoch runs through end-to-end on a tiny synthetic
  dataset (5 samples, 3 classes, 32×32 images, B0 backbone for speed)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ----------------------------------------------------------------------
# auto_batch_size
# ----------------------------------------------------------------------


def test_auto_batch_size_cpu_returns_4() -> None:
    pytest.importorskip("torch")
    from src.disease.train import auto_batch_size

    with patch("torch.cuda.is_available", return_value=False):
        assert auto_batch_size(380) == 4


def test_auto_batch_size_buckets_by_vram(monkeypatch) -> None:
    pytest.importorskip("torch")
    import torch

    from src.disease.train import auto_batch_size

    def _make_props(total_memory_bytes: int):
        return MagicMock(total_memory=total_memory_bytes)

    cases = [
        (40 * 1024**3, 48),   # A100 40GB
        (24 * 1024**3, 32),   # L4 24GB
        (15 * 1024**3, 16),   # T4 15GB
        (6 * 1024**3, 8),     # small unknown GPU
    ]
    for vram_bytes, expected in cases:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties",
            lambda i, _p=vram_bytes: _make_props(_p),
        )
        assert auto_batch_size() == expected, f"VRAM={vram_bytes / 1024**3:.0f} GiB"


# ----------------------------------------------------------------------
# TrainingMetrics
# ----------------------------------------------------------------------


def test_training_metrics_per_class_correctness() -> None:
    torch = pytest.importorskip("torch")
    from src.disease.train import TrainingMetrics

    metrics = TrainingMetrics(num_classes=3)
    # Construct logits whose argmax matches a known prediction pattern:
    # 3 of class 0, 2 of class 1, 1 of class 2 — but mis-predict one
    # of class 1 as class 2.
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],   # pred 0
            [5.0, 0.0, 0.0],   # pred 0
            [5.0, 0.0, 0.0],   # pred 0
            [0.0, 5.0, 0.0],   # pred 1
            [0.0, 0.0, 5.0],   # pred 2  (will be the wrong call)
            [0.0, 0.0, 5.0],   # pred 2
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    metrics.update(logits, labels, loss_value=0.5, batch_size=labels.shape[0])

    # 5 of 6 right -> 83.3% top-1.
    assert abs(metrics.top1_accuracy - 5 / 6) < 1e-9

    per = metrics.per_class()
    by_idx = {int(p["class_idx"]): p for p in per}
    # Class 0: 3 TP, 0 FP, 0 FN -> P=1, R=1, F1=1
    assert by_idx[0]["precision"] == pytest.approx(1.0)
    assert by_idx[0]["recall"] == pytest.approx(1.0)
    # Class 1: 1 TP, 0 FP, 1 FN -> P=1, R=0.5, F1=2/3
    assert by_idx[1]["precision"] == pytest.approx(1.0)
    assert by_idx[1]["recall"] == pytest.approx(0.5)
    # Class 2: 1 TP, 1 FP, 0 FN -> P=0.5, R=1, F1=2/3
    assert by_idx[2]["precision"] == pytest.approx(0.5)
    assert by_idx[2]["recall"] == pytest.approx(1.0)

    # Confusion matrix shape + sums.
    cm = metrics.confusion
    assert len(cm) == 3 and len(cm[0]) == 3
    assert sum(sum(row) for row in cm) == 6


# ----------------------------------------------------------------------
# CheckpointManager — save → load round-trip with mocked HF API
# ----------------------------------------------------------------------


def test_checkpoint_manager_save_load_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from src.disease.train import CheckpointManager

    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api = MagicMock()
        fake_api_cls.return_value = fake_api
        manager = CheckpointManager("ankit-iiitdmj/iks-fixture-model", work_dir=tmp_path)
        manager.ensure_repo(private=True)
        fake_api.create_repo.assert_called_once()

        # Fake model + optimizer + scheduler with state_dicts that
        # serialize trivially.
        class _DummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)
        model = _DummyModule()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        history = [{"epoch": 1, "val_acc": 0.5}]
        path = manager.save_epoch(
            model=model,
            optimizer=opt,
            scheduler=sched,
            epoch=1,
            best_val_acc=0.5,
            history=history,
            is_best=True,
        )
        assert path.is_file()
        # save_epoch wrote 3 things in this case:
        #   checkpoint_latest.pt (always)
        #   checkpoint_best.pt   (is_best=True)
        #   history.json         (always)
        upload_calls = fake_api.upload_file.call_args_list
        repo_paths = [c.kwargs["path_in_repo"] for c in upload_calls]
        assert "checkpoint_latest.pt" in repo_paths
        assert "checkpoint_best.pt" in repo_paths
        assert "history.json" in repo_paths

        # Now mock the download to return the local file, and verify
        # round-trip.
        with patch(
            "huggingface_hub.hf_hub_download", return_value=str(path),
        ):
            loaded = manager.try_load_latest()
        assert loaded is not None
        assert loaded["epoch"] == 1
        assert loaded["best_val_acc"] == 0.5
        assert loaded["history"] == history
        # Model state dict survives serialization.
        loaded_state = loaded["model_state"]
        assert "fc.weight" in loaded_state


# ----------------------------------------------------------------------
# One training epoch on tiny synthetic data (B0 backbone for speed)
# ----------------------------------------------------------------------


def test_train_one_stage_runs_one_epoch_synthetic(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("huggingface_hub")
    from torch.utils.data import DataLoader, TensorDataset

    from src.disease.train import (
        CheckpointManager,
        STAGE_INFO,
        train_one_stage,
    )

    # 6 samples / batch_size=3 → two clean batches with no size-1
    # trailing batch (EfficientNet BatchNorm refuses to compute on 1
    # element during training).
    images = torch.randn(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    ds = TensorDataset(images, labels)
    train_loader = DataLoader(ds, batch_size=3, drop_last=True)
    val_loader = DataLoader(ds, batch_size=3)

    # Stub the stage's num_classes so STAGE_INFO doesn't drive a 38-
    # class head when we want 3.
    monkeypatch.setitem(STAGE_INFO["pretrain"], "num_classes", 3)

    # Build a tiny model via timm B0 (faster than B4 on CPU).
    import timm
    from torch import nn

    class _Wrapper:
        def __init__(self, num_classes: int) -> None:
            backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            head = nn.Sequential(nn.Dropout(0.1), nn.Linear(backbone.num_features, num_classes))
            self._module = nn.Sequential(backbone, head)
            self._backbone = backbone
            self._head = head
            self.num_classes = num_classes

        @property
        def head(self):
            return self._head

        def get_feature_extractor(self):
            return self._backbone

        def freeze_backbone(self) -> int:
            for p in self._backbone.parameters():
                p.requires_grad = False
            return sum(p.numel() for p in self._head.parameters())

        def unfreeze_backbone(self) -> int:
            for p in self._module.parameters():
                p.requires_grad = True
            return sum(p.numel() for p in self._module.parameters())

        def parameters(self):
            return self._module.parameters()

        def state_dict(self):
            return self._module.state_dict()

        def load_state_dict(self, sd, strict=True):
            return self._module.load_state_dict(sd, strict=strict)

        def to(self, *a, **k):
            self._module = self._module.to(*a, **k)
            return self

        def train(self, mode: bool = True):
            self._module.train(mode)
            return self

        def eval(self):
            self._module.eval()
            return self

        def __call__(self, x):
            return self._module(x)

    model = _Wrapper(num_classes=3)

    @dataclass_like_config
    class _Cfg:
        seed = 42
        lr_head = 1e-3
        lr_backbone = 1e-4
        weight_decay = 0.0
        gradient_clip = 1.0
        mixed_precision = False
        freeze_backbone_epochs = 0
        pretrain_epochs = 1

    # Mock HF API completely.
    with patch("huggingface_hub.HfApi") as fake_api_cls:
        fake_api_cls.return_value = MagicMock()
        manager = CheckpointManager(
            "ankit-iiitdmj/iks-fixture-model",
            work_dir=tmp_path / "ckpts",
        )
        result = train_one_stage(
            stage_name="pretrain",
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            config=_Cfg,
            ckpt_manager=manager,
            start_epoch=0,
            total_epochs=1,
            history=[],
            device="cpu",
        )

    assert "history" in result
    assert len(result["history"]) == 1
    h = result["history"][0]
    assert h["stage"] == "pretrain"
    assert h["epoch"] == 1
    assert 0.0 <= h["train_acc"] <= 1.0
    assert 0.0 <= h["val_acc"] <= 1.0


# small helper just so the dataclass-shaped class above is recognised
def dataclass_like_config(cls):  # noqa: D401 — minimal stub
    return cls
