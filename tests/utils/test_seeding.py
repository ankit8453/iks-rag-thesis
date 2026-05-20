"""Tests for src.utils.seeding."""

from __future__ import annotations

import os

import pytest

from src.utils.seeding import DEFAULT_SEED, assert_seed_set, set_global_seed


def test_set_global_seed_sets_pythonhashseed() -> None:
    set_global_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_set_global_seed_default_is_42() -> None:
    set_global_seed()
    assert os.environ["PYTHONHASHSEED"] == str(DEFAULT_SEED)


def test_torch_outputs_are_reproducible() -> None:
    """Seeding twice with the same value should give identical torch tensors."""
    torch = pytest.importorskip("torch")

    set_global_seed(42)
    first = torch.randn(5)

    set_global_seed(42)
    second = torch.randn(5)

    assert torch.equal(first, second), "set_global_seed did not produce deterministic output"


def test_numpy_outputs_are_reproducible() -> None:
    np = pytest.importorskip("numpy")

    set_global_seed(7)
    first = np.random.rand(4)
    set_global_seed(7)
    second = np.random.rand(4)

    assert (first == second).all()


def test_assert_seed_set_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError):
        assert_seed_set()


def test_assert_seed_set_passes_after_seeding() -> None:
    set_global_seed(0)
    assert_seed_set()  # should not raise
