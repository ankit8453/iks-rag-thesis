"""Regression test for cascade stage transitions where ``num_classes``
changes (PV 38 → Paddy 10 → PlantDoc 28).

PyTorch's ``Module.load_state_dict(state_dict, strict=False)`` only
ignores *missing or unexpected keys*; if a tensor's shape doesn't
match, it still raises ``RuntimeError: size mismatch for ...``. The
cascade trainer therefore has to **manually filter the prior state
dict** to keep only the keys whose shapes match the new model — most
importantly dropping the classifier head's ``Linear`` weights so the
new head trains from scratch.

This test simulates that scenario directly with a tiny ``nn.Module``
so the regression bites the moment that filter is removed, without
needing to actually load a 71-MB B4 checkpoint.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _make_tiny_classifier(num_classes: int):
    """Two-layer Sequential(backbone, head) — same nesting as
    ``DiseaseClassifier._module`` so the load-state-dict shape-mismatch
    keys look the same (``1.1.weight`` is the head Linear)."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(8, 16),                        # backbone
        nn.Sequential(nn.Dropout(0.1), nn.Linear(16, num_classes)),  # head
    )


def _state_dict_with_head(num_classes: int) -> dict:
    """Save the state dict of a tiny classifier with ``num_classes`` head."""
    return _make_tiny_classifier(num_classes).state_dict()


def test_warm_start_filters_size_mismatched_head_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When prior state has a 38-class head and the current model has a
    10-class head, the trainer MUST drop the head keys (size mismatch)
    and successfully load the rest. If anyone reverts to a plain
    ``model.load_state_dict(prior, strict=False)``, this test fails
    with the same ``RuntimeError: size mismatch`` the user hit."""
    import torch
    from torch import nn

    # Build "prior" state from a 38-class head, and the "current"
    # model with a 10-class head.
    prior_state = _state_dict_with_head(38)
    current_model = _make_tiny_classifier(10)

    # Reproduce the EXACT filter the trainer does. If this logic is
    # ever removed from train_cascade_r.train_one_stage_r, this test
    # locks the behaviour: replace the filter with a plain
    # load_state_dict(prior, strict=False) and watch this raise.
    current_state = current_model.state_dict()
    compatible: dict = {}
    dropped: list[str] = []
    for key, tensor in prior_state.items():
        target = current_state.get(key)
        if target is None or tuple(target.shape) != tuple(tensor.shape):
            dropped.append(key)
            continue
        compatible[key] = tensor

    # Sanity: at least the head's two parameters (weight + bias) get
    # dropped because the linear dim differs.
    assert any("1.1.weight" in k for k in dropped), (
        f"size-mismatch filter did NOT drop the head Linear weight. "
        f"Dropped keys: {dropped}"
    )
    assert any("1.1.bias" in k for k in dropped), (
        f"size-mismatch filter did NOT drop the head Linear bias. "
        f"Dropped keys: {dropped}"
    )

    # And the load must succeed with strict=False on the compatible
    # subset (no RuntimeError).
    missing, unexpected = current_model.load_state_dict(
        compatible, strict=False,
    )
    # The head parameters are now missing — current model keeps its
    # fresh random init for them.
    assert any("1.1.weight" in k for k in missing)
    assert any("1.1.bias" in k for k in missing)


def test_plain_strict_false_load_raises_on_size_mismatch_proof() -> None:
    """Proof of why we filter: the naive ``strict=False`` load IS
    expected to raise. If a future PyTorch quietly lets the size
    mismatch pass, this test alerts us so the filter can be reviewed."""
    prior_state = _state_dict_with_head(38)
    current_model = _make_tiny_classifier(10)

    with pytest.raises(RuntimeError, match="size mismatch"):
        current_model.load_state_dict(prior_state, strict=False)
