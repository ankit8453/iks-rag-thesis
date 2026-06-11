"""Regression test for the Phase 5-R stage-info registration.

The Phase 5-R cascade trainer (``train_one_stage_r``) delegates the
actual epoch loop to the original :func:`src.disease.train.train_one_stage`.
That function does ``STAGE_INFO[stage_name]["num_classes"]`` at the top
of its body. ``STAGE_INFO`` only ships with the three OLD stage names
(``pretrain`` / ``finetune_paddy`` / ``finetune_plantdoc``), so unless
the ``_r`` stages are registered into the same dict at import time, the
trainer raises ``KeyError: 'pretrain_r'`` the moment Cell 6 calls it.

This test locks the registration in place. If anyone later refactors
:mod:`src.disease.train_cascade_r` and drops the registration call by
accident, this test fails immediately.
"""

from __future__ import annotations

import pytest


def test_stage_info_r_is_registered_with_original_trainer() -> None:
    """Every ``*_r`` stage must be visible in :data:`src.disease.train.STAGE_INFO`
    with at least ``num_classes`` and ``epochs_field`` populated."""
    # Importing the module triggers the registration side effect.
    from src.disease.train import STAGE_INFO
    from src.disease.train_cascade_r import STAGE_INFO_R

    missing: list[str] = []
    for name, info_r in STAGE_INFO_R.items():
        if name not in STAGE_INFO:
            missing.append(name)
            continue
        registered = STAGE_INFO[name]
        assert "num_classes" in registered, (
            f"_r stage {name!r} registered but missing 'num_classes'"
        )
        assert "epochs_field" in registered, (
            f"_r stage {name!r} registered but missing 'epochs_field'"
        )
        # And the registered values match the _r table — otherwise the
        # trainer would see stale num_classes (e.g. PlantDoc 27 instead
        # of 28 with the no_leaf class).
        assert registered["num_classes"] == info_r["num_classes"]
        assert registered["epochs_field"] == info_r["epochs_field"]

    assert not missing, (
        f"Phase 5-R cascade trainer would crash with KeyError for stages: "
        f"{missing}. The registration call in src/disease/train_cascade_r.py "
        f"must populate src.disease.train.STAGE_INFO with each _r stage."
    )


def test_registration_does_not_clobber_original_stages() -> None:
    """The original (non-``_r``) stages must keep their original
    num_classes — the registration step must NOT widen the original
    PlantDoc 27-class stage to 28."""
    from src.disease.train import STAGE_INFO

    assert STAGE_INFO["pretrain"]["num_classes"] == 38
    assert STAGE_INFO["finetune_paddy"]["num_classes"] == 10
    assert STAGE_INFO["finetune_plantdoc"]["num_classes"] == 27
