"""Constants module sanity checks for ``app.config``.

The Phase 10 UI is steered by these constants — they're the seam where
"swap to Phase 5-R model" / "swap to Llama-3.2-3B" happens. This test
asserts the seam still exists and the dropdowns are non-empty.
"""

from __future__ import annotations

import pytest


def test_imports_clean() -> None:
    """``app.config`` must import without touching GPU / network."""
    from app import config  # noqa: F401


def test_disease_switch_documented_and_consistent() -> None:
    """The two-line swap-to-R recipe is the contract; lock it.

    If someone flips ``HAS_NO_LEAF_CLASS`` to True without changing the
    repo, the guardrail will look for a ``no_leaf`` class in the OLD
    27-class model and explode. Catch the inconsistency at import time.
    """
    from app import config

    assert isinstance(config.DISEASE_MODEL_REPO, str)
    assert isinstance(config.HAS_NO_LEAF_CLASS, bool)

    is_r_model = "iks-disease-r-" in config.DISEASE_MODEL_REPO
    assert is_r_model == config.HAS_NO_LEAF_CLASS, (
        f"HAS_NO_LEAF_CLASS={config.HAS_NO_LEAF_CLASS} is inconsistent with "
        f"DISEASE_MODEL_REPO={config.DISEASE_MODEL_REPO!r}. The R model "
        "(repo containing 'iks-disease-r-') ships the no_leaf class; the "
        "OLD model (27-class 'iks-disease-plantdoc') does NOT. Flip both "
        "together or neither."
    )


def test_dropdowns_non_empty() -> None:
    from app import config

    assert len(config.CROP_CHOICES) >= 2
    assert "other" in config.CROP_CHOICES, (
        "Crop list must include 'other' so users can type a custom crop."
    )

    assert len(config.CAUSAL_CHOICES) == 4, (
        "Causal options must cover all 4 CausalPathway enum values."
    )

    from src.integration.causation import CausalPathway

    config_values = {value for value, _ in config.CAUSAL_CHOICES}
    enum_values = {p.value for p in CausalPathway}
    assert config_values == enum_values, (
        f"Causal-pathway dropdown values {config_values} drifted from "
        f"the CausalPathway enum {enum_values}."
    )


def test_guardrail_band_sane() -> None:
    """Fallback guardrail uses [MIN, MAX] as the accept band. Sanity:
    MIN < MAX and both in (0, 1)."""
    from app import config

    assert 0.0 < config.LEAF_FOREGROUND_MIN < config.LEAF_FOREGROUND_MAX < 1.0


def test_default_strategy_is_phase8_winner() -> None:
    """Phase 8 evaluation crowned Strategy B (0.59-0.96) over A (0.01-
    0.04). The default must reflect that until a re-evaluation flips it.
    """
    from app import config

    assert config.DEFAULT_STRATEGY == "B", (
        "Default strategy must be B (Phase 8 winner). If you've re-run "
        "the Phase 8 evaluation and A wins, update this test too."
    )


def test_disclaimer_present_and_substantive() -> None:
    """§39 disclaimer is a hard requirement; lock it."""
    from app import config

    assert len(config.DISCLAIMER) > 80
    assert "professional" in config.DISCLAIMER.lower()
