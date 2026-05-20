"""Shared pytest fixtures for the IKS Agricultural Advisory System.

Conventions:
- Heavy fixtures (torch tensors, PIL images) are session-scoped where
  possible to keep the test suite fast.
- Anything that needs determinism uses :func:`src.utils.seeding.set_global_seed`
  rather than mutating RNG state ad-hoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from src.utils.seeding import set_global_seed


@pytest.fixture
def tmp_corpus_dir(tmp_path: Path) -> Path:
    """A throwaway corpus directory with the expected subfolder layout."""
    for sub in ("raw", "cleaned", "chunks", "vector_db"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def seeded_rng() -> Generator[int, None, None]:
    """Seed every RNG with 42 for the duration of one test."""
    set_global_seed(42)
    yield 42


@pytest.fixture
def tiny_dummy_image() -> "object":
    """A 16x16 RGB PIL image — useful for dataset smoke tests.

    Lazily imports PIL so the fixture is only built when requested.
    """
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (16, 16), color=(128, 128, 128))
    return img


@pytest.fixture
def sample_retrieved_chunks() -> list[dict[str, str]]:
    """Three fake retrieved chunks for prompt-template tests."""
    return [
        {"chunk_id": "vrikshayurveda:1", "text": "Neem oil application reduces leaf curl."},
        {"chunk_id": "krishi_parashara:5", "text": "Sandy soils require frequent watering."},
        {"chunk_id": "upavanavinoda:12", "text": "Composted cow dung enriches root growth."},
    ]
