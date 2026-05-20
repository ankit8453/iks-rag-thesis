"""Training entrypoint for the plant disease classifier (Phase 5).

This module will contain the full training loop: optimizer (AdamW),
learning-rate schedule (cosine with warmup), mixed precision, checkpoint
saving, and TensorBoard logging. Stubbed for now.
"""

from __future__ import annotations

from pathlib import Path

from src.disease.config import DiseaseConfig
from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

_LOGGER = get_logger(__name__)


def train(config: DiseaseConfig, output_dir: Path) -> dict[str, float]:
    """Train the disease classifier end-to-end.

    Parameters
    ----------
    config : DiseaseConfig
        Validated training configuration.
    output_dir : Path
        Where to write checkpoints, TensorBoard logs, and the final
        metrics summary.

    Returns
    -------
    dict[str, float]
        Final-epoch metrics: ``{"accuracy", "macro_f1", "loss"}``.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 5 — Week 16/17.
    """
    set_global_seed(config.seed)
    _LOGGER.info("Disease training will write to %s", output_dir)
    raise NotImplementedError("Phase 5 — Week 16/17: disease training loop.")
