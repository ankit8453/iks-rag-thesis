"""Multi-task training entrypoint for the soil classifier (Phase 6).

Trains :class:`~src.soil.model.SoilClassifier` with a sum of per-head
cross-entropy losses. Supports both random and region-held-out validation
splits per guardrail #4.
"""

from __future__ import annotations

from pathlib import Path

from src.soil.config import SoilConfig
from src.utils.logging_setup import get_logger
from src.utils.seeding import set_global_seed

_LOGGER = get_logger(__name__)


def train(config: SoilConfig, output_dir: Path) -> dict[str, float]:
    """Train the multi-task soil classifier.

    Parameters
    ----------
    config : SoilConfig
        Validated training config.
    output_dir : Path
        Where checkpoints and metrics are written.

    Returns
    -------
    dict[str, float]
        Per-head macro F1 plus mean macro F1.

    Raises
    ------
    NotImplementedError
        Implementation deferred to Phase 6 — Week 19.
    """
    set_global_seed(config.seed)
    _LOGGER.info("Soil training will write to %s", output_dir)
    raise NotImplementedError("Phase 6 — Week 19: multi-task soil training loop.")
