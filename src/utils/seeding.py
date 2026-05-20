"""Global determinism utilities.

Per master reference §3 (Reproducibility guardrail): any script that
exercises randomness must call :func:`set_global_seed` before doing so.
"""

from __future__ import annotations

import logging
import os
import random

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG the project uses so runs are reproducible.

    Seeds Python's ``random`` module, NumPy, PyTorch (CPU and CUDA), and the
    ``PYTHONHASHSEED`` environment variable. Also forces cuDNN into
    deterministic, non-benchmarking mode. NumPy and PyTorch are imported
    lazily so this module stays importable in environments where they are
    not yet installed (e.g. early CI bootstrap steps).

    Parameters
    ----------
    seed : int, optional
        Seed value applied to every RNG. Defaults to :data:`DEFAULT_SEED`
        (42).

    Notes
    -----
    Determinism is best-effort: a handful of CUDA kernels remain
    non-deterministic regardless of these flags. See the PyTorch docs on
    reproducibility for the full list.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep at runtime
        _LOGGER.warning("NumPy not installed; skipping numpy seeding.")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep at runtime
        _LOGGER.warning("PyTorch not installed; skipping torch seeding.")

    _LOGGER.info("Global seed set to %d", seed)


def assert_seed_set() -> None:
    """Raise if ``PYTHONHASHSEED`` was never set.

    Useful as the first line of a training script to fail fast when somebody
    forgot to call :func:`set_global_seed`.
    """
    if "PYTHONHASHSEED" not in os.environ:
        raise RuntimeError(
            "PYTHONHASHSEED is not set. Call set_global_seed() before "
            "running stochastic code."
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    set_global_seed()
