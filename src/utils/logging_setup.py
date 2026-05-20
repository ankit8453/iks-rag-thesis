"""Project-wide logging configuration.

Uses the standard library only (no ``structlog`` — see ADR-0002). The
formatter, level, and file destination are configured once on first
:func:`get_logger` call; subsequent calls return loggers from the existing
hierarchy.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_configured: bool = False


def _logs_dir() -> Path:
    """Return the directory log files belong in, creating it if needed.

    Imported lazily so importing :mod:`logging_setup` never triggers an
    import-time circular dependency on :mod:`paths`.
    """
    # Inline import keeps the dependency graph one-way (utils.paths imports
    # nothing; utils.logging_setup may import utils.paths).
    from src.utils.paths import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def _configure_logging() -> None:
    """Attach a stderr handler and a per-day file handler to the root logger.

    Idempotent — calling multiple times is a no-op after the first.
    """
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    root = logging.getLogger()
    root.setLevel(level)

    # stderr handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # File handler — best-effort: if the project root is read-only (e.g.
    # running tests in a sandboxed CI image), skip rather than crash.
    try:
        log_file = _logs_dir() / f"{datetime.now():%Y-%m-%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not attach file log handler; continuing with stderr only.")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name. Use ``__name__`` from the call site so log records
        carry the module path.

    Returns
    -------
    logging.Logger
        The named logger. The root logger is configured once on first call
        with a stderr handler and (if writable) a dated file handler under
        ``results/logs/``.
    """
    _configure_logging()
    return logging.getLogger(name)


__all__ = ["get_logger"]
