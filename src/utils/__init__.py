"""Shared utilities for the IKS Agricultural Advisory System.

This package centralises everything that should look the same across modules:
seeding, paths, logging, and config loading. Importing names from here gives a
stable surface so deeper modules do not reach into private files.

Per master reference §3 (Reproducibility): every script that involves
randomness must call :func:`set_global_seed`.
"""

from src.utils.config import BaseConfig, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import (
    CONFIGS_DIR,
    CORPUS_CHUNKS_DIR,
    CORPUS_CLEANED_DIR,
    CORPUS_RAW_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    VECTOR_DB_DIR,
)
from src.utils.seeding import set_global_seed

__all__ = [
    "BaseConfig",
    "CONFIGS_DIR",
    "CORPUS_CHUNKS_DIR",
    "CORPUS_CLEANED_DIR",
    "CORPUS_RAW_DIR",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "VECTOR_DB_DIR",
    "get_logger",
    "load_config",
    "set_global_seed",
]
