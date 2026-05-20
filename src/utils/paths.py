"""Project-wide path constants.

Importing this module is the only sanctioned way to refer to top-level
project directories. Hard-coded relative paths scattered across modules
break the moment somebody runs a script from a different working directory.

The constants resolve from this file's location, so they work whether the
project is installed editable, run from a notebook, or imported from a
script in ``scripts/``.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at <PROJECT_ROOT>/src/utils/paths.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Per master reference §41: corpus/ holds ONLY the IKS classical-text
# corpus; image datasets live under data/ and never inside corpus/.
CORPUS_DIR: Path = PROJECT_ROOT / "corpus"
CORPUS_RAW_DIR: Path = CORPUS_DIR / "raw"
CORPUS_CLEANED_DIR: Path = CORPUS_DIR / "cleaned"
CORPUS_CHUNKS_DIR: Path = CORPUS_DIR / "chunks"
VECTOR_DB_DIR: Path = CORPUS_DIR / "vector_db"

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_PLANT_DISEASE_DIR: Path = DATA_DIR / "plant_disease"
DATA_SOIL_DIR: Path = DATA_DIR / "soil"
DATA_SPLITS_DIR: Path = DATA_DIR / "splits"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = RESULTS_DIR / "logs"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NOTES_DIR: Path = PROJECT_ROOT / "notes"
DECISIONS_DIR: Path = PROJECT_ROOT / "decisions"
JOURNAL_DIR: Path = PROJECT_ROOT / "research_journal"

_TRACKED_DIRS: tuple[Path, ...] = (
    CORPUS_RAW_DIR,
    CORPUS_CLEANED_DIR,
    CORPUS_CHUNKS_DIR,
    VECTOR_DB_DIR,
    DATA_DIR,
    DATA_PLANT_DISEASE_DIR,
    DATA_SOIL_DIR,
    DATA_SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    CONFIGS_DIR,
)


def ensure_dirs() -> None:
    """Create any missing project directories.

    Called automatically on import so freshly-cloned checkouts behave the
    same as long-lived ones. The directories themselves are git-ignored
    where appropriate (see ``.gitignore``); ``.gitkeep`` files preserve the
    layout in version control.
    """
    for directory in _TRACKED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


ensure_dirs()


__all__ = [
    "CONFIGS_DIR",
    "CORPUS_CHUNKS_DIR",
    "CORPUS_CLEANED_DIR",
    "CORPUS_DIR",
    "CORPUS_RAW_DIR",
    "DATA_DIR",
    "DATA_PLANT_DISEASE_DIR",
    "DATA_SOIL_DIR",
    "DATA_SPLITS_DIR",
    "DECISIONS_DIR",
    "FIGURES_DIR",
    "JOURNAL_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "NOTES_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "VECTOR_DB_DIR",
    "ensure_dirs",
]
