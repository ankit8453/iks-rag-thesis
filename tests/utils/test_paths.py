"""Tests for src.utils.paths."""

from __future__ import annotations

from src.utils import paths


def test_project_root_contains_pyproject() -> None:
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()


def test_tracked_directories_exist_after_import() -> None:
    """Importing src.utils.paths triggers ensure_dirs(); they should all exist."""
    for d in (
        paths.CORPUS_RAW_DIR,
        paths.CORPUS_CLEANED_DIR,
        paths.CORPUS_CHUNKS_DIR,
        paths.VECTOR_DB_DIR,
        paths.MODELS_DIR,
        paths.RESULTS_DIR,
        paths.LOGS_DIR,
        paths.FIGURES_DIR,
        paths.CONFIGS_DIR,
    ):
        assert d.is_dir(), f"Expected directory {d} to exist"


def test_paths_are_under_project_root() -> None:
    for d in (
        paths.CORPUS_RAW_DIR,
        paths.MODELS_DIR,
        paths.RESULTS_DIR,
        paths.CONFIGS_DIR,
    ):
        assert paths.PROJECT_ROOT in d.parents, f"{d} is not inside PROJECT_ROOT"
