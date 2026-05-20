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
        paths.DATA_DIR,
        paths.DATA_PLANT_DISEASE_DIR,
        paths.DATA_SOIL_DIR,
        paths.DATA_SPLITS_DIR,
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
        paths.DATA_PLANT_DISEASE_DIR,
        paths.DATA_SOIL_DIR,
        paths.DATA_SPLITS_DIR,
        paths.MODELS_DIR,
        paths.RESULTS_DIR,
        paths.CONFIGS_DIR,
    ):
        assert paths.PROJECT_ROOT in d.parents, f"{d} is not inside PROJECT_ROOT"


def test_corpus_dir_contains_no_dataset_paths() -> None:
    """Per master reference §41: corpus/ is IKS text only, never image data."""
    corpus_paths = (
        paths.CORPUS_RAW_DIR,
        paths.CORPUS_CLEANED_DIR,
        paths.CORPUS_CHUNKS_DIR,
        paths.VECTOR_DB_DIR,
    )
    for p in corpus_paths:
        assert paths.DATA_DIR not in p.parents, f"{p} unexpectedly under data/"
        assert "plant_disease" not in p.parts
        assert "soil" not in p.parts


def test_data_subdirs_are_under_data_not_corpus() -> None:
    """The image-dataset constants must point inside data/, not corpus/."""
    for d in (
        paths.DATA_PLANT_DISEASE_DIR,
        paths.DATA_SOIL_DIR,
        paths.DATA_SPLITS_DIR,
    ):
        assert paths.DATA_DIR in d.parents
        assert paths.CORPUS_DIR not in d.parents
