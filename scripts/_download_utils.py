"""Shared helpers for the Phase 4 dataset download scripts.

Per master reference guardrails:
- Idempotent: skip if the target ``raw/`` directory is already non-empty.
- All paths come from :mod:`src.utils.paths`.
- Logging via :func:`src.utils.logging_setup.get_logger` — never ``print``.

Each ``scripts/download_*.py`` script imports the helpers it needs from
this module so the per-dataset scripts stay short.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from src.utils.logging_setup import get_logger

_LOGGER = get_logger(__name__)


def is_directory_populated(directory: Path) -> bool:
    """Return True if ``directory`` exists and contains at least one entry.

    Anything that's not a ``.gitkeep`` counts as content. Used by every
    download script to short-circuit when the dataset is already on disk.
    """
    if not directory.is_dir():
        return False
    for entry in directory.iterdir():
        if entry.name != ".gitkeep":
            return True
    return False


def ensure_raw_dir(directory: Path) -> None:
    """Create ``directory`` (including parents). No-op if it already exists."""
    directory.mkdir(parents=True, exist_ok=True)


def kaggle_competition_download(
    competition: str,
    target_dir: Path,
    *,
    unzip: bool = True,
) -> None:
    """Download a Kaggle *competition* archive via the Kaggle Python API.

    Mirrors :func:`kaggle_download` but uses ``competition_download_files``
    instead. The Kaggle account must have joined the competition on the
    web UI first; otherwise the API returns 403.

    Parameters
    ----------
    competition : str
        Kaggle competition slug, e.g. ``paddy-disease-classification``.
    target_dir : Path
        Where the archive lands. Must exist.
    unzip : bool, default True
        Extract the archive in-place after download.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    _LOGGER.info("Kaggle competition download: %s -> %s", competition, target_dir)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path=str(target_dir), quiet=False)

    if not unzip:
        return

    for archive in sorted(target_dir.glob("*.zip")):
        unzip_into(archive, target_dir)
        try:
            archive.unlink()
        except OSError:
            pass


def kaggle_download(
    slug: str,
    target_dir: Path,
    *,
    unzip: bool = True,
) -> None:
    """Download a Kaggle dataset via the Kaggle Python API.

    Parameters
    ----------
    slug : str
        Kaggle dataset slug, e.g. ``abdallahalidev/plantvillage-dataset``.
    target_dir : Path
        Where the dataset should end up. Must exist (call
        :func:`ensure_raw_dir` first).
    unzip : bool, default True
        Extract the archive in-place. If False, the zip is left next to
        the data.

    Notes
    -----
    Uses the Kaggle Python API (``kaggle.api.dataset_download_files``)
    rather than ``python -m kaggle`` because the ``kaggle`` package does
    not expose a ``__main__`` module on recent versions. The API call
    handles authentication via ``~/.kaggle/kaggle.json``.
    """
    # Lazy import so this module remains import-safe in environments
    # without the kaggle SDK installed.
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    _LOGGER.info("Kaggle download: %s -> %s", slug, target_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(target_dir), unzip=unzip, quiet=False)


def git_clone(repo_url: str, target_dir: Path) -> None:
    """``git clone`` ``repo_url`` into ``target_dir``.

    ``target_dir`` must not exist (git refuses to clone into a populated
    directory). Caller should arrange that, e.g. with
    ``shutil.rmtree(target_dir, ignore_errors=True)``.
    """
    _LOGGER.info("git clone %s -> %s", repo_url, target_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True,
    )


def http_download_with_retry(
    url: str,
    target_path: Path,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
) -> None:
    """Stream ``url`` to ``target_path`` with exponential backoff.

    Raises the last exception if all retries fail. ``requests`` is imported
    lazily so this module stays importable on systems where it isn't yet
    installed (e.g. early CI bootstrap).
    """
    import requests  # noqa: PLC0415 — lazy so module stays import-safe

    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _LOGGER.info("HTTP GET %s (attempt %d/%d)", url, attempt, max_retries)
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with target_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _LOGGER.warning("Download attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                _LOGGER.info("Sleeping %.1fs before retry", sleep_for)
                time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def sha256sum(path: Path) -> str:
    """Return the lower-case hex SHA-256 of ``path`` (streaming)."""
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unzip_into(archive_path: Path, target_dir: Path) -> None:
    """Extract a zip into ``target_dir``."""
    _LOGGER.info("Unzipping %s -> %s", archive_path, target_dir)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(target_dir)


def move_tree(src: Path, dst: Path) -> None:
    """Move a directory tree, equivalent to ``mv src dst``."""
    shutil.move(str(src), str(dst))
