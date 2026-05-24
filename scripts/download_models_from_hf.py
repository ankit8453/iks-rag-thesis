"""Mirror Phase 5 trained disease checkpoints from HF Hub to ``models/disease/``.

Pulls every artifact in the three private model repos under
``ankit-iiitdmj`` to a local ``models/disease/<subdir>/`` so the
trained checkpoints survive a Hub outage and so Phase 8 integration
can load them via plain ``torch.load`` paths.

- Pre-flight: refuses to proceed unless the HF token belongs to
  ``ankit-iiitdmj``.
- Per .pt file: SHA-256 the downloaded copy and compare against the
  Hub's published LFS sha256. Logs a clear `MISMATCH` if they differ.
- Idempotent: if a local copy already exists AND its SHA-256 matches
  the Hub, the file is skipped (re-runs are cheap).
- Soft-fails on missing optional files (e.g. an older repo without
  ``eval_metrics_test.json``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._download_utils import sha256sum  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from src.utils.paths import MODELS_DIR  # noqa: E402

if TYPE_CHECKING:
    from huggingface_hub import HfApi  # noqa: F401

_LOGGER = get_logger(__name__)

EXPECTED_HF_USERNAME = "ankit-iiitdmj"

MODEL_REPOS: list[tuple[str, str]] = [
    ("ankit-iiitdmj/iks-disease-plantvillage", "plantvillage"),
    ("ankit-iiitdmj/iks-disease-paddy-doctor", "paddy-doctor"),
    ("ankit-iiitdmj/iks-disease-plantdoc", "plantdoc"),
]

FILES_TO_PULL: list[str] = [
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "history.json",
    "eval_metrics.json",
    "eval_metrics_test.json",
]

DISEASE_MODELS_ROOT: Path = MODELS_DIR / "disease"


@dataclass
class _PullResult:
    repo: str
    filename: str
    local_path: Path | None
    size_bytes: int
    sha256_match: bool | None      # None for non-.pt files
    status: str                    # "downloaded" / "skipped" / "missing" / "mismatch"


def _hub_sha256_lookup(api: "HfApi", repo_id: str) -> dict[str, str]:
    """Return a ``{filename: lfs_sha256}`` map for one repo's siblings."""
    info = api.model_info(repo_id, files_metadata=True)
    out: dict[str, str] = {}
    for sib in info.siblings or []:
        if sib.lfs is not None:
            # LFS-tracked file — the sha256 lives in the LFS pointer info.
            sha = sib.lfs.get("sha256") if isinstance(sib.lfs, dict) else getattr(sib.lfs, "sha256", None)
            if sha:
                out[sib.rfilename] = sha
    return out


def _verify_sha256(local_path: Path, expected_sha: str | None, repo_id: str) -> bool | None:
    """Return True/False/None depending on whether SHA matched.

    None means the Hub didn't expose an LFS sha for this file (i.e. it
    was a tiny JSON not tracked by LFS).
    """
    if expected_sha is None:
        return None
    actual = sha256sum(local_path)
    matches = actual.lower() == expected_sha.lower()
    if matches:
        _LOGGER.info("SHA-256 ok: %s/%s", repo_id, local_path.name)
    else:
        _LOGGER.error(
            "SHA-256 MISMATCH for %s/%s — local=%s vs hub=%s",
            repo_id, local_path.name, actual, expected_sha,
        )
    return matches


def _preflight_auth() -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().whoami()
    actual = info.get("name")
    if actual != EXPECTED_HF_USERNAME:
        raise PermissionError(
            f"HF Hub token belongs to '{actual}', expected "
            f"'{EXPECTED_HF_USERNAME}'. Run `huggingface-cli login` with "
            f"the correct token before re-running this script."
        )
    _LOGGER.info("HF Hub pre-flight ok: user=%s", actual)


def _pull_one_file(
    api: "HfApi",
    repo_id: str,
    filename: str,
    target_dir: Path,
    hub_lfs_shas: dict[str, str],
) -> _PullResult:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import EntryNotFoundError  # noqa: PLC0415

    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / filename
    expected_sha = hub_lfs_shas.get(filename)

    # Idempotency: if file already exists and (for LFS files) SHA matches, skip.
    if local_path.is_file():
        if expected_sha is None:
            _LOGGER.info("Skip (already present, non-LFS): %s/%s", repo_id, filename)
            return _PullResult(
                repo=repo_id, filename=filename, local_path=local_path,
                size_bytes=local_path.stat().st_size, sha256_match=None,
                status="skipped",
            )
        if sha256sum(local_path).lower() == expected_sha.lower():
            _LOGGER.info("Skip (already present, SHA matches): %s/%s", repo_id, filename)
            return _PullResult(
                repo=repo_id, filename=filename, local_path=local_path,
                size_bytes=local_path.stat().st_size, sha256_match=True,
                status="skipped",
            )
        _LOGGER.warning("Local %s/%s exists but SHA differs — re-downloading.", repo_id, filename)
        try:
            local_path.unlink()
        except OSError:
            pass

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
    except EntryNotFoundError:
        _LOGGER.warning("Repo %s has no %s — skipping (optional file).", repo_id, filename)
        return _PullResult(
            repo=repo_id, filename=filename, local_path=None,
            size_bytes=0, sha256_match=None, status="missing",
        )

    downloaded_path = Path(downloaded)
    sha_match = _verify_sha256(downloaded_path, expected_sha, repo_id)
    status = "downloaded"
    if sha_match is False:
        status = "mismatch"
    return _PullResult(
        repo=repo_id, filename=filename, local_path=downloaded_path,
        size_bytes=downloaded_path.stat().st_size, sha256_match=sha_match,
        status=status,
    )


def main() -> int:
    _preflight_auth()
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    results: list[_PullResult] = []
    mismatches = 0

    for repo_id, subdir in MODEL_REPOS:
        _LOGGER.info("===== %s -> models/disease/%s/ =====", repo_id, subdir)
        target_dir = DISEASE_MODELS_ROOT / subdir
        try:
            hub_shas = _hub_sha256_lookup(api, repo_id)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Could not fetch siblings for %s: %s", repo_id, exc)
            continue

        for filename in FILES_TO_PULL:
            r = _pull_one_file(api, repo_id, filename, target_dir, hub_shas)
            results.append(r)
            if r.sha256_match is False:
                mismatches += 1

    # Final summary table.
    print()
    print("=" * 88)
    print(
        f"{'Repo':<42} {'File':<24} {'Size':>10}  {'SHA':<6}  Status"
    )
    print("-" * 88)
    for r in results:
        size_str = f"{r.size_bytes / 1024**2:.1f} MB" if r.size_bytes > 1024**2 else f"{r.size_bytes / 1024:.1f} KB"
        sha_str = (
            "ok" if r.sha256_match is True
            else ("MISS" if r.sha256_match is False else "—")
        )
        print(
            f"{r.repo:<42} {r.filename:<24} {size_str:>10}  {sha_str:<6}  {r.status}"
        )
    print("=" * 88)
    print(f"  {len(results)} files processed, {mismatches} SHA mismatches.")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
