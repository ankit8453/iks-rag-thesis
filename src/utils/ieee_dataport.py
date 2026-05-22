"""IEEE DataPort download stub.

Currently a placeholder: the IRSID full version (DOI 10.21227/2zz3-f173)
is on IEEE DataPort and requires institutional or paid access. The
Kaggle mirror used in Phase 4 omits the sieve-analysis ground-truth CSV
that the IEEE version ships with.

When IIITDM institutional IEEE access is confirmed, fill in
:func:`ieee_dataport_download` to authenticate and fetch the archive,
then update ``scripts/download_irsid.py`` to call this function instead
of the Kaggle mirror path.

Per master reference §14, the sieve-analysis CSV is informational only
and must NOT be exposed as a model target by any downstream code.
"""

from __future__ import annotations

from pathlib import Path


def ieee_dataport_download(doi: str, target_dir: Path) -> None:
    """Fetch an IEEE DataPort dataset by DOI into ``target_dir``.

    Parameters
    ----------
    doi : str
        DOI of the IEEE DataPort record (e.g. ``"10.21227/2zz3-f173"`` for
        IRSID).
    target_dir : Path
        Filesystem destination. Must exist.

    Raises
    ------
    NotImplementedError
        Always — fill in once IIITDM institutional IEEE access is
        confirmed.
    """
    raise NotImplementedError(
        "Phase 4 — pending IIITDM IEEE subscription confirmation. "
        f"Would fetch DOI {doi} into {target_dir}."
    )
