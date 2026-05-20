"""IKS corpus loader.

Per master reference §12 the corpus comprises Vrikshayurveda, Krishi
Parashara, and Upavanavinoda (plus three optional extensions). Each
source text is digitised in ``corpus/raw/`` then cleaned and chunked into
``corpus/cleaned/`` / ``corpus/chunks/``. Metadata schema captured here
is what contribution C1 promises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IKSDocument:
    """A single source text in the corpus.

    Attributes
    ----------
    source_id : str
        Stable identifier (e.g. ``"vrikshayurveda"``).
    title : str
        Human-readable title.
    language : str
        ISO 639-1 code of the *cleaned* text (e.g. ``"en"`` for an English
        translation). The original Sanskrit may be tracked separately.
    translator : str | None
        Translator credit for English versions.
    license : str | None
        Best-effort copyright / license note. See
        ``notes/iks/copyright_status.md``.
    raw_path : Path
        Path under ``corpus/raw/``.
    cleaned_path : Path | None
        Path under ``corpus/cleaned/`` once preprocessing has run.
    metadata : dict[str, str]
        Free-form tags (era, school, etc.) used at retrieval time.
    """

    source_id: str
    title: str
    language: str
    translator: str | None
    license: str | None
    raw_path: Path
    cleaned_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class IKSCorpus:
    """Loader and registry for all IKS source texts.

    Parameters
    ----------
    raw_dir : Path
        ``corpus/raw/`` directory.
    cleaned_dir : Path
        ``corpus/cleaned/`` directory.
    """

    def __init__(self, raw_dir: Path, cleaned_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.cleaned_dir = Path(cleaned_dir)
        self._documents: list[IKSDocument] = []

    def discover(self) -> list[IKSDocument]:
        """Scan ``raw_dir`` for source texts and populate the registry.

        Returns
        -------
        list[IKSDocument]

        Raises
        ------
        NotImplementedError
            Implementation deferred to Phase 4 — Week 13 (corpus prep).
        """
        raise NotImplementedError("Phase 4 — Week 13: discover IKS source texts.")

    def load_cleaned(self, source_id: str) -> str:
        """Return the cleaned, plain-text body of one source.

        Raises
        ------
        NotImplementedError
            Phase 4 — Week 13.
        """
        raise NotImplementedError("Phase 4 — Week 13: implement cleaned-text loading.")
