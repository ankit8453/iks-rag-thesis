"""Citation verification.

Per supervisor guardrail #5: the RAG generator must cite chunk IDs in the
form ``[chunk_id]``; the eval harness verifies those IDs actually appear
in the retrieved context. This file contains both the IDs-in-text
extractor and the verifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[([a-zA-Z0-9_:\-./]+)\]")


@dataclass
class CitationReport:
    """Outcome of verifying citations in one generated answer.

    Attributes
    ----------
    cited_ids : list[str]
        All chunk IDs extracted from the answer text, in order.
    valid_ids : list[str]
        Subset of ``cited_ids`` that appeared in the retrieved context.
    invalid_ids : list[str]
        Subset of ``cited_ids`` that did NOT appear — potential
        hallucination signal (contribution C4).
    coverage : float
        Fraction of retrieved chunks that were actually cited
        (``len(set(cited)∩retrieved)/len(retrieved)``).
    """

    cited_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    coverage: float = 0.0


def extract_citations(answer: str) -> list[str]:
    """Pull all ``[chunk_id]`` substrings out of an answer.

    Parameters
    ----------
    answer : str
        Generated answer text.

    Returns
    -------
    list[str]
        Chunk IDs in the order they appear. Duplicates are preserved.
    """
    return _CITATION_RE.findall(answer)


def verify_citations_in_context(
    answer: str,
    retrieved_chunk_ids: list[str],
) -> CitationReport:
    """Check that every cited chunk ID actually appears in the retrieved set.

    Parameters
    ----------
    answer : str
        The model's generated answer.
    retrieved_chunk_ids : list[str]
        Chunk IDs actually passed into the prompt.

    Returns
    -------
    CitationReport

    Raises
    ------
    NotImplementedError
        The extractor is implemented; the full verifier with coverage
        statistics is deferred to Phase 9 — Week 30.
    """
    cited = extract_citations(answer)
    retrieved_set = set(retrieved_chunk_ids)
    # Minimal implementation: valid/invalid split + coverage.
    # The Phase 9 version will add per-claim alignment scoring.
    valid = [c for c in cited if c in retrieved_set]
    invalid = [c for c in cited if c not in retrieved_set]
    coverage = (
        len({c for c in cited if c in retrieved_set}) / len(retrieved_set) if retrieved_set else 0.0
    )
    return CitationReport(
        cited_ids=cited,
        valid_ids=valid,
        invalid_ids=invalid,
        coverage=coverage,
    )
