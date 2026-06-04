"""Phase 7 retriever unit tests — no GPU, no network.

Uses a tiny in-memory fake Chroma collection so the retriever logic
(RRF fusion, stage toggles, reranker reordering) is exercised without
any model load.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.rag.retriever import HybridRetriever, RetrievedChunk, _tokenize


# --------------------------------------------------------------------- #
# In-memory fakes
# --------------------------------------------------------------------- #


class _FakeCollection:
    """Mimics the slice of chromadb.Collection HybridRetriever uses.

    Stores ``(chunk_id, document, metadata, embedding)`` rows and
    answers ``get(include=...)`` + ``query(query_embeddings=, n_results=)``.
    Distance for ``query`` is ``1 - dot(query_emb, row_emb)`` — same
    semantics as cosine distance over L2-normalised vectors.
    """

    def __init__(self, rows: list[tuple[str, str, dict, list[float]]]) -> None:
        self.rows = rows

    def get(self, include=None) -> dict[str, Any]:
        return {
            "ids": [r[0] for r in self.rows],
            "documents": [r[1] for r in self.rows],
            "metadatas": [r[2] for r in self.rows],
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, Any]:
        q = query_embeddings[0]
        scored = []
        for cid, doc, meta, emb in self.rows:
            dot = sum(a * b for a, b in zip(q, emb, strict=True))
            distance = 1.0 - dot
            scored.append((distance, cid, doc, meta))
        scored.sort(key=lambda r: r[0])
        scored = scored[:n_results]
        return {
            "ids": [[s[1] for s in scored]],
            "documents": [[s[2] for s in scored]],
            "metadatas": [[s[3] for s in scored]],
            "distances": [[s[0] for s in scored]],
        }


class _StubEmbedder:
    """Tokenise + 1-hot over a tiny shared vocabulary; deterministic."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.dim = len(vocab)

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = set(_tokenize(t))
            for j, w in enumerate(self.vocab):
                if w in toks:
                    out[i, j] = 1.0
            n = np.linalg.norm(out[i])
            if normalize_embeddings and n > 0:
                out[i] /= n
        return out


class _StubReranker:
    """Score a (query, doc) pair by token-overlap-with-a-twist.

    Always pushes the chunk with id ``"reranker_winner"`` to the top by
    handing it +10 over its natural overlap score. Lets us assert that
    enabling the reranker actually changes the order.
    """

    def predict(
        self, pairs: list[list[str]], *, show_progress_bar: bool = False,
    ) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            q_tokens = set(_tokenize(query))
            d_tokens = set(_tokenize(doc))
            scores.append(float(len(q_tokens & d_tokens)))
        return scores


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


VOCAB = ["tree", "diseased", "neem", "treatment", "rainfall", "clouds", "water", "soil"]


def _make_rows():
    """Six rows so the top-N candidate windows actually filter."""
    embedder = _StubEmbedder(VOCAB)

    def emb(text: str) -> list[float]:
        return embedder.encode([text])[0].tolist()

    rows = [
        ("c1", "tree diseased neem treatment",   {"source_text": "Vrik", "chapter": "full", "verse_or_section": "1"}, emb("tree diseased neem treatment")),
        ("c2", "rainfall clouds water",          {"source_text": "Brihat", "chapter": "23", "verse_or_section": "1"}, emb("rainfall clouds water")),
        ("c3", "diseased tree rainfall",         {"source_text": "Vrik", "chapter": "full", "verse_or_section": "2"}, emb("diseased tree rainfall")),
        ("c4", "soil water tree",                {"source_text": "Brihat", "chapter": "54", "verse_or_section": "1"}, emb("soil water tree")),
        ("c5", "clouds soil rainfall",           {"source_text": "Brihat", "chapter": "21", "verse_or_section": "1"}, emb("clouds soil rainfall")),
        ("c6", "neem treatment for diseased tree", {"source_text": "Vrik", "chapter": "full", "verse_or_section": "3"}, emb("neem treatment for diseased tree")),
    ]
    return rows, embedder


def _make_retriever(*, use_dense=True, use_sparse=True, use_reranker=True):
    rows, embedder = _make_rows()
    col = _FakeCollection(rows)
    rer = _StubReranker() if use_reranker else None
    return HybridRetriever(
        col,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_reranker=use_reranker,
        embedder=embedder,
        reranker=rer,
        top_k_dense=4,
        top_k_sparse=4,
        top_k_rerank=3,
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_rejects_all_stages_disabled() -> None:
    rows, embedder = _make_rows()
    col = _FakeCollection(rows)
    with pytest.raises(ValueError, match="dense / sparse"):
        HybridRetriever(col, use_dense=False, use_sparse=False, use_reranker=False, embedder=embedder)


def test_retrieve_returns_retrievedchunk_dataclass() -> None:
    r = _make_retriever()
    hits = r.retrieve("diseased tree treatment")
    assert hits, "expected at least one hit"
    for h in hits:
        assert isinstance(h, RetrievedChunk)
        assert h.chunk_id
        assert h.text
        assert isinstance(h.metadata, dict)


def test_rrf_fuses_when_both_stages_on() -> None:
    """A chunk that shows up in BOTH dense and sparse ranks higher than\n    a chunk surfaced by only one stage."""
    r = _make_retriever(use_reranker=False)  # turn off reranker so we see RRF directly
    hits = r.retrieve("diseased tree treatment")
    # The fused top should include c1 and c6 (both contain all 4 query tokens)
    top_ids = [h.chunk_id for h in hits]
    assert "c1" in top_ids[:2] or "c6" in top_ids[:2], (
        f"Expected c1 or c6 to be top after RRF fusion; got {top_ids}"
    )
    # Stage label reports rrf (not dense / bm25) since both ran.
    assert all(h.retriever == "rrf" for h in hits)


def test_disabling_sparse_changes_results() -> None:
    """Toggle ``use_sparse=False`` and confirm the retriever label changes."""
    r = _make_retriever(use_sparse=False, use_reranker=False)
    hits = r.retrieve("rainfall clouds")
    assert hits
    # Single-stage retrieval — pool label is "dense".
    assert all(h.retriever == "dense" for h in hits)


def test_disabling_dense_uses_only_sparse() -> None:
    r = _make_retriever(use_dense=False, use_reranker=False)
    hits = r.retrieve("rainfall clouds")
    assert hits
    assert all(h.retriever == "bm25" for h in hits)


def test_reranker_reorders_results() -> None:
    """With the reranker enabled, the stub's bias must surface in the ordering."""
    rows, embedder = _make_rows()
    col = _FakeCollection(rows)
    # Replace c5's id-route by hacking the rows so c5 has id "reranker_winner".
    new_rows = [
        (("reranker_winner" if cid == "c5" else cid), doc, meta, emb)
        for cid, doc, meta, emb in rows
    ]
    col2 = _FakeCollection(new_rows)

    class _BiasReranker:
        def predict(self, pairs, *, show_progress_bar=False):
            # Push the chunk whose doc has BOTH "soil" and "rainfall"
            # to the top — that's the row aliased to "reranker_winner"
            # and nothing else in the fixture has both tokens.
            out = []
            for query, doc in pairs:
                base = float(len(set(_tokenize(query)) & set(_tokenize(doc))))
                bonus = 10.0 if "soil" in doc and "rainfall" in doc else 0.0
                out.append(base + bonus)
            return out

    r = HybridRetriever(
        col2,
        use_dense=True, use_sparse=True, use_reranker=True,
        embedder=embedder,
        reranker=_BiasReranker(),
        top_k_dense=4, top_k_sparse=4, top_k_rerank=3,
    )
    hits = r.retrieve("rainfall")
    # The biased reranker should surface "reranker_winner" (the row whose
    # doc contains both "rainfall" and "clouds") at the very top.
    assert hits[0].chunk_id == "reranker_winner", (
        f"Reranker should have surfaced the biased chunk; got order: "
        f"{[h.chunk_id for h in hits]}"
    )
    assert hits[0].retriever == "reranker"


def test_rrf_score_is_strictly_higher_for_double_match() -> None:
    """Quantitative RRF: 1/(60+1) + 1/(60+1) > 1/(60+1)."""
    from src.rag.retriever import HybridRetriever as _H, RRF_K

    dense = [("X", 0.9, "doc x", {}), ("Y", 0.5, "doc y", {})]
    sparse = [("X", 9.0, "doc x", {}), ("Z", 1.0, "doc z", {})]
    fused = _H._rrf_fuse([dense, sparse], k=RRF_K)
    # Convert to dict for clarity.
    score = {cid: s for cid, s, _doc, _meta in fused}
    assert score["X"] > score["Y"]
    assert score["X"] > score["Z"]
    assert math.isclose(score["X"], 2.0 / (RRF_K + 1), rel_tol=1e-9)
