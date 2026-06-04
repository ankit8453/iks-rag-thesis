"""Phase 7 hybrid retriever — dense (BGE) + sparse (BM25) + cross-encoder rerank.

Pipeline:

1. **Dense:** ChromaDB similarity over BAAI/bge-large-en-v1.5 embeddings.
   The collection is built by :func:`src.rag.corpus_loader.build_chroma`
   (Colab) or by Phase 3's local pipeline (laptop).
2. **Sparse:** BM25 over an in-memory list of whitespace-tokenised chunk
   texts. Index is rebuilt at construction since the corpus is small
   (~hundreds of chunks).
3. **Fusion:** Reciprocal Rank Fusion (RRF) of the dense and sparse
   top-N candidate lists with the canonical ``k=60`` constant.
4. **Rerank:** ``BAAI/bge-reranker-base`` cross-encoder scores the fused
   candidates and returns the top-``k``.

All three stages are independently toggleable via the constructor
flags ``use_dense``, ``use_sparse``, ``use_reranker`` — Phase 11 §27
ablations are config flips, not rewrites. Disabling all three is a
configuration error and raises at construction time.

The :class:`RetrievedChunk` dataclass at the bottom of this module is
shared with :mod:`src.explain.chunk_highlight`; do NOT change its
public field names without updating that consumer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder

_LOGGER = get_logger(__name__)


DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-base"

# Per-stage candidate depth before fusion + rerank.
DEFAULT_TOP_K_DENSE: int = 20
DEFAULT_TOP_K_SPARSE: int = 20

# Final number of chunks returned after rerank.
DEFAULT_TOP_K_RERANK: int = 5

# Canonical RRF constant from Cormack et al. 2009.
RRF_K: int = 60


# --------------------------------------------------------------------- #
# Public dataclass (consumed by src/explain/chunk_highlight.py too)
# --------------------------------------------------------------------- #


@dataclass
class RetrievedChunk:
    """A retrieved chunk plus its provenance and score.

    Attributes
    ----------
    chunk_id
        Deterministic sha1 id from Phase 3 chunking. Used as the
        chunk's primary key in ChromaDB and in citation strings.
    text
        Cleaned English passage text.
    metadata
        Phase 3 metadata dict — at minimum ``source_text``, ``chapter``,
        ``verse_or_section``, ``translator``, ``book_id``,
        ``topic_tags``. The generator's prompt-builder reads these
        fields to render precise citations.
    score
        Final score (rerank score if reranker ran, else fused RRF
        score, else single-stage score). Higher is better.
    retriever
        Which stage produced the score: ``"reranker"``, ``"rrf"``,
        ``"dense"``, or ``"bm25"``.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    retriever: str = "reranker"

    # Optional debug fields populated by HybridRetriever — not part of
    # the public contract with src.explain.chunk_highlight.
    debug: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


_TOKEN_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokeniser shared by BM25 indexing and queries."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


# --------------------------------------------------------------------- #
# HybridRetriever
# --------------------------------------------------------------------- #


class HybridRetriever:
    """Dense + sparse + cross-encoder hybrid retriever.

    Parameters
    ----------
    collection
        A ChromaDB collection (``chromadb.PersistentClient`` →
        ``get_or_create_collection``) populated by
        :func:`src.rag.corpus_loader.build_chroma`.
    use_dense, use_sparse, use_reranker
        Independent toggles for each stage. At least one of ``dense``
        / ``sparse`` must be on.
    embedder
        Optional pre-loaded ``SentenceTransformer`` for dense query
        encoding. If ``None``, the constructor loads
        :data:`DEFAULT_EMBEDDING_MODEL`. Passing your own keeps a
        single shared model across notebook cells.
    reranker
        Optional pre-loaded ``CrossEncoder``. If ``None`` and
        ``use_reranker`` is True, the constructor loads
        :data:`DEFAULT_RERANKER_MODEL`.
    top_k_dense, top_k_sparse
        Candidate depth per stage before fusion.
    top_k_rerank
        Final size of the returned list.

    Methods
    -------
    retrieve(query, k=None)
        Return ``RetrievedChunk`` results in descending score order.
    """

    def __init__(
        self,
        collection: "Collection",
        *,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_reranker: bool = True,
        embedder: "SentenceTransformer | None" = None,
        reranker: "CrossEncoder | None" = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        top_k_dense: int = DEFAULT_TOP_K_DENSE,
        top_k_sparse: int = DEFAULT_TOP_K_SPARSE,
        top_k_rerank: int = DEFAULT_TOP_K_RERANK,
    ) -> None:
        if not (use_dense or use_sparse):
            raise ValueError(
                "HybridRetriever needs at least one of dense / sparse enabled."
            )
        self.collection = collection
        self.use_dense = bool(use_dense)
        self.use_sparse = bool(use_sparse)
        self.use_reranker = bool(use_reranker)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.top_k_dense = int(top_k_dense)
        self.top_k_sparse = int(top_k_sparse)
        self.top_k_rerank = int(top_k_rerank)

        # ---- dense
        self._embedder: SentenceTransformer | None = embedder
        if self.use_dense and self._embedder is None:
            self._embedder = self._load_embedder(embedding_model)

        # ---- sparse: snapshot the full collection into BM25
        self._sparse_index = None
        self._sparse_ids: list[str] = []
        self._sparse_texts: list[str] = []
        self._sparse_metas: list[dict[str, Any]] = []
        if self.use_sparse:
            self._build_sparse_index()

        # ---- reranker
        self._reranker: CrossEncoder | None = reranker
        if self.use_reranker and self._reranker is None:
            self._reranker = self._load_reranker(reranker_model)

        _LOGGER.info(
            "HybridRetriever: dense=%s sparse=%s rerank=%s "
            "top_k_dense=%d top_k_sparse=%d top_k_rerank=%d",
            self.use_dense, self.use_sparse, self.use_reranker,
            self.top_k_dense, self.top_k_sparse, self.top_k_rerank,
        )

    # ---- lazy loaders ------------------------------------------------ #

    def _load_embedder(self, name: str) -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOGGER.info("Loading embedding model %s on %s ...", name, device)
        return SentenceTransformer(name, device=device)

    def _load_reranker(self, name: str) -> "CrossEncoder":
        from sentence_transformers.cross_encoder import CrossEncoder  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOGGER.info("Loading cross-encoder reranker %s on %s ...", name, device)
        return CrossEncoder(name, device=device)

    # ---- sparse build ------------------------------------------------ #

    def _build_sparse_index(self) -> None:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        # Pull the entire collection — fine for the Phase 3 corpus size.
        snapshot = self.collection.get(include=["documents", "metadatas"])
        self._sparse_ids = list(snapshot.get("ids") or [])
        self._sparse_texts = list(snapshot.get("documents") or [])
        self._sparse_metas = list(snapshot.get("metadatas") or [])
        if not self._sparse_ids:
            _LOGGER.warning("HybridRetriever: collection is empty; BM25 index will return no hits.")
            self._sparse_index = None
            return
        tokenised = [_tokenize(doc) for doc in self._sparse_texts]
        self._sparse_index = BM25Okapi(tokenised)
        _LOGGER.info(
            "HybridRetriever: BM25 index built over %d chunks.", len(self._sparse_ids),
        )

    # ---- per-stage candidates --------------------------------------- #

    def _dense_candidates(self, query: str, n: int) -> list[tuple[str, float, str, dict]]:
        """Return ``[(chunk_id, similarity_score, document, metadata), ...]``."""
        if self._embedder is None:
            return []
        query_emb = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        res = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=n,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[tuple[str, float, str, dict]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            # Cosine distance → similarity (higher is better).
            similarity = float(1.0 - dist) if dist is not None else 0.0
            out.append((cid, similarity, doc or "", meta or {}))
        return out

    def _sparse_candidates(self, query: str, n: int) -> list[tuple[str, float, str, dict]]:
        if self._sparse_index is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._sparse_index.get_scores(tokens)
        # Argsort descending.
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:n]
        out: list[tuple[str, float, str, dict]] = []
        for i in order:
            out.append(
                (
                    self._sparse_ids[i],
                    float(scores[i]),
                    self._sparse_texts[i],
                    self._sparse_metas[i],
                )
            )
        return out

    # ---- fusion + rerank ------------------------------------------- #

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[tuple[str, float, str, dict]]],
        k: int = RRF_K,
    ) -> list[tuple[str, float, str, dict]]:
        """Reciprocal Rank Fusion across multiple per-stage ranked lists.

        For each chunk, the fused score is ``sum(1 / (k + rank))`` over
        every list that surfaced it. Higher is better. The fused list
        is returned in descending fused-score order. Document and
        metadata are taken from whichever list first contained the
        chunk (sparse and dense agree on text + metadata anyway).
        """
        scores: dict[str, float] = {}
        records: dict[str, tuple[str, dict]] = {}
        for ranked in ranked_lists:
            for rank, (cid, _stage_score, doc, meta) in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                records.setdefault(cid, (doc, meta))
        fused = [
            (cid, scores[cid], records[cid][0], records[cid][1])
            for cid in scores
        ]
        fused.sort(key=lambda r: -r[1])
        return fused

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, str, dict]],
        k: int,
    ) -> list[tuple[str, float, str, dict]]:
        if not candidates or self._reranker is None:
            return candidates[:k]
        pairs = [[query, c[2]] for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        scored = [
            (cand[0], float(score), cand[2], cand[3])
            for cand, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda r: -r[1])
        return scored[:k]

    # ---- public API ------------------------------------------------- #

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Run the toggled pipeline and return the top-``k`` chunks.

        Parameters
        ----------
        query
            User question / topic.
        k
            Optional override for the returned-list size. Defaults to
            ``top_k_rerank`` from the constructor.

        Returns
        -------
        list[RetrievedChunk]
            Descending-score order. Empty list if the collection is
            empty or every stage was disabled (caller filtered).
        """
        if k is None:
            k = self.top_k_rerank

        per_stage: list[list[tuple[str, float, str, dict]]] = []
        if self.use_dense:
            dense_cands = self._dense_candidates(query, self.top_k_dense)
            per_stage.append(dense_cands)
        if self.use_sparse:
            sparse_cands = self._sparse_candidates(query, self.top_k_sparse)
            per_stage.append(sparse_cands)

        # Pre-rerank candidate pool: RRF-fused if both stages were on, else
        # whichever single stage was on.
        if len(per_stage) == 2:
            pool = self._rrf_fuse(per_stage, k=RRF_K)
            pool_stage = "rrf"
        elif len(per_stage) == 1:
            pool = per_stage[0]
            pool_stage = "dense" if self.use_dense else "bm25"
        else:
            return []

        if self.use_reranker:
            final = self._rerank(query, pool, k=k)
            final_stage = "reranker"
        else:
            final = pool[:k]
            final_stage = pool_stage

        return [
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                metadata=meta,
                score=score,
                retriever=final_stage,
                debug={"pool_stage": pool_stage, "pool_rank": rank + 1},
            )
            for rank, (cid, score, doc, meta) in enumerate(final)
        ]


# --------------------------------------------------------------------- #
# Deprecated backward-compat stubs (Phase 4 era)
# --------------------------------------------------------------------- #
#
# The dense and BM25 stages live inside ``HybridRetriever`` now. The
# Phase-4 ``DenseRetriever`` / ``BM25Retriever`` classes were never
# implemented (every method raised ``NotImplementedError``); they're
# kept as deprecated docstring-bearing aliases so a few pre-Phase-7
# smoke tests (``tests/rag/test_smoke.py``) still import cleanly.


class DenseRetriever:
    """Deprecated: dense retrieval lives inside :class:`HybridRetriever`.

    Set ``use_dense=True, use_sparse=False`` on :class:`HybridRetriever`
    for the single-stage equivalent.
    """


class BM25Retriever:
    """Deprecated: BM25 retrieval lives inside :class:`HybridRetriever`.

    Set ``use_sparse=True, use_dense=False`` on :class:`HybridRetriever`
    for the single-stage equivalent.
    """


__all__ = [
    "BM25Retriever",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_TOP_K_DENSE",
    "DEFAULT_TOP_K_RERANK",
    "DEFAULT_TOP_K_SPARSE",
    "DenseRetriever",
    "HybridRetriever",
    "RRF_K",
    "RetrievedChunk",
]
