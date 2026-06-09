"""Multimodal-embedding integration strategy (Strategy C, honest ablation).

Concatenates the penultimate disease feature vector (1792-dim B4) + the
penultimate soil feature vector (1280-dim B0) + a crop-name embedding
(1024-dim bge-large), then projects the concatenation linearly into the
1024-dim corpus embedding space and retrieves by cosine similarity from
the ChromaDB collection.

**Honest framing — this is the ablation that demonstrates *why* text
queries win for a text corpus, not a competitor to Strategies A / B.**
The projection is trained on WEAK pairs — for each demo sample, we
embed Strategy A's templated query with the same bge-large encoder, take
the top-1 retrieved chunk's embedding, and fit the projector to map
the concatenated visual embeddings → that target embedding. No manual
relevance labels. A few epochs of MSE. The modality gap (visual
embeddings live on a different manifold from text embeddings) means
this is expected to under-perform Strategies A and B; the comparison
notebook makes that visible. A fully-trained version is future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.integration.config import MultimodalEmbeddingStrategyConfig
from src.integration.context import MultimodalContext

# Phase 7 corpus embedding model & dimension.
CORPUS_EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
CORPUS_EMBEDDING_DIM: int = 1024

# Phase 5 / 6 backbone feature dimensions.
DISEASE_FEAT_DIM: int = 1792   # efficientnet_b4
SOIL_FEAT_DIM: int = 1280      # efficientnet_b0


@dataclass
class WeakProjectionTrainingResult:
    """Returned by :func:`MultimodalEmbeddingStrategy.train_weak_projection`."""

    epochs: int
    final_loss: float
    n_samples: int


class MultimodalProjector(nn.Module):
    """Linear (disease_dim + soil_dim + crop_emb_dim → 1024).

    A single ``nn.Linear`` layer. Light by design — the modality gap is
    not crossed by depth, it's crossed by data. Phase 8 calls this out
    explicitly as an ablation and defers a properly-trained projector
    (with manual relevance judgments) to future work.
    """

    def __init__(
        self,
        disease_dim: int = DISEASE_FEAT_DIM,
        soil_dim: int = SOIL_FEAT_DIM,
        crop_dim: int = CORPUS_EMBEDDING_DIM,
        out_dim: int = CORPUS_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        in_dim = disease_dim + soil_dim + crop_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.disease_dim = disease_dim
        self.soil_dim = soil_dim
        self.crop_dim = crop_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultimodalEmbeddingStrategy:
    """Strategy C orchestrator: build query vector, train projector, retrieve.

    Parameters
    ----------
    config
        :class:`MultimodalEmbeddingStrategyConfig`. Currently only
        ``projection_dim`` is read for sanity-checking against the corpus
        embedding dim.
    """

    def __init__(self, config: MultimodalEmbeddingStrategyConfig) -> None:
        self.config = config
        if config.projection_dim != CORPUS_EMBEDDING_DIM:
            # Soft-warn rather than raise — the bge-large corpus dim is
            # locked, but a researcher might run an ablation with a
            # different embedder.
            pass

    # ------------------------------------------------------------- #
    # Crop name → text embedding
    # ------------------------------------------------------------- #

    def crop_embedding(self, crop_type: str, embedder: Any) -> np.ndarray:
        """Embed the crop name with the same model that embedded the corpus.

        Parameters
        ----------
        crop_type : str
            Farmer-supplied crop name (e.g. ``"rice"``).
        embedder
            A ``SentenceTransformer`` instance whose ``.encode()`` returns
            unit-normalised vectors of dim :data:`CORPUS_EMBEDDING_DIM`.
            (Phase 7's ``corpus_loader._embed_texts`` uses one of these.)
        """
        text = (crop_type or "unspecified crop").strip()
        vec = embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(vec[0], dtype=np.float32)

    # ------------------------------------------------------------- #
    # Build the fused visual+crop vector that feeds the projector
    # ------------------------------------------------------------- #

    def build_input_vector(
        self,
        context: MultimodalContext,
        embedder: Any,
    ) -> np.ndarray:
        """Concatenate (disease_emb, soil_emb, crop_emb) → 1-D float32 array."""
        if context.disease_emb is None or context.soil_emb is None:
            raise ValueError(
                "Strategy C needs context.disease_emb and context.soil_emb. "
                "Re-run build_multimodal_context(..., capture_embeddings=True)."
            )
        crop_emb = self.crop_embedding(context.crop_type, embedder)
        vec = np.concatenate(
            [
                np.asarray(context.disease_emb, dtype=np.float32),
                np.asarray(context.soil_emb, dtype=np.float32),
                crop_emb,
            ]
        )
        return vec

    # ------------------------------------------------------------- #
    # Weak-signal projector training
    # ------------------------------------------------------------- #

    def train_weak_projection(
        self,
        samples: list[MultimodalContext],
        embedder: Any,
        template_query_fn: Any,
        chroma_collection: Any,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ) -> tuple[MultimodalProjector, WeakProjectionTrainingResult]:
        """Fit the projector against Strategy A's top-1 retrieved chunk.

        Parameters
        ----------
        samples
            List of :class:`MultimodalContext` objects (each with the
            penultimate embeddings populated).
        embedder
            Sentence-transformers encoder used by Phase 7 to embed the
            corpus.
        template_query_fn
            ``MultimodalContext -> str`` — typically
            :meth:`~src.integration.strategy_template.TemplateStrategy.build_query`.
            Used to produce the weak target query for each sample.
        chroma_collection
            ChromaDB collection (already populated with the 206 chunks).
        epochs, lr, seed
            Training hyperparameters. Defaults are deliberately modest —
            this is an ablation, not a tuned competitor.

        Returns
        -------
        (MultimodalProjector, WeakProjectionTrainingResult)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---- assemble (input, target) pairs -----------------------
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for ctx in samples:
            x = self.build_input_vector(ctx, embedder)
            query = template_query_fn(ctx)
            qvec = embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
            )[0]
            hit = chroma_collection.query(
                query_embeddings=[qvec.tolist()],
                n_results=1,
                include=["embeddings"],
            )
            embs = hit.get("embeddings") or []
            if not embs or not embs[0]:
                continue
            top1 = np.asarray(embs[0][0], dtype=np.float32)
            inputs.append(x)
            targets.append(top1)

        if not inputs:
            raise RuntimeError(
                "train_weak_projection: no weak pairs assembled — "
                "the ChromaDB returned no chunk embeddings."
            )

        X = torch.from_numpy(np.stack(inputs)).to(device).float()
        Y = torch.from_numpy(np.stack(targets)).to(device).float()

        projector = MultimodalProjector(
            disease_dim=DISEASE_FEAT_DIM,
            soil_dim=SOIL_FEAT_DIM,
            crop_dim=CORPUS_EMBEDDING_DIM,
            out_dim=CORPUS_EMBEDDING_DIM,
        ).to(device)
        opt = torch.optim.Adam(projector.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        projector.train()
        last_loss = float("nan")
        for _ in range(epochs):
            pred = projector(X)
            loss = loss_fn(pred, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        projector.eval()

        return projector, WeakProjectionTrainingResult(
            epochs=epochs, final_loss=last_loss, n_samples=len(inputs),
        )

    # ------------------------------------------------------------- #
    # Retrieval
    # ------------------------------------------------------------- #

    def retrieve_via_embedding(
        self,
        context: MultimodalContext,
        projector: MultimodalProjector,
        chroma_collection: Any,
        embedder: Any,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Project visual+crop → 1024-d query vector → top-k ChromaDB hits.

        Returns a list of dicts with ``chunk_id`` / ``text`` /
        ``metadata`` / ``score`` so the shape mirrors
        :class:`~src.rag.retriever.RetrievedChunk`. Used by
        :func:`src.integration.compare.run_all_strategies`.
        """
        x = self.build_input_vector(context, embedder)
        with torch.no_grad():
            qvec = projector(torch.from_numpy(x).float().unsqueeze(0)).squeeze(0)
        # cosine similarity is the corpus metric (bge-large is normalised).
        qvec = qvec / max(float(qvec.norm().item()), 1e-12)
        results = chroma_collection.query(
            query_embeddings=[qvec.cpu().numpy().tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunk_ids = (results.get("ids") or [[]])[0]
        texts = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for cid, txt, meta, dist in zip(chunk_ids, texts, metas, dists, strict=False):
            # bge cosine distance in [0, 2]; convert to similarity in [-1, 1].
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append({
                "chunk_id": cid,
                "text": txt,
                "metadata": meta or {},
                "score": score,
                "retriever": "embedding_projection",
            })
        return out

    # ------------------------------------------------------------- #
    # Legacy stub kept for backward-compat with earlier test imports.
    # ------------------------------------------------------------- #

    def build_query_embedding(self, context: MultimodalContext) -> np.ndarray:
        """Deprecated thin wrapper — left in place because the Phase 4
        skeleton's tests used to call it.

        Use :meth:`build_input_vector` (followed by projection) instead.
        """
        raise NotImplementedError(
            "Use build_input_vector(...) + projector(...) — Strategy C's "
            "fused visual+crop concatenation has moved to "
            "build_input_vector; the projection is applied at retrieval time."
        )
