"""Matplotlib-based visualisations for the Phase 9 explainability layer.

Two figure factories + one disk-save helper. Kept matplotlib-only so the
same code works in the notebook (inline) and in the Phase 10 Streamlit
UI (``st.pyplot``) without code duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logging_setup import get_logger
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    import matplotlib.figure  # noqa: F401

    from src.explain.chunk_highlight import ExplainedChunk  # noqa: F401
    from src.explain.gradcam import GradCAMResult  # noqa: F401

_LOGGER = get_logger(__name__)

# Default output root. Master plan §41: results/ is fine to commit; the
# images are PNGs (a few hundred KB each), not raw data.
DEFAULT_OUT_ROOT: Path = PROJECT_ROOT / "results" / "explainability"


# --------------------------------------------------------------------- #
# Vision panel
# --------------------------------------------------------------------- #


def render_vision_panel(
    sample_name: str,
    original_leaf: Any,
    disease_cam: "GradCAMResult",
    original_soil: Any,
    soil_cams: dict[str, "GradCAMResult"],
) -> Any:
    """Compose a 2-row vision-explanation figure.

    Row 1 — original leaf | disease Grad-CAM (captioned with the
            predicted disease label + confidence).
    Row 2 — original soil | soil_type CAM | moisture CAM | texture CAM
            (each captioned with its head's predicted label + confidence).

    Parameters
    ----------
    sample_name
        Short label used in the figure suptitle (e.g.
        ``"tomato / alluvial / soil_driven"``).
    original_leaf, original_soil
        H×W×3 uint8 numpy arrays of the original images.
    disease_cam
        :class:`GradCAMResult` from
        :func:`~src.explain.gradcam.disease_gradcam`.
    soil_cams
        Dict keyed by ``"soil_type"`` / ``"moisture"`` / ``"texture"`` →
        :class:`GradCAMResult`.

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for closing the figure (or letting
        :func:`save_explanation` do it).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 2 rows × 4 cols — row 1 only uses cols 0+1, row 2 uses all 4.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Phase 9 — Vision explanation: {sample_name}", fontsize=14, y=0.99)

    # ----- row 1: leaf + disease CAM ------------------------------ #
    axes[0, 0].imshow(original_leaf)
    axes[0, 0].set_title("Original leaf")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(disease_cam.overlay_rgb)
    axes[0, 1].set_title(
        f"Disease Grad-CAM\n{disease_cam.pred_label} ({disease_cam.pred_conf:.2f})",
    )
    axes[0, 1].axis("off")

    # Hide unused tiles in row 1 (the disease model has one head).
    for col in (2, 3):
        axes[0, col].axis("off")

    # ----- row 2: soil + 3 head CAMs ------------------------------ #
    axes[1, 0].imshow(original_soil)
    axes[1, 0].set_title("Original soil")
    axes[1, 0].axis("off")

    for col, head in enumerate(("soil_type", "moisture", "texture"), start=1):
        cam = soil_cams.get(head)
        ax = axes[1, col]
        if cam is None:
            ax.set_title(f"{head}: (missing)")
            ax.axis("off")
            continue
        ax.imshow(cam.overlay_rgb)
        ax.set_title(
            f"{head} Grad-CAM\n{cam.pred_label} ({cam.pred_conf:.2f})",
        )
        ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# --------------------------------------------------------------------- #
# Retrieval panel
# --------------------------------------------------------------------- #


def render_retrieval_panel(
    sample_name: str,
    query: str,
    explained_chunks: list["ExplainedChunk"],
) -> Any:
    """Compose a 2-column retrieval-explanation figure.

    Left column — horizontal bar chart of the top-k chunks' similarity
                  scores (rank 1 at the top).
    Right column — wrapped-text listing of each chunk's
                   ``source ch.X v.Y`` header, the matched query terms,
                   and a short ``text_with_markers`` snippet.

    Parameters
    ----------
    sample_name
        Short label for the figure suptitle.
    query
        The retrieval query the chunks were ranked for (rendered above
        the listing so reviewers can read query and chunk side-by-side).
    explained_chunks
        Output of :func:`~src.explain.chunk_highlight.explain_chunks`.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    k = len(explained_chunks)
    if k == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        ax.text(
            0.5, 0.5, "no chunks retrieved", ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(16, max(4, k * 1.2)))
    fig.suptitle(f"Phase 9 — Retrieval explanation: {sample_name}", fontsize=14, y=0.98)

    # ----- bar chart ---------------------------------------------- #
    ranks = [c.rank for c in explained_chunks]
    scores = [c.score for c in explained_chunks]
    labels = [
        f"#{c.rank} {c.source_text} ch.{c.chapter} v.{c.verse_or_section}"
        for c in explained_chunks
    ]
    ax_bar.barh(range(k), scores, color="#3b78c2")
    ax_bar.set_yticks(range(k))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("similarity score")
    ax_bar.set_title("Top-k chunks by similarity")
    for rank_idx, score in zip(ranks, scores, strict=True):
        ax_bar.text(score, rank_idx - 1, f" {score:.3f}", va="center", fontsize=8)

    # ----- text listing ------------------------------------------- #
    ax_text.axis("off")
    lines: list[str] = [f"QUERY: {query}", ""]
    for c in explained_chunks:
        header = (
            f"#{c.rank}  score={c.score:.3f}  "
            f"{c.source_text} ch.{c.chapter} v.{c.verse_or_section}"
        )
        matched = ", ".join(c.matched_terms) if c.matched_terms else "(no overlap)"
        snippet = _shorten(c.text_with_markers, max_chars=240)
        lines.extend([header, f"  matched: {matched}", f"  {snippet}", ""])
    ax_text.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left",
        family="monospace", fontsize=9, transform=ax_text.transAxes,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def _shorten(text: str, max_chars: int) -> str:
    """One-paragraph collapse + ellipsis truncation."""
    if not text:
        return ""
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3].rstrip() + "..."


# --------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------- #


def save_explanation(
    sample_name: str,
    vision_fig: Any,
    retrieval_fig: Any,
    out_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Persist both figures as PNGs under ``results/explainability/<sample_name>/``.

    Returns
    -------
    (vision_path, retrieval_path)
        Absolute :class:`Path` objects to the two saved PNGs.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    root = Path(out_dir) if out_dir is not None else DEFAULT_OUT_ROOT
    safe = _safe_name(sample_name)
    dest = root / safe
    dest.mkdir(parents=True, exist_ok=True)

    vision_path = dest / "vision_panel.png"
    retrieval_path = dest / "retrieval_panel.png"

    vision_fig.savefig(vision_path, dpi=140, bbox_inches="tight")
    retrieval_fig.savefig(retrieval_path, dpi=140, bbox_inches="tight")
    plt.close(vision_fig)
    plt.close(retrieval_fig)

    _LOGGER.info("Saved vision panel    → %s", vision_path)
    _LOGGER.info("Saved retrieval panel → %s", retrieval_path)
    return vision_path, retrieval_path


def _safe_name(name: str) -> str:
    """Filesystem-friendly directory name from a free-form sample label."""
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "/", "\\"):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "sample"


__all__ = [
    "DEFAULT_OUT_ROOT",
    "render_retrieval_panel",
    "render_vision_panel",
    "save_explanation",
]
