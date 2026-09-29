"""Draw the chapter's figures as clean block diagrams (PNG, 300 dpi).

Fig 1  system architecture — two vision models, the symptom bridge, retrieval, grounded
       answer or refusal
Fig 2  where the two traditions meet — classical texts organise plant disorder by CAUSE,
       modern diagnosis by APPEARANCE; only the overlap is answerable
Fig 3  what moved the numbers — grounded-answer rate and over-refusal across the three
       evaluation stages

Plain matplotlib, no styling library: boxes, arrows, one accent colour. Labels are placed
in the gaps between boxes, never across them (v1 had three collisions). Run once; the
Markdown draft references paper/icssr_chapter/figures/fig*.png.

Usage:  python paper/icssr_chapter/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1f2a37"
ACCENT = "#1F4E79"
SOFT = "#e9eef5"
SOFT2 = "#f4efe6"
GREY = "#6b7280"
RED = "#9b3b3b"
GREEN = "#2f6f4e"
FONT = {"family": "DejaVu Sans"}


def box(ax, x, y, w, h, title, body="", *, fill=SOFT, edge=ACCENT, tsize=10, bsize=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=1.4))
    ty = y + h * (0.70 if body else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize,
            fontweight="bold", color=INK, **FONT)
    if body:
        ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", fontsize=bsize,
                color=INK, **FONT, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, *, lw=1.5, color=ACCENT):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2))


def line(ax, x1, y1, x2, y2, *, lw=1.2, color=GREY):
    ax.plot([x1, x2], [y1, y2], lw=lw, color=color, solid_capstyle="round")


def label(ax, x, y, text, *, ha="center", va="center", color=GREY, size=7.6):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, **FONT)


# ------------------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # column 1: inputs
    box(ax, 0.03, 0.74, 0.17, 0.15, "Leaf photograph", "from a phone camera", fill=SOFT2)
    box(ax, 0.03, 0.50, 0.17, 0.15, "Soil photograph", "of the field surface", fill=SOFT2)
    box(ax, 0.03, 0.26, 0.17, 0.15, "Crop name", "typed by the farmer", fill=SOFT2)

    # column 2: vision models
    box(ax, 0.27, 0.74, 0.21, 0.15, "Disease model",
        "EfficientNet-B4, retrained on\nleaf crops; heat-map checked")
    box(ax, 0.27, 0.50, 0.21, 0.15, "Soil model",
        "EfficientNet-B0: soil type,\nmoisture, texture")

    # column 3: bridge over corpus (x 0.55-0.72)
    box(ax, 0.55, 0.56, 0.17, 0.30, "Symptom bridge",
        "Llama 3.1 rewrites the label:\n“Apple scab” becomes\n“dark rough corky patches\nspreading over the leaves”",
        bsize=7.4)
    box(ax, 0.55, 0.14, 0.17, 0.30, "Digital corpus",
        "270 passages, 7 sources:\nsix classical treatises\n+ one modern manual;\nsearch and re-ranking",
        bsize=7.4)

    # column 4: outcomes (x 0.83-0.97), leaving a 0.11 gap for the two labels
    box(ax, 0.83, 0.56, 0.14, 0.30, "Grounded answer",
        "cites [book, chapter,\nverse] for every claim", fill="#e6f2ea", edge=GREEN, bsize=7.4)
    box(ax, 0.83, 0.14, 0.14, 0.30, "Honest refusal",
        "“the texts do not\ncontain enough\ninformation”", fill="#fbeaea", edge=RED, bsize=7.4)

    # inputs -> models
    arrow(ax, 0.20, 0.815, 0.27, 0.815)
    arrow(ax, 0.20, 0.575, 0.27, 0.575)

    # models -> bridge. Only the disease arrow is labelled: it is the thing the bridge
    # rewrites. The soil arrow is self-explanatory and a label there collided with the
    # crop route.
    arrow(ax, 0.48, 0.815, 0.55, 0.78)
    label(ax, 0.515, 0.855, "disease label")
    arrow(ax, 0.48, 0.575, 0.55, 0.64)

    # crop -> bridge, as an elbow up the gap column (context only)
    line(ax, 0.20, 0.335, 0.515, 0.335)
    arrow(ax, 0.515, 0.335, 0.515, 0.56, color=GREY, lw=1.2)
    label(ax, 0.355, 0.355, "crop, as background context only", va="bottom")

    # bridge -> corpus; label to the RIGHT of the arrow, in the pocket between the
    # column-3 boxes and the green diagonal
    arrow(ax, 0.635, 0.56, 0.635, 0.44)
    label(ax, 0.652, 0.50, "query, in the\ntexts’ own words", ha="left")

    # corpus -> outcomes. "passages found" sits in the pocket between the two
    # right-hand boxes, just above the diagonal; "nothing fits" is stacked in the gap.
    arrow(ax, 0.72, 0.40, 0.895, 0.56, color=GREEN)
    label(ax, 0.795, 0.505, "passages\nfound", color=GREEN)
    arrow(ax, 0.72, 0.22, 0.83, 0.22, color=RED)
    label(ax, 0.775, 0.27, "nothing\nfits", color=RED, size=7.0)

    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------- Fig 2
def fig2():
    fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.04, 0.64, 0.34, 0.28, "Classical treatises organise disorder by CAUSE",
        "wind · bile · phlegm imbalance\nover-watering · over-manuring\ninsects · wounds · fire · unhealthy soil",
        fill=SOFT2, edge="#8a6d3b", tsize=9.2)
    box(ax, 0.04, 0.26, 0.34, 0.32, "…and describe symptoms as",
        "paleness, yellowness, drying,\nwithering, shedding, dieback,\nfalling bark, oozing,\nfailure to flower or fruit",
        fill=SOFT2, edge="#8a6d3b", tsize=9.2)

    box(ax, 0.62, 0.64, 0.34, 0.28, "Modern diagnosis names disease by APPEARANCE",
        "apple scab · rust · Septoria leaf spot\nearly / late blight · powdery mildew\nleaf mould · black rot", tsize=9.2)
    box(ax, 0.62, 0.26, 0.34, 0.32, "…and describe symptoms as",
        "spots with pale centres, pustules,\ncorky patches, powdery coating,\nfuzzy mould, sunken decaying areas", tsize=9.2)

    box(ax, 0.41, 0.40, 0.18, 0.36, "Overlap",
        "yellowing · drying\ninsect damage\npoor soil · wounds\n\nanswerable", fill="#e6f2ea", edge=GREEN, tsize=9.6)
    arrow(ax, 0.38, 0.58, 0.41, 0.58, color=GREEN)
    arrow(ax, 0.62, 0.58, 0.59, 0.58, color=GREEN)

    ax.text(0.50, 0.19, "No passage in any of the seven sources describes a lesion’s appearance:\n"
            "11 of 17 disease-name questions have no answer.",
            ha="center", va="top", fontsize=8.4, color=RED, **FONT, linespacing=1.4)
    fig.savefig(OUT / "fig2_boundary.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------- Fig 3
def fig3():
    stages = ["Stage 1\n233 passages\nhand-written wording\nduplicate labels",
              "Stage 2\n233 passages\nsystem’s own wording\nunique labels",
              "Stage 3 (final)\n270 passages\n+ Vishvavallabha\ncitation format fixed"]
    grounded = [13.6, 16.7, 51.9]
    over = [81.8, 83.3, 40.7]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=300)
    x = range(len(stages)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], grounded, w, color=ACCENT, label="answers backed by a real citation")
    b2 = ax.bar([i + w / 2 for i in x], over, w, color="#c9a227", label="answerable questions refused")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=INK, **FONT)
    ax.set_xticks(list(x)); ax.set_xticklabels(stages, fontsize=8.2, **FONT)
    ax.set_ylim(0, 100); ax.set_ylabel("per cent of answerable questions", fontsize=9, **FONT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / "fig3_results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    for f in sorted(OUT.glob("fig*.png")):
        print("wrote", f.name, f.stat().st_size // 1024, "KB")
