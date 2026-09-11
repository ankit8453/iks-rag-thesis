"""Build the 8-slide, 5-minute seminar deck (16:9 .pptx) + the phone-readable
speaker script (.docx). One SCRIPT list feeds both.

Logical slides: 1 title · 2 problem (+hallucination) · 3 six texts (compact, one real
verse) · 4 architecture (5 build-up frames = one animated diagram) · 5 the
contribution · 6 the interface · 7 trust · 8 why it matters + next. Plus a backup
slide after "Thank you" with the architecture as a looping GIF.

Physical slide count = 12 (+1 GIF backup): the architecture occupies slides 4-8 as
build-up frames; pressing -> lights up the next stage.

Usage:
    python seminar/build_deck.py               # build deck + script
    python seminar/build_deck.py --gif DIR     # DIR holds slide4..slide8 PNGs exported by
                                               # PowerPoint; builds the GIF and appends the
                                               # backup slide
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT_PPTX = ROOT / "seminar" / "ICSSR_Seminar_Deck_2026-09-11.pptx"
OUT_DOCX = ROOT / "seminar" / "SPEAKER_SCRIPT_2026-09-11.docx"
OUT_GIF = ROOT / "seminar" / "architecture_flow.gif"
IMG = ROOT / "demo_images"

GREEN = RGBColor(0x1F, 0x4D, 0x2E)
GREEN2 = RGBColor(0x2E, 0x6B, 0x41)
OCHRE = RGBColor(0xC8, 0x89, 0x1E)
RED = RGBColor(0xB0, 0x3A, 0x2E)
PURPLE = RGBColor(0x5A, 0x4A, 0x7A)
PARCH = RGBColor(0xF6, 0xF1, 0xE7)
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xFF, 0xFC, 0xF5)
LINE = RGBColor(0xDD, 0xD4, 0xC2)
SOFT = RGBColor(0xE9, 0xF0, 0xEA)
SOFT2 = RGBColor(0xFB, 0xEF, 0xD9)
CREAM = RGBColor(0xEC, 0xD9, 0xB0)
PALE = RGBColor(0xD9, 0xE4, 0xDB)
DIM = RGBColor(0xEB, 0xE8, 0xE0)      # inactive box fill
DIMTX = RGBColor(0xA8, 0xA3, 0x98)    # inactive text
DIMLN = RGBColor(0xD2, 0xCD, 0xC2)    # inactive arrow

SERIF, SANS = "Georgia", "Calibri"
W, H = Inches(13.333), Inches(7.5)

# --------------------------------------------------------------------------- the script
SCRIPT: list[tuple[str, str, str]] = [
    ("Title", "0:00-0:15",
     "Good evening, everyone. I'm Ankit Pawar from IIITDM Jabalpur; this work is with my "
     "supervisor, Dr. Akshay Pandey. In five minutes I'll show you a system that lets a farmer "
     "consult India's classical agricultural texts from a photograph of a sick plant - built so "
     "that the AI can never put words in those texts' mouths."),
    ("The problem", "0:15-1:00",
     "Here is the real situation. A farmer sees dark, scabby patches on apple leaves. Where does "
     "advice come from today? A chemical dealer - or an AI chatbot. And chatbots have a known "
     "flaw called hallucination: they answer confidently even when they don't know, and they "
     "invent facts and sources. For indigenous knowledge that few people can verify, that is "
     "dangerous. Meanwhile, six classical treatises hold centuries of organic remedies - but "
     "three locks keep them on the shelf: Sanskrit or rare translations; no searchable digital "
     "form; and they describe problems by symptom, never by a modern disease name."),
    ("The six texts", "1:00-1:25",
     "The six texts, very briefly. Vrikshayurveda - plant life and treating sick trees. Brihat "
     "Samhita - rain, finding water, crops, tree treatment. Krishi Parashara - the farming "
     "almanac. Upavanavinoda - garden and tree care. Kashyapiya Krishisukti - cultivation from "
     "land to harvest. Vishvavallabha - water, horticulture and plant disease, the one we add "
     "next. And here is one real line, Vrikshayurveda verse 193: to remove insects from the roots "
     "and branches, water the tree with cold water for seven days. Symptom, then remedy, numbered "
     "- that is what makes these texts searchable."),
    ("Architecture - one diagram, five clicks", "1:25-2:55",
     "[CLICK 1] The farmer sends a leaf photo and a soil photo, names the crop, and may add a "
     "suspected cause. All of it goes to the combiner - the hub of the system. "
     "[CLICK 2] The combiner passes the leaf to the disease model, which returns the disease name "
     "and a confidence - here, apple scab at 0.86 - and the soil photo to the soil model, which "
     "returns soil type, moisture and texture. "
     "[CLICK 3] Now the key step. The combiner writes one question in the language the texts use "
     "- not 'apple scab', but 'dark, rough, scabby patches spreading over the leaf of a tree in "
     "moist alluvial soil'. "
     "[CLICK 4] That question searches the knowledge base - the six texts cut into 327 numbered "
     "passages - and the five best-matching passages come back. "
     "[CLICK 5] The answer writer composes the advice using only those five passages and cites "
     "the verse - or refuses if nothing fits. The farmer receives the advice, the source verse, a "
     "heat-map of where the model looked, and the passages themselves."),
    ("The contribution - search by symptom, not by name", "2:55-3:35",
     "Why is that rewriting the contribution? Search the texts for 'apple scab' and you get "
     "almost nothing - relevance 0.01 to 0.04 - because the phrase does not exist in classical "
     "literature. Search for 'dark, rough, scabby patches spreading over the leaf' and relevance "
     "jumps to 0.59 to 0.96. The texts prescribe by symptom, not by crop. And because a symptom "
     "carries across plants, the system can advise - cautiously, with a confidence figure - even "
     "on a crop it was never trained for."),
    ("What the farmer sees", "3:35-4:10",
     "This is the interface. On the left, the inputs: leaf photo, soil photo, the crop, an "
     "optional suspected cause. On the right: the diagnosis with a confidence, a heat-map of the "
     "leaf region the model examined, the recommendation with its citation from Vrikshayurveda, "
     "and the retrieved passages themselves so anyone can check the source. If the texts are "
     "silent, this panel says so honestly."),
    ("Trust by design", "4:10-4:40",
     "Four guarantees make this safe for indigenous knowledge. It cites the verse. It refuses "
     "when the texts are silent - in our evaluation, one hundred percent honest refusal and zero "
     "fabricated citations. Its attention is verified - early on the heat-map showed the model "
     "reading the background of field photos, and we retrained until it reads the lesion. And "
     "uncertain cases go to an agricultural expert - a human in the loop."),
    ("Why it matters - and next", "4:40-5:00",
     "Why this matters for revitalising IKS: a verse that is retrieved and cited in a farmer's "
     "advisory is preserved, transmitted and validated at once - and nothing is distorted, because "
     "everything traces to its source. Next: validation with agronomists, adding Vishvavallabha, "
     "and an expert-built evaluation set. Thank you - happy to take questions."),
]

QA = [
    ("Does it replace the agricultural expert?",
     "No. It retrieves and cites; uncertain cases are routed to an agronomist. It is an access tool, not an authority."),
    ("How do you know the AI isn't making things up?",
     "It can only compose from passages it actually retrieved, must cite them, and a check confirms each citation was really retrieved. In evaluation: zero fabricated citations."),
    ("What if the texts don't cover the disease?",
     "It says so. Refusing is a feature. Coverage is the current limiter, so we are adding texts - Vishvavallabha next, which specifically covers plant disease."),
    ("Are the translations reliable?",
     "We use published scholarly translations (e.g. Asian Agri-History Foundation editions) and keep verse numbers, so any passage can be checked against the printed edition."),
    ("Which crops does it support?",
     "Trained on common crops; because the texts prescribe by symptom, it can advise cautiously on others, with a confidence figure and an explicit caution."),
    ("Can farmers use it today?",
     "The prototype is a web application. Farmer-facing deployment and field validation are the next phase."),
]

DO_NOT_SAY = [
    "Model or library names (EfficientNet, Llama, BM25...) - say 'image model', 'language model', 'search'.",
    "That all six texts are digitised - five are; Vishvavallabha is next.",
    "The over-refusal percentage. If asked: 'it refuses about half the time when the texts are thin - deliberately; coverage is the limiter.'",
    "That Sanskrit was machine-translated.",
    "Anything about the NITI manual beyond 'a modern IKS-derived manual is included, tiered separately'.",
]

ARCH_STEPS = [
    "Step 1 of 5 - the farmer sends two photos, the crop and a suspected cause",
    "Step 2 of 5 - two models read the leaf and the soil, and report back",
    "Step 3 of 5 - the combiner writes one question in the texts' own language",
    "Step 4 of 5 - the question searches 327 numbered passages; the 5 best come back",
    "Step 5 of 5 - advice is written only from those passages, cited, and returned with the heat-map",
]


# --------------------------------------------------------------------------- helpers
def _e(v):
    """OOXML needs integer EMUs; arithmetic like `w / 2` yields floats that PowerPoint rejects."""
    return int(round(v))


def _rect(sl, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None, lw=1.25):
    s = sl.shapes.add_shape(shape, _e(x), _e(y), _e(w), _e(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    return s


def _text(sl, x, y, w, h, text, *, size=18, bold=False, color=INK, font=SANS,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, ls=1.08):
    tb = sl.shapes.add_textbox(_e(x), _e(y), _e(w), _e(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.07)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, ln in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.line_spacing = align, ls
        r = p.add_run()
        r.text = ln
        r.font.name, r.font.size, r.font.bold, r.font.italic = font, Pt(size), bold, italic
        r.font.color.rgb = color
    return tb


def _bullets(sl, x, y, w, h, items, *, size=17, color=INK, gap=6):
    tb = sl.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.07)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "•  " + it
        r.font.name, r.font.size, r.font.color.rgb = SANS, Pt(size), color
    return tb


def _connector(sl, x1, y1, x2, y2, color, width=2.5, elbow=False):
    """Line with an arrowhead at the (x2, y2) end."""
    c = sl.shapes.add_connector(MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT,
                                _e(x1), _e(y1), _e(x2), _e(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = OxmlElement("a:tailEnd")
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    ln.append(tail)
    return c


def _base(prs, title, kicker, footer=True):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(sl, 0, 0, W, H, PARCH)
    _rect(sl, 0, 0, W, Inches(0.14), GREEN)
    _text(sl, Inches(0.5), Inches(0.28), Inches(9), Inches(0.35), kicker.upper(), size=12, bold=True, color=OCHRE)
    _text(sl, Inches(0.5), Inches(0.55), Inches(12.3), Inches(0.8), title, size=30, bold=True, color=GREEN, font=SERIF)
    if footer:
        _text(sl, Inches(0.5), Inches(7.08), Inches(9), Inches(0.3),
              "Ankit Pawar · Dr. Akshay Pandey  |  PDPM IIITDM Jabalpur", size=10, color=MUTED)
    return sl


def _notes(sl, i, extra=""):
    name, t, body = SCRIPT[i]
    sl.notes_slide.notes_text_frame.text = f"[{t}]  {extra}{body}"


def _square(path: Path, size: int = 600) -> Path:
    from PIL import Image  # noqa: PLC0415
    out = path.parent / f"_sq_{path.stem}.jpg"
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = min(w, h)
    im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((size, size)).save(out, quality=90)
    return out


def _card(sl, x, y, w, h, title, body, accent, *, tsize=16, bsize=14):
    _rect(sl, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(sl, x + Inches(0.15), y, w - Inches(0.3), Inches(0.09), accent)
    _text(sl, x + Inches(0.2), y + Inches(0.18), w - Inches(0.4), Inches(0.5), title, size=tsize, bold=True, color=GREEN, font=SERIF)
    _text(sl, x + Inches(0.2), y + Inches(0.75), w - Inches(0.4), h - Inches(0.9), body, size=bsize, color=INK, ls=1.12)


# --------------------------------------------------------------------------- architecture diagram
def _node(sl, x, y, w, h, title, lines, accent, active, *, shape=MSO_SHAPE.ROUNDED_RECTANGLE, tsize=14, lsize=11):
    fill = CARD if active else DIM
    tcol = GREEN if active else DIMTX
    bcol = INK if active else DIMTX
    _rect(sl, x, y, w, h, fill, shape, line=LINE if active else DIMLN)
    if shape != MSO_SHAPE.OVAL:
        _rect(sl, x + Inches(0.15), y, w - Inches(0.3), Inches(0.08), accent if active else DIMLN)
    _text(sl, x + Inches(0.05), y + (Inches(0.12) if shape != MSO_SHAPE.OVAL else Inches(0.35)), w - Inches(0.1), Inches(0.42),
          title, size=tsize, bold=True, color=tcol, align=PP_ALIGN.CENTER)
    if lines:
        _text(sl, x + Inches(0.08), y + (Inches(0.5) if shape != MSO_SHAPE.OVAL else Inches(0.8)), w - Inches(0.16), h - Inches(0.55),
              lines, size=lsize, color=bcol, align=PP_ALIGN.CENTER, ls=1.1)


def _label(sl, x, y, w, text, active, *, size=11, align=PP_ALIGN.CENTER):
    _text(sl, x, y, w, Inches(0.5), text, size=size, italic=True, bold=active,
          color=(OCHRE if active else DIMTX), align=align)


def _chip(sl, x, y, w, h, text, *, fill=SOFT2, color=INK, size=11, bold=True):
    _rect(sl, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE, line=OCHRE, lw=1.5)
    _text(sl, x + Inches(0.05), y, w - Inches(0.1), h, text, size=size, bold=bold, color=color,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def draw_architecture(sl, stage, leaf, soil):
    """One frame of the build-up. stage 1..5 controls what is lit."""
    on = lambda *k: stage in k  # noqa: E731

    F = (Inches(0.5), Inches(2.85), Inches(2.0), Inches(2.1))           # farmer
    HUB = (Inches(4.0), Inches(2.65), Inches(2.3), Inches(2.3))         # combiner (oval)
    DIS = (Inches(3.55), Inches(1.25), Inches(3.2), Inches(1.1))       # disease model
    SOL = (Inches(3.55), Inches(5.45), Inches(3.2), Inches(1.15))       # soil model
    KB = (Inches(9.0), Inches(1.25), Inches(3.8), Inches(1.25))         # knowledge base
    ANS = (Inches(9.0), Inches(3.05), Inches(3.8), Inches(1.5))         # answer writer
    EXP = (Inches(9.0), Inches(5.25), Inches(3.8), Inches(1.15))        # explain & deliver

    # ---- arrows first (nodes sit on top) ----
    a1 = on(1)
    _connector(sl, F[0] + F[2], F[1] + Inches(0.75), HUB[0], HUB[1] + Inches(0.75), OCHRE if a1 else DIMLN, 3 if a1 else 1.5)
    _label(sl, F[0] + F[2] - Inches(0.1), F[1] + Inches(0.05), Inches(1.7), "leaf + soil photo, crop, cause", a1)

    a2 = on(2)
    hx = HUB[0] + HUB[2] / 2
    _connector(sl, hx - Inches(0.25), HUB[1], hx - Inches(0.25), DIS[1] + DIS[3], OCHRE if a2 else DIMLN, 3 if a2 else 1.5)
    _connector(sl, hx + Inches(0.25), DIS[1] + DIS[3], hx + Inches(0.25), HUB[1], OCHRE if a2 else DIMLN, 3 if a2 else 1.5)
    _connector(sl, hx - Inches(0.25), HUB[1] + HUB[3], hx - Inches(0.25), SOL[1], OCHRE if a2 else DIMLN, 3 if a2 else 1.5)
    _connector(sl, hx + Inches(0.25), SOL[1], hx + Inches(0.25), HUB[1] + HUB[3], OCHRE if a2 else DIMLN, 3 if a2 else 1.5)
    _label(sl, hx + Inches(0.35), DIS[1] + DIS[3] + Inches(0.05), Inches(2.2), "disease · confidence", a2, align=PP_ALIGN.LEFT)
    _label(sl, hx + Inches(0.35), SOL[1] - Inches(0.45), Inches(2.4), "soil type · moisture · texture", a2, align=PP_ALIGN.LEFT)

    a3 = on(3)
    _connector(sl, HUB[0] + HUB[2] - Inches(0.15), HUB[1] + Inches(0.55), KB[0], KB[1] + KB[3] - Inches(0.25), OCHRE if a3 else DIMLN, 3 if a3 else 1.5)
    _label(sl, Inches(6.55), Inches(2.1), Inches(2.3), "symptom question", a3)

    a4 = on(4)
    _connector(sl, KB[0] + KB[2] / 2, KB[1] + KB[3], KB[0] + KB[2] / 2, ANS[1], OCHRE if a4 else DIMLN, 3 if a4 else 1.5)
    _label(sl, KB[0] + KB[2] / 2 + Inches(0.1), KB[1] + KB[3] + Inches(0.12), Inches(2.0), "5 best passages", a4, align=PP_ALIGN.LEFT)

    a5 = on(5)
    _connector(sl, ANS[0] + ANS[2] / 2, ANS[1] + ANS[3], ANS[0] + ANS[2] / 2, EXP[1], OCHRE if a5 else DIMLN, 3 if a5 else 1.5)
    _label(sl, ANS[0] + ANS[2] / 2 + Inches(0.1), ANS[1] + ANS[3] + Inches(0.12), Inches(2.4), "cited advice · or honest refusal", a5, align=PP_ALIGN.LEFT)
    _connector(sl, EXP[0], EXP[1] + EXP[3] - Inches(0.2), F[0] + F[2] / 2, F[1] + F[3], OCHRE if a5 else DIMLN, 3 if a5 else 1.5, elbow=True)
    _label(sl, Inches(0.5), Inches(6.62), Inches(3.2), "advice + verse + heat-map + passages", a5, align=PP_ALIGN.LEFT)

    # ---- nodes ----
    _node(sl, *F, "FARMER", ["leaf photo", "soil photo", "crop · suspected cause"], OCHRE, on(1, 5), tsize=15, lsize=12)
    if on(1, 5):
        sl.shapes.add_picture(str(leaf), F[0] + Inches(0.35), F[1] + Inches(1.3), width=Inches(0.6))
        sl.shapes.add_picture(str(soil), F[0] + Inches(1.05), F[1] + Inches(1.3), width=Inches(0.6))
    _node(sl, *HUB, "THE COMBINER", ["the hub: routes the photos,", "writes the question,", "delivers the answer"],
          OCHRE, on(1, 2, 3, 5), shape=MSO_SHAPE.OVAL, tsize=15, lsize=11)
    _node(sl, *DIS, "MODEL 1 · Plant disease", ["leaf photo → disease name, healthy/diseased, confidence, heat-map"], GREEN, on(2))
    _node(sl, *SOL, "MODEL 2 · Soil", ["soil photo → soil type, moisture, texture (visual only)"], GREEN, on(2))
    _node(sl, *KB, "KNOWLEDGE BASE", ["six classical texts → 327 numbered passages", "search by meaning + by keyword"], GREEN2, on(4))
    _node(sl, *ANS, "ANSWER WRITER", ["language model — may use ONLY the passages found;", "every line cites the verse; refuses if nothing fits"], RED, on(5))
    _node(sl, *EXP, "EXPLAIN & DELIVER", ["heat-map of the leaf region used", "+ the passages, so anyone can verify"], PURPLE, on(5))

    # ---- travelling payloads ----
    if stage == 1:
        sl.shapes.add_picture(str(leaf), Inches(2.85), Inches(3.95), width=Inches(0.5))
        sl.shapes.add_picture(str(soil), Inches(3.4), Inches(3.95), width=Inches(0.5))
    if stage == 2:
        _chip(sl, Inches(6.85), Inches(2.25), Inches(2.0), Inches(0.42), "apple scab · 0.86")
        _chip(sl, Inches(6.85), Inches(4.85), Inches(2.4), Inches(0.42), "alluvial · moist · medium")
    if stage == 3:
        _chip(sl, Inches(5.35), Inches(2.2), Inches(3.6), Inches(0.75),
              "“dark, rough, scabby patches spreading over the leaf of a tree in moist alluvial soil”", size=11)
    if stage == 4:
        _chip(sl, Inches(9.15), Inches(2.42), Inches(3.5), Inches(0.5),
              "Vrikshayurveda v.173, 191, 193, 194 · Upavanavinoda …", size=10)
    if stage == 5:
        _chip(sl, Inches(6.45), Inches(3.55), Inches(2.45), Inches(0.62),
              "“water with cold water for seven days…” [Vrikshayurveda v.193]", size=10)

    _rect(sl, Inches(3.9), Inches(6.98), Inches(0.12), Inches(0.28), OCHRE)
    _text(sl, Inches(4.1), Inches(6.93), Inches(8.7), Inches(0.4), ARCH_STEPS[stage - 1], size=13, bold=True, color=GREEN)


# --------------------------------------------------------------------------- deck
def build_deck() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    leaf = _square(IMG / "leaf_apple_scab.jpg")
    soil = _square(IMG / "soil_alluvial.jpg")

    # ===== 1. Title =====
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, GREEN)
    _rect(s, 0, Inches(5.85), W, Inches(0.07), OCHRE)
    _text(s, Inches(0.8), Inches(0.6), Inches(11.7), Inches(0.4),
          "International Seminar on Revitalising Indigenous Knowledge Systems  ·  ICSSR  ·  11–12 September 2026", size=13, color=PALE)
    _text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.4),
          "Sub-theme: Digital Preservation, Artificial Intelligence, and Documentation of Indigenous Knowledge", size=13, italic=True, color=PALE)
    _text(s, Inches(0.8), Inches(1.85), Inches(11.7), Inches(1.4), "An IKS-Grounded Multimodal Advisory System", size=38, bold=True, color=WHITE, font=SERIF)
    _text(s, Inches(0.8), Inches(3.05), Inches(11.7), Inches(1.0),
          "Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts", size=22, color=CREAM, font=SERIF)
    _text(s, Inches(0.8), Inches(4.45), Inches(11.5), Inches(0.5),
          "वृक्षायुर्वेद  ·  बृहत्संहिता  ·  कृषिपराशर  ·  उपवनविनोद  ·  काश्यपीयकृषिसूक्ति  ·  विश्ववल्लभ",
          size=17, color=OCHRE)
    _text(s, Inches(0.8), Inches(6.1), Inches(11.5), Inches(0.5), "Ankit Pawar   ·   Dr. Akshay Pandey (Supervisor)", size=21, bold=True, color=WHITE)
    _text(s, Inches(0.8), Inches(6.55), Inches(11.5), Inches(0.5),
          "Department of Computer Science & Engineering, PDPM Indian Institute of Information Technology, Design and Manufacturing, Jabalpur", size=13, color=PALE)
    _notes(s, 0)

    # ===== 2. The problem (+ hallucination) =====
    s = _base(prs, "Centuries of plant-care knowledge — locked on the shelf", "The problem")
    _rect(s, Inches(0.5), Inches(1.5), Inches(4.3), Inches(5.3), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    s.shapes.add_picture(str(leaf), Inches(0.75), Inches(1.75), width=Inches(1.5))
    _text(s, Inches(2.4), Inches(1.75), Inches(2.3), Inches(1.6), ["A farmer sees", "dark, scabby patches", "on apple leaves."], size=16, color=INK)
    _text(s, Inches(0.75), Inches(3.4), Inches(3.9), Inches(0.4), "Where does advice come from today?", size=15, bold=True, color=GREEN)
    _bullets(s, Inches(0.75), Inches(3.8), Inches(3.9), Inches(0.9), ["A chemical dealer", "An AI chatbot ↓"], size=14)
    _rect(s, Inches(0.75), Inches(4.65), Inches(3.8), Inches(2.0), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=OCHRE, lw=1.5)
    _text(s, Inches(0.9), Inches(4.72), Inches(3.5), Inches(0.35), "THE HALLUCINATION PROBLEM", size=11, bold=True, color=RED)
    _text(s, Inches(0.9), Inches(5.05), Inches(3.5), Inches(1.6),
          "Chatbots answer confidently even when they do not know — and invent facts and sources. "
          "For indigenous knowledge that few people can verify, that is dangerous.", size=13, color=INK, ls=1.12)
    _text(s, Inches(5.1), Inches(1.5), Inches(7.8), Inches(0.4), "Six classical treatises hold the knowledge…", size=16, bold=True, color=GREEN)
    books = [["Vriksha-", "yurveda"], ["Brihat", "Samhita"], ["Krishi", "Parashara"], ["Upavana-", "vinoda"], ["Kashyapiya", "Krishisukti"], ["Vishva-", "vallabha"]]
    tones = [GREEN, GREEN2, OCHRE, RGBColor(0x8C, 0x5A, 0x14), RGBColor(0x3B, 0x6B, 0x4A), RGBColor(0xA3, 0x72, 0x25)]
    bw, bh = Inches(1.22), Inches(1.5)
    for i, b in enumerate(books):
        x = Inches(5.1) + i * (bw + Inches(0.09))
        _rect(s, x, Inches(1.95), bw, bh, tones[i])
        _rect(s, x, Inches(2.1), bw, Inches(0.04), CREAM)
        _text(s, x, Inches(2.2), bw, bh - Inches(0.3), b, size=12, bold=True, color=WHITE, font=SERIF, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, Inches(5.1), Inches(3.6), Inches(7.8), Inches(0.4), "…but three locks keep it on the shelf", size=16, bold=True, color=RED)
    locks = [("1  Language", "Sanskrit, or rare scattered translations"),
             ("2  No digital form", "Nothing structured or searchable"),
             ("3  Symptom, not name", "The texts say “dark scabby patches spreading” — never “apple scab”. Modern labels do not exist in them.")]
    y = Inches(4.05)
    for t, sub in locks:
        _rect(s, Inches(5.1), y, Inches(7.8), Inches(0.85), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _rect(s, Inches(5.1), y + Inches(0.15), Inches(0.08), Inches(0.55), RED)
        _text(s, Inches(5.3), y + Inches(0.06), Inches(2.4), Inches(0.75), t, size=15, bold=True, color=GREEN, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, Inches(7.6), y + Inches(0.06), Inches(5.25), Inches(0.75), sub, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.93)
    _notes(s, 1)

    # ===== 3. The six texts (compact) + one real verse =====
    s = _base(prs, "The six texts — what each one contains", "The sources, in one line each")
    texts = [
        ("Vrikshayurveda", "Surapala, c. 10th c.", "The science of plant life — planting, nourishing, and treating sick trees", GREEN),
        ("Brihat Samhita", "Varahamihira, 6th c.", "Encyclopaedia — rain, finding underground water, crop growth, tree treatment", GREEN2),
        ("Krishi Parashara", "Parashara", "The farming almanac — rain forecasting, sowing calendar, seed storage", OCHRE),
        ("Upavanavinoda", "Sarngadhara", "Garden and orchard care — planting, grafting, watering, protection", RGBColor(0x8C, 0x5A, 0x14)),
        ("Kashyapiya Krishisukti", "Kashyapa, before 800 CE", "Cultivation from land to harvest — land, water, seeds, sowing, grain, vegetables", RGBColor(0x3B, 0x6B, 0x4A)),
        ("Vishvavallabha", "Chakrapani Mishra, 1577", "Water-finding, horticulture, plant disease and pests — being added next", RGBColor(0xA3, 0x72, 0x25)),
    ]
    y = Inches(1.5)
    for name, who, what, col in texts:
        _rect(s, Inches(0.5), y, Inches(12.3), Inches(0.62), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _rect(s, Inches(0.5), y + Inches(0.12), Inches(0.1), Inches(0.38), col)
        _text(s, Inches(0.75), y + Inches(0.02), Inches(2.9), Inches(0.58), name, size=16, bold=True, color=GREEN, font=SERIF, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, Inches(3.6), y + Inches(0.02), Inches(2.2), Inches(0.58), who, size=11, italic=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, Inches(5.8), y + Inches(0.02), Inches(6.9), Inches(0.58), what, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.7)
    _rect(s, Inches(0.5), Inches(5.85), Inches(12.3), Inches(1.05), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(0.7), Inches(5.9), Inches(3), Inches(0.3), "ONE REAL LINE, AS PUBLISHED", size=10, bold=True, color=RED)
    _text(s, Inches(0.7), Inches(6.15), Inches(11.9), Inches(0.7),
          "Vrikshayurveda v.193 — “To remove insects from the roots and branches of the trees, wise men should water the trees "
          "with cold water for seven days.”   → symptom, then remedy, numbered: that is what makes the texts searchable.",
          size=13, italic=True, color=INK, font=SERIF)
    _notes(s, 2)

    # ===== 4. Architecture: five build-up frames =====
    for stage in range(1, 6):
        s = _base(prs, "How it works — from photo to cited advice", "Architecture  ·  press → to advance", footer=False)
        draw_architecture(s, stage, leaf, soil)
        _notes(s, 3, extra=f"(frame {stage}/5)  ")

    # ===== 5. The contribution =====
    s = _base(prs, "The contribution: search by symptom, not by name", "Why the combiner is the key idea")
    lw = Inches(5.95)
    _rect(s, Inches(0.5), Inches(1.5), lw, Inches(3.15), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, Inches(0.65), Inches(1.5), lw - Inches(0.3), Inches(0.08), RED)
    _text(s, Inches(0.7), Inches(1.65), lw - Inches(0.4), Inches(0.4), "SEARCH WITH THE MODERN LABEL", size=12, bold=True, color=RED)
    _text(s, Inches(0.7), Inches(2.05), lw - Inches(0.4), Inches(0.6), "“Apple scab”", size=26, bold=True, color=INK, font=SERIF)
    _text(s, Inches(0.7), Inches(2.7), lw - Inches(0.4), Inches(0.5), "→ the texts return almost nothing. The phrase does not exist in them.", size=14, color=MUTED)
    _text(s, Inches(0.7), Inches(3.35), lw - Inches(0.4), Inches(0.9), "relevance  0.01 – 0.04", size=28, bold=True, color=RED)
    rx = Inches(0.5) + lw + Inches(0.4)
    _rect(s, rx, Inches(1.5), lw, Inches(3.15), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, rx + Inches(0.15), Inches(1.5), lw - Inches(0.3), Inches(0.08), GREEN)
    _text(s, rx + Inches(0.2), Inches(1.65), lw - Inches(0.4), Inches(0.4), "RE-DESCRIBE AS THE SYMPTOM", size=12, bold=True, color=GREEN)
    _text(s, rx + Inches(0.2), Inches(2.05), lw - Inches(0.4), Inches(0.9), "“dark, rough, scabby patches spreading over the leaf”", size=20, bold=True, color=INK, font=SERIF)
    _text(s, rx + Inches(0.2), Inches(2.85), lw - Inches(0.4), Inches(0.5), "→ strong matches — the texts speak this language.", size=14, color=MUTED)
    _text(s, rx + Inches(0.2), Inches(3.35), lw - Inches(0.4), Inches(0.9), "relevance  0.59 – 0.96", size=28, bold=True, color=GREEN)
    _rect(s, Inches(0.5), Inches(4.9), Inches(12.3), Inches(1.95), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(0.7), Inches(5.0), Inches(4.0), Inches(0.4), "How the rewriting is done", size=14, bold=True, color=GREEN)
    _text(s, Inches(0.7), Inches(5.35), Inches(5.6), Inches(1.4),
          "A language model is given the diagnosis and asked to describe the visible symptoms in plain words — the way a cultivator "
          "would describe them. The crop becomes context; the symptom leads.", size=13, color=INK)
    _text(s, Inches(6.7), Inches(5.0), Inches(6.0), Inches(0.4), "What this unlocks", size=14, bold=True, color=GREEN)
    _bullets(s, Inches(6.7), Inches(5.35), Inches(6.0), Inches(1.45), [
        "Searching by meaning beats searching by keywords: 0.94 vs 0.70",
        "A symptom carries across plants → cautious advice even for untrained crops, with a confidence figure",
        "Matches how the texts themselves prescribe — by symptom, not by crop",
    ], size=13, gap=4)
    _notes(s, 4)

    # ===== 6. Interface =====
    s = _base(prs, "What the farmer sees — the interface", "User interface")
    _rect(s, Inches(0.5), Inches(1.45), Inches(12.3), Inches(5.3), WHITE, line=RGBColor(0xC9, 0xC2, 0xB2))
    _rect(s, Inches(0.5), Inches(1.45), Inches(12.3), Inches(0.42), GREEN)
    _text(s, Inches(0.65), Inches(1.47), Inches(8), Inches(0.4), "IKS Agricultural Advisory  ·  prototype", size=13, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    lx, ly = Inches(0.7), Inches(2.05)
    _text(s, lx, ly, Inches(3.6), Inches(0.35), "INPUTS", size=11, bold=True, color=OCHRE)
    _text(s, lx, ly + Inches(0.3), Inches(1.8), Inches(0.3), "Leaf photo", size=12, bold=True, color=INK)
    s.shapes.add_picture(str(leaf), lx, ly + Inches(0.58), width=Inches(1.35))
    _text(s, lx + Inches(1.9), ly + Inches(0.3), Inches(1.8), Inches(0.3), "Soil photo", size=12, bold=True, color=INK)
    s.shapes.add_picture(str(soil), lx + Inches(1.9), ly + Inches(0.58), width=Inches(1.35))
    for i, (lab, val) in enumerate([("Crop", "Apple  ▾"), ("Suspected cause (optional)", "Not sure  ▾")]):
        yy = ly + Inches(2.15) + i * Inches(0.85)
        _text(s, lx, yy, Inches(3.6), Inches(0.3), lab, size=12, bold=True, color=INK)
        _rect(s, lx, yy + Inches(0.3), Inches(3.4), Inches(0.4), PARCH, line=LINE)
        _text(s, lx + Inches(0.08), yy + Inches(0.3), Inches(3.2), Inches(0.4), val, size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    _rect(s, lx, ly + Inches(4.0), Inches(3.4), Inches(0.5), GREEN, MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(s, lx, ly + Inches(4.0), Inches(3.4), Inches(0.5), "Get advice", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _rect(s, Inches(4.45), Inches(1.95), Inches(0.02), Inches(4.7), LINE)
    rx, ry = Inches(4.7), Inches(2.05)
    _text(s, rx, ry, Inches(8), Inches(0.35), "RESULT", size=11, bold=True, color=OCHRE)
    _rect(s, rx, ry + Inches(0.32), Inches(3.9), Inches(1.15), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(0.12), ry + Inches(0.37), Inches(3.7), Inches(0.35), "Diagnosis", size=11, bold=True, color=MUTED)
    _text(s, rx + Inches(0.12), ry + Inches(0.65), Inches(3.7), Inches(0.4), "Apple — scab   ·   confidence 0.86", size=14, bold=True, color=GREEN)
    _text(s, rx + Inches(0.12), ry + Inches(1.02), Inches(3.7), Inches(0.4), "Soil: alluvial · moist · medium texture", size=12, color=INK)
    _rect(s, rx + Inches(4.1), ry + Inches(0.32), Inches(3.9), Inches(1.15), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(4.22), ry + Inches(0.37), Inches(3.7), Inches(0.35), "Where the model looked", size=11, bold=True, color=MUTED)
    s.shapes.add_picture(str(leaf), rx + Inches(4.25), ry + Inches(0.68), width=Inches(0.75))
    _rect(s, rx + Inches(4.45), ry + Inches(0.8), Inches(0.35), Inches(0.35), RGBColor(0xE0, 0x4E, 0x2A), MSO_SHAPE.OVAL)
    _text(s, rx + Inches(5.1), ry + Inches(0.68), Inches(2.85), Inches(0.75), "heat-map on the lesion — not the background", size=11, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    _rect(s, rx, ry + Inches(1.6), Inches(8.0), Inches(1.75), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(0.12), ry + Inches(1.65), Inches(7.7), Inches(0.35), "Recommendation (from the texts only)", size=11, bold=True, color=MUTED)
    _text(s, rx + Inches(0.12), ry + Inches(1.95), Inches(7.7), Inches(1.4),
          ["Cut away the affected parts and sprinkle the spots with water mixed with milk; feed the roots with kunapa water and "
           "cow-dung paste. Water with cold water for seven days.",
           "Source: [Vrikshayurveda, v. 193–194]   ·   [Vrikshayurveda, v. 191]"], size=12, color=INK, ls=1.12)
    _rect(s, rx, ry + Inches(3.45), Inches(8.0), Inches(1.2), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(0.12), ry + Inches(3.5), Inches(7.7), Inches(0.35), "Retrieved passages (so anyone can verify)", size=11, bold=True, color=MUTED)
    _text(s, rx + Inches(0.12), ry + Inches(3.78), Inches(7.7), Inches(0.9),
          ["v.193  “To remove insects from the roots and branches … water the trees with cold water for seven days.”",
           "v.194  “The worms can be overcome by the paste of milk, kunapa water, and cow dung …”"], size=11, italic=True, color=INK, font=SERIF)
    _text(s, Inches(0.5), Inches(6.78), Inches(12.3), Inches(0.3),
          "Illustrative screen with real corpus passages. If the texts are silent, the result panel states: “The classical texts do not contain enough information for this case.”",
          size=10, italic=True, color=MUTED)
    _notes(s, 5)

    # ===== 7. Trust =====
    s = _base(prs, "Trust by design — four guarantees against fabrication", "Safety for indigenous knowledge")
    cw, ch, cy = Inches(2.95), Inches(3.3), Inches(1.5)
    cards = [
        ("Cites the verse", ["Every recommendation is composed only from retrieved passages.", "", "[Vrikshayurveda, v. 193]"], GREEN),
        ("Refuses when silent", ["If the texts do not cover a case, it says so instead of inventing.", "", "100% honest refusal", "0% fabricated citations"], OCHRE),
        ("Attention verified", ["The heat-map must sit on the lesion. Early on it sat on the background of field photos — we retrained until it did not."], GREEN2),
        ("Human in the loop", ["Uncertain or out-of-scope cases are collected for an agronomist to review.", "", "No silent self-learning from unverified guesses"], PURPLE),
    ]
    for i, (t, body, col) in enumerate(cards):
        _card(s, Inches(0.5) + i * (cw + Inches(0.17)), cy, cw, ch, t, body, col)
    ny = Inches(5.05)
    for i, (big, lab) in enumerate([("0.01–0.04 → 0.59–0.96", "relevance: modern label vs symptom rewriting"),
                                    ("100%  ·  0%", "honest refusal · fabricated citations"),
                                    ("0.94 vs 0.70", "search by meaning vs by keyword")]):
        x = Inches(0.5) + i * Inches(4.15)
        _rect(s, x, ny, Inches(4.0), Inches(1.15), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _text(s, x + Inches(0.1), ny + Inches(0.08), Inches(3.8), Inches(0.6), big, size=20, bold=True, color=OCHRE, font=SERIF, align=PP_ALIGN.CENTER)
        _text(s, x + Inches(0.1), ny + Inches(0.66), Inches(3.8), Inches(0.45), lab, size=12, color=INK, align=PP_ALIGN.CENTER)
    _text(s, Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.5),
          "For a plant the model was never trained on, it still advises — with a confidence figure and an explicit caution. Always “confidence”, never “accuracy”.",
          size=12, italic=True, color=GREEN)
    _notes(s, 6)

    # ===== 8. Why it matters + next + thanks =====
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, GREEN)
    _text(s, Inches(0.8), Inches(0.5), Inches(8), Inches(0.4), "WHY THIS MATTERS FOR REVITALISING IKS", size=12, bold=True, color=OCHRE)
    _text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(1.6), "A text that is cited every day is preserved — and alive.", size=38, bold=True, color=WHITE, font=SERIF)
    _bullets(s, Inches(0.8), Inches(2.7), Inches(11.5), Inches(1.9), [
        "Documentation and use reinforce each other — retrieval is preservation in practice",
        "Verifiable, not distorted — every answer traces to its source verse",
        "In the farmer’s hands, not only in archives — a template for AI on Indigenous knowledge beyond agriculture",
    ], size=18, color=WHITE, gap=8)
    _rect(s, Inches(0.8), Inches(4.65), Inches(11.7), Inches(0.05), OCHRE)
    _text(s, Inches(0.8), Inches(4.8), Inches(8), Inches(0.4), "NEXT STEPS", size=12, bold=True, color=OCHRE)
    _bullets(s, Inches(0.8), Inches(5.15), Inches(11.5), Inches(1.2),
             ["Validation with agronomists   ·   add Vishvavallabha   ·   expert-built evaluation set"], size=16, color=WHITE)
    _text(s, Inches(0.8), Inches(6.25), Inches(11.5), Inches(0.6), "Thank you", size=32, bold=True, color=WHITE, font=SERIF)
    _text(s, Inches(0.8), Inches(6.85), Inches(11.5), Inches(0.4), "Ankit Pawar  ·  Dr. Akshay Pandey  ·  PDPM IIITDM Jabalpur", size=14, color=CREAM)
    _notes(s, 7)

    prs.save(OUT_PPTX)
    for p in (leaf, soil):
        p.unlink(missing_ok=True)
    return OUT_PPTX


# --------------------------------------------------------------------------- GIF + backup slide
def add_gif_slide(png_dir: Path) -> Path:
    """Build architecture_flow.gif from the five exported frames and append a backup slide."""
    from PIL import Image  # noqa: PLC0415

    frames = [Image.open(png_dir / f"slide{i}.png").convert("RGB") for i in range(4, 9)]
    wpx, hpx = frames[0].size
    box = (int(wpx * 0.03), int(hpx * 0.17), int(wpx * 0.97), int(hpx * 0.985))
    frames = [f.crop(box).resize((1400, int(1400 * (box[3] - box[1]) / (box[2] - box[0])))) for f in frames]
    seq, durs = [], []
    for i, f in enumerate(frames):
        seq.append(f); durs.append(1600)
        if i < len(frames) - 1:
            for a in (0.33, 0.66):
                seq.append(Image.blend(f, frames[i + 1], a)); durs.append(90)
    seq[0].save(OUT_GIF, save_all=True, append_images=seq[1:], duration=durs, loop=0, optimize=False)

    prs = Presentation(OUT_PPTX)
    s = _base(prs, "Backup — the flow as a loop", "Appendix  ·  plays automatically in slideshow", footer=False)
    gw = Inches(12.3)
    gh = _e(gw * (seq[0].height / seq[0].width))
    s.shapes.add_picture(str(OUT_GIF), Inches(0.5), Inches(1.45), width=gw, height=gh)
    s.notes_slide.notes_text_frame.text = ("Backup slide. The same five stages loop automatically. Use during questions if someone "
                                           "asks to see the flow again.")
    prs.save(OUT_PPTX)
    return OUT_GIF


# --------------------------------------------------------------------------- speaker script (.docx)
def build_script() -> Path:
    from docx import Document
    from docx.shared import Pt as DPt

    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", DPt(14)

    d.add_heading("Speaker script — ICSSR seminar, 11 Sep 2026 (5 minutes)", level=1)
    d.add_paragraph().add_run(
        "Read the paragraph for each slide while it is shown. The bold line under each paragraph is the cue to press →. "
        "Slide 4 is ONE diagram over five frames: press → at each [CLICK]. Total ≈ 5:00. "
        "If the moderator is cutting time, shorten slide 3 (skip the verse) and slide 6 — never cut slide 4, 5 or 7."
    ).italic = True

    for i, (name, t, body) in enumerate(SCRIPT, 1):
        d.add_heading(f"Slide {i} — {name}   [{t}]", level=2)
        if "[CLICK" in body:
            for part in body.split("[CLICK"):
                part = part.strip()
                if not part:
                    continue
                n, _, txt = part.partition("]")
                p = d.add_paragraph()
                p.add_run(f"→ press  (frame {n.strip()})   ").bold = True
                p.add_run(txt.strip())
        else:
            d.add_paragraph(body)
        r = d.add_paragraph().add_run("→ next slide" if i < len(SCRIPT) else "→ stop, smile, wait for questions")
        r.bold = True
        r.font.size = DPt(12)

    d.add_heading("If they ask — short answers", level=1)
    for q, a in QA:
        p = d.add_paragraph()
        p.add_run(q + "  ").bold = True
        p.add_run(a)
    d.add_heading("Do not say", level=1)
    for x in DO_NOT_SAY:
        d.add_paragraph(x, style="List Bullet")
    for cand in [OUT_DOCX] + [OUT_DOCX.with_name(f"{OUT_DOCX.stem}_v{n}{OUT_DOCX.suffix}") for n in range(2, 10)]:
        try:
            d.save(cand)
            if cand is not OUT_DOCX:
                print(f"  ! {OUT_DOCX.name} is open in Word - saved as {cand.name} instead")
            return cand
        except PermissionError:
            continue
    raise PermissionError(f"close {OUT_DOCX.name} in Word and re-run")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--gif":
        g = add_gif_slide(Path(sys.argv[2]))
        print(f"gif    : {g.relative_to(ROOT)}  ({g.stat().st_size // 1024} KB) + backup slide appended")
    else:
        pp = build_deck()
        dx = build_script()
        prs = Presentation(pp)
        words = sum(len(b.replace("[CLICK", "").split()) for _, _, b in SCRIPT)
        print(f"deck   : {pp.relative_to(ROOT)}  ({len(prs.slides)} physical slides = 8 logical; notes on all="
              f"{all(sl.notes_slide.notes_text_frame.text.strip() for sl in prs.slides)})")
        print(f"script : {dx.relative_to(ROOT)}  ({words} words ≈ {words / 145:.1f} min at 145 wpm)")
