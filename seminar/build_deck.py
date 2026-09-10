"""Build the 8-slide, 5-minute seminar deck (16:9 .pptx) from PRESENTATION_BRIEF.md.

ICSSR International Seminar on Revitalising IKS, online session 11 Sep 2026.
Every slide carries its speaker notes. Real leaf/soil photos from demo_images/.
Slides 5 and 6 leave a clearly-marked spot for the Colab screenshots (Grad-CAM
before/after; app result page) — drop them in later if captured.

Usage:  python seminar/build_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "seminar" / "ICSSR_Seminar_Deck_2026-09-11.pptx"
IMG = ROOT / "demo_images"

# ---- palette (brief §10) ----
GREEN = RGBColor(0x1F, 0x4D, 0x2E)
OCHRE = RGBColor(0xC8, 0x89, 0x1E)
PARCH = RGBColor(0xF6, 0xF1, 0xE7)
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xFF, 0xFC, 0xF5)

SERIF = "Georgia"
SANS = "Calibri"

W, H = Inches(13.333), Inches(7.5)


# --------------------------------------------------------------------------- helpers
def _rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def _text(slide, x, y, w, h, text, *, size=24, bold=False, color=INK, font=SANS,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, line_spacing=1.1):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def _bullets(slide, x, y, w, h, items, *, size=22, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "•  " + it
        r.font.name = SANS
        r.font.size = Pt(size)
        r.font.color.rgb = color
    return tb


def _base(prs, title=None, kicker=None):
    """Parchment slide with a green top band; optional kicker + title."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    _rect(slide, 0, 0, W, H, PARCH)
    _rect(slide, 0, 0, W, Inches(0.16), GREEN)
    if kicker:
        _text(slide, Inches(0.6), Inches(0.35), Inches(9), Inches(0.4), kicker.upper(),
              size=13, bold=True, color=OCHRE)
    if title:
        _text(slide, Inches(0.6), Inches(0.68), Inches(12.1), Inches(1.0), title,
              size=36, bold=True, color=GREEN, font=SERIF)
    # footer
    _text(slide, Inches(0.6), Inches(7.05), Inches(9), Inches(0.35),
          "Ankit Pawar · Dr. Akshay Pandey  |  PDPM IIITDM Jabalpur", size=11, color=MUTED)
    return slide


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def _card(slide, x, y, w, h, heading, body, *, accent=GREEN, body_size=18):
    _rect(slide, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=RGBColor(0xE3, 0xDB, 0xCB))
    _rect(slide, x + Inches(0.25), y, w - Inches(0.5), Inches(0.1), accent, MSO_SHAPE.RECTANGLE)
    _text(slide, x + Inches(0.3), y + Inches(0.2), w - Inches(0.5), Inches(0.7), heading,
          size=22, bold=True, color=GREEN, font=SERIF)
    _text(slide, x + Inches(0.3), y + Inches(0.95), w - Inches(0.5), h - Inches(1.1), body,
          size=body_size, color=INK)


def _stat(slide, x, y, w, h, number, label):
    _rect(slide, x, y, w, h, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=RGBColor(0xE3, 0xDB, 0xCB))
    _text(slide, x, y + Inches(0.25), w, Inches(1.1), number, size=48, bold=True,
          color=OCHRE, font=SERIF, align=PP_ALIGN.CENTER)
    _text(slide, x + Inches(0.2), y + Inches(1.4), w - Inches(0.4), h - Inches(1.5), label,
          size=17, color=INK, align=PP_ALIGN.CENTER)


def _arrow(slide, x, y, w=Inches(0.55), h=Inches(0.5)):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.fill.solid()
    a.fill.fore_color.rgb = OCHRE
    a.line.fill.background()
    return a


def _placeholder(slide, x, y, w, h, label):
    """Dashed drop-zone for a screenshot to be added later."""
    s = _rect(slide, x, y, w, h, RGBColor(0xEF, 0xEA, 0xDD), MSO_SHAPE.RECTANGLE,
              line=RGBColor(0xBD, 0xB3, 0x9E))
    _text(slide, x, y, w, h, label, size=14, italic=True, color=MUTED,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return s


def _square(path: Path, size: int = 600) -> Path:
    """Centre-crop a photo to a square thumbnail (the soil photo is a wide panorama)."""
    from PIL import Image  # noqa: PLC0415
    out = path.parent / f"_sq_{path.stem}.jpg"
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((size, size))
    im.save(out, quality=90)
    return out


# --------------------------------------------------------------------------- slides
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------- 1. Title ----------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, GREEN)
    _rect(s, 0, Inches(5.9), W, Inches(0.08), OCHRE)
    _text(s, Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.5),
          "International Seminar on Revitalising Indigenous Knowledge Systems  ·  ICSSR  ·  11–12 September 2026",
          size=14, color=RGBColor(0xD9, 0xE4, 0xDB))
    _text(s, Inches(0.8), Inches(1.05), Inches(11.5), Inches(0.5),
          "Sub-theme: Digital Preservation, Artificial Intelligence, and Documentation of Indigenous Knowledge",
          size=14, italic=True, color=RGBColor(0xD9, 0xE4, 0xDB))
    _text(s, Inches(0.8), Inches(1.9), Inches(11.7), Inches(2.2),
          ["An IKS-Grounded Multimodal Advisory System",
           "Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts"],
          size=36, bold=True, color=WHITE, font=SERIF, line_spacing=1.15)
    # second line smaller: rewrite runs
    tf = s.shapes[-1].text_frame
    tf.paragraphs[1].runs[0].font.size = Pt(24)
    tf.paragraphs[1].runs[0].font.bold = False
    tf.paragraphs[1].runs[0].font.color.rgb = RGBColor(0xEC, 0xD9, 0xB0)
    _text(s, Inches(0.8), Inches(4.5), Inches(11), Inches(0.5),
          "वृक्षायुर्वेद  ·  बृहत्संहिता  ·  कृषिपराशर  ·  उपवनविनोद  ·  काश्यपीयकृषिसूक्ति",
          size=18, color=OCHRE)
    _text(s, Inches(0.8), Inches(6.15), Inches(11), Inches(0.5),
          "Ankit Pawar   ·   Dr. Akshay Pandey (Supervisor)", size=22, bold=True, color=WHITE)
    _text(s, Inches(0.8), Inches(6.6), Inches(11), Inches(0.5),
          "Department of Computer Science & Engineering, PDPM Indian Institute of Information "
          "Technology, Design and Manufacturing, Jabalpur", size=14, color=RGBColor(0xD9, 0xE4, 0xDB))
    _notes(s, "Good evening. I'm Ankit Pawar from IIITDM Jabalpur. In five minutes I'll show you "
              "how we made India's classical agricultural texts answerable from a photograph — "
              "without letting AI put words in their mouth.  [0:00–0:15]")

    # ---------------- 2. Heritage on the shelf ----------------
    s = _base(prs, "Documented — but dead on the shelf", "The problem")
    books = ["Vrikshayurveda", "Brihat Samhita", "Krishi Parashara",
             "Upavanavinoda", "Kashyapiya Krishisukti", "Vishvavallabha"]
    x0, y0, bw, bh, gap = Inches(0.6), Inches(1.95), Inches(1.95), Inches(2.3), Inches(0.08)
    tones = [GREEN, RGBColor(0x2E, 0x5E, 0x3D), OCHRE, RGBColor(0x8C, 0x5A, 0x14),
             RGBColor(0x3B, 0x6B, 0x4A), RGBColor(0xA3, 0x72, 0x25)]
    for i, b in enumerate(books):
        x = x0 + i * (bw + gap)
        _rect(s, x, y0, bw, bh, tones[i])
        _rect(s, x, y0 + Inches(0.18), bw, Inches(0.05), RGBColor(0xEC, 0xD9, 0xB0))
        _text(s, x, y0 + Inches(0.5), bw, bh - Inches(0.6), b, size=15, bold=True,
              color=WHITE, font=SERIF, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, Inches(0.6), Inches(4.55), Inches(12), Inches(0.45),
          "Three locks keep this knowledge on the shelf:", size=20, bold=True, color=GREEN)
    cw = Inches(3.9)
    _card(s, Inches(0.6), Inches(5.1), cw, Inches(1.75), "Sanskrit",
          "Survives in Sanskrit and rare, scattered translations.", body_size=16)
    _card(s, Inches(0.6) + cw + Inches(0.2), Inches(5.1), cw, Inches(1.75), "Scattered editions",
          "No structured, searchable digital form.", body_size=16, accent=OCHRE)
    _card(s, Inches(0.6) + 2 * (cw + Inches(0.2)), Inches(5.1), cw, Inches(1.75),
          "Symptom, not name",
          "“Dark scabby patches spreading” — never “apple scab”.", body_size=16)
    _notes(s, "These six treatises describe organic remedies, soil selection, seed care. But a farmer "
              "or extension worker can't consult them: they're in Sanskrit or rare translations, and "
              "crucially they don't use disease names. Vrikshayurveda never says 'apple scab'. It says "
              "'dark scabby patches spreading on the leaf'. So the knowledge is documented — and dead "
              "on the shelf.  [0:15–1:00]")

    # ---------------- 3. The idea in one picture ----------------
    s = _base(prs, "Make the texts consultable from a photograph", "The idea")
    steps = [
        ("1  Photograph", "Affected leaf + the soil", None),
        ("2  Diagnose", "Condition and visible soil properties", None),
        ("3  Rephrase", "…as the symptoms the texts use", None),
        ("4  Advise", "Only from the texts, cited to the verse", None),
    ]
    sx, sy, sw, sh = Inches(0.6), Inches(2.0), Inches(2.75), Inches(3.9)
    for i, (h1, body, _) in enumerate(steps):
        x = sx + i * (sw + Inches(0.42))
        accent = OCHRE if i == 2 else GREEN
        _rect(s, x, sy, sw, sh, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=RGBColor(0xE3, 0xDB, 0xCB))
        _rect(s, x, sy, sw, Inches(0.12), accent)
        _text(s, x + Inches(0.2), sy + Inches(0.25), sw - Inches(0.4), Inches(0.6), h1,
              size=22, bold=True, color=GREEN, font=SERIF)
        _text(s, x + Inches(0.2), sy + Inches(0.85), sw - Inches(0.4), Inches(0.9), body,
              size=17, color=INK)
        if i < 3:
            _arrow(s, x + sw + Inches(-0.05), sy + Inches(1.65))
    # real photos in step 1
    leaf = IMG / "leaf_apple_scab.jpg"
    soil = IMG / "soil_alluvial.jpg"
    if leaf.is_file():
        s.shapes.add_picture(str(_square(leaf)), sx + Inches(0.2), sy + Inches(1.9), width=Inches(1.15))
    if soil.is_file():
        s.shapes.add_picture(str(_square(soil)), sx + Inches(1.45), sy + Inches(1.9), width=Inches(1.15))
    # step 2 text
    _text(s, sx + (sw + Inches(0.42)) + Inches(0.2), sy + Inches(1.95), sw - Inches(0.4), Inches(1.6),
          ["e.g.  Apple — scab", "soil: alluvial, moist"], size=17, italic=True, color=MUTED)
    # step 3 example
    _text(s, sx + 2 * (sw + Inches(0.42)) + Inches(0.2), sy + Inches(1.95), sw - Inches(0.4), Inches(1.8),
          "“dark, rough, scabby patches spreading over the leaf”", size=17, italic=True, color=OCHRE)
    # step 4 example
    _text(s, sx + 3 * (sw + Inches(0.42)) + Inches(0.2), sy + Inches(1.95), sw - Inches(0.4), Inches(1.8),
          ["Recommendation…", "[Vrikshayurveda, ch. X, v. Y]", "— or: “the texts do not cover this.”"],
          size=15, italic=True, color=MUTED)
    _text(s, Inches(0.6), Inches(6.2), Inches(12), Inches(0.5),
          "The system never answers from its own memory — only from what it retrieved.",
          size=18, italic=True, color=GREEN)
    _notes(s, "A farmer photographs the affected plant and the soil. The system identifies the "
              "condition and the soil's visible properties. Then — the key step — it translates that "
              "diagnosis into the language the texts actually use, searches them, and composes a "
              "recommendation drawn only from what it found, citing text, chapter and verse.  [1:00–1:45]")

    # ---------------- 4. Key insight ----------------
    s = _base(prs, "Speak the texts’ language: symptom, not crop", "The key insight (our contribution)")
    lw = Inches(5.9)
    _card(s, Inches(0.6), Inches(1.95), lw, Inches(3.5), "Search with the modern label",
          "", accent=RGBColor(0xB0, 0x3A, 0x2E))
    _text(s, Inches(0.9), Inches(2.85), lw - Inches(0.6), Inches(0.7), "“Apple scab”",
          size=26, bold=True, color=INK, font=SERIF)
    _text(s, Inches(0.9), Inches(3.55), lw - Inches(0.6), Inches(0.5), "→ the texts return almost nothing",
          size=18, color=MUTED)
    _text(s, Inches(0.9), Inches(4.2), lw - Inches(0.6), Inches(0.9), "relevance 0.01 – 0.04",
          size=30, bold=True, color=RGBColor(0xB0, 0x3A, 0x2E))
    _card(s, Inches(0.6) + lw + Inches(0.4), Inches(1.95), lw, Inches(3.5),
          "Re-describe as the symptom", "", accent=OCHRE)
    _text(s, Inches(0.9) + lw + Inches(0.4), Inches(2.85), lw - Inches(0.6), Inches(0.9),
          "“dark, rough, scabby patches spreading over the leaf”", size=22, bold=True, color=INK, font=SERIF)
    _text(s, Inches(0.9) + lw + Inches(0.4), Inches(3.75), lw - Inches(0.6), Inches(0.5),
          "→ strong matches in the texts", size=18, color=MUTED)
    _text(s, Inches(0.9) + lw + Inches(0.4), Inches(4.2), lw - Inches(0.6), Inches(0.9),
          "relevance 0.59 – 0.96", size=30, bold=True, color=GREEN)
    _text(s, Inches(0.6), Inches(5.75), Inches(12.1), Inches(0.9),
          "Because the symptom carries across plants, the system can also advise — cautiously, with a "
          "confidence figure — on crops it was never trained for.", size=18, italic=True, color=GREEN)
    _notes(s, "This is the contribution. Modern labels don't exist in classical texts, so direct search "
              "fails almost completely. Re-expressing the diagnosis as symptoms — the way the texts "
              "themselves reason — lifts retrieval from near zero to strong matches. It also means the "
              "system can help with a crop it was never trained on, because the symptom carries across "
              "plants — and it says so, with a confidence figure and a caution.  [1:45–2:30]")

    # ---------------- 5. Trust by design ----------------
    s = _base(prs, "Three guarantees against fabrication", "Trust by design")
    cw = Inches(3.95)
    _card(s, Inches(0.6), Inches(1.95), cw, Inches(3.3), "Cites the verse",
          ["Every recommendation is composed only from retrieved passages.",
           "", "e.g.  [Vrikshayurveda, ch. 3, v. 12]"], body_size=17)
    _card(s, Inches(0.6) + cw + Inches(0.15), Inches(1.95), cw, Inches(3.3), "Refuses when silent",
          ["If the texts don’t cover a case, it says so instead of inventing.",
           "", "100% honest refusal", "0% fabricated citations"], body_size=17, accent=OCHRE)
    _card(s, Inches(0.6) + 2 * (cw + Inches(0.15)), Inches(1.95), cw, Inches(3.3),
          "Looks at the lesion",
          ["The image model is checked visually — it must read the leaf, not the background.",
           "", "Explainability as a check, not decoration."], body_size=17)
    _text(s, Inches(0.6), Inches(5.55), Inches(12.1), Inches(1.2),
          "Early on, the heat-map showed the model reading the background of field photos. We retrained until it reads the lesion — and only then trusted it.",
          size=19, italic=True, color=GREEN)
    _notes(s, "Generative AI's danger for indigenous knowledge is confident fabrication. So we constrain "
              "it: it may only speak from retrieved passages, and must cite them. If the texts don't "
              "cover a case, it refuses — in evaluation, 100% honest refusal and zero fabricated "
              "citations. And the image model is verified visually: early on the heat-map showed it "
              "reading the background of field photos; we retrained until it reads the lesion. "
              "Explainability as a check, not decoration.  [2:30–3:20]")

    # ---------------- 6. What exists today ----------------
    s = _base(prs, "A working prototype and a real corpus", "What exists today")
    tw, th, ty = Inches(2.85), Inches(2.4), Inches(1.95)
    _stat(s, Inches(0.6), ty, tw, th, "327", "verse-level passages, searchable and citable")
    _stat(s, Inches(0.6) + (tw + Inches(0.23)), ty, tw, th, "5", "classical texts digitised\n(Vishvavallabha next)")
    _stat(s, Inches(0.6) + 2 * (tw + Inches(0.23)), ty, tw, th, "0%", "fabricated citations in evaluation")
    _stat(s, Inches(0.6) + 3 * (tw + Inches(0.23)), ty, tw, th, "✓", "end-to-end prototype:\nphoto → cited advice")
    _card(s, Inches(0.6), Inches(4.7), Inches(12.1), Inches(2.1), "Human in the loop",
          "Uncertain cases are collected for an agricultural expert to review — it improves under "
          "oversight, never by learning from unverified guesses.", body_size=16, accent=OCHRE)
    _notes(s, "This isn't a proposal. The corpus exists — 327 searchable, citable passages. The "
              "prototype runs end to end. When the system is unsure, the case is collected for an "
              "agricultural expert to review, so it improves under human oversight rather than learning "
              "from unverified guesses.  [3:20–4:00]")

    # ---------------- 7. Why it matters ----------------
    s = _base(prs, None, "Why this matters for revitalising IKS")
    _text(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.0),
          "A text that is cited every day is preserved — and alive.",
          size=44, bold=True, color=GREEN, font=SERIF, line_spacing=1.1)
    _rect(s, Inches(0.8), Inches(3.55), Inches(1.6), Inches(0.08), OCHRE)
    _bullets(s, Inches(0.8), Inches(3.9), Inches(11.5), Inches(2.9), [
        "Documentation and use reinforce each other — retrieval is preservation in practice",
        "Verifiable, not distorted — every answer traces to its source verse",
        "In the farmer’s hands, not only in archives",
        "A template for AI on Indigenous knowledge beyond agriculture",
    ], size=22)
    _notes(s, "For this seminar's theme: preservation and use are not separate goals. A verse that is "
              "retrieved and cited in a farmer's advisory is being preserved, transmitted and validated "
              "at once. And because every answer traces to its source, the AI transmits the tradition "
              "without distorting it. We think this is a template for Indigenous knowledge beyond "
              "agriculture.  [4:00–4:40]")

    # ---------------- 8. Next steps + thanks ----------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, GREEN)
    _rect(s, 0, Inches(4.55), W, Inches(0.08), OCHRE)
    _text(s, Inches(0.8), Inches(0.5), Inches(8), Inches(0.4), "NEXT STEPS", size=13, bold=True, color=OCHRE)
    _bullets(s, Inches(0.8), Inches(1.0), Inches(11.5), Inches(3.3), [
        "Validation of recommendations with agronomists",
        "Add Vishvavallabha — the treatise on plant disease and pest management",
        "Expert-built evaluation set with domain scholars",
    ], size=24, color=WHITE, gap=14)
    _text(s, Inches(0.8), Inches(4.85), Inches(11.5), Inches(1.0), "Thank you", size=44, bold=True,
          color=WHITE, font=SERIF)
    _text(s, Inches(0.8), Inches(5.85), Inches(11.5), Inches(0.5),
          "Ankit Pawar  ·  Dr. Akshay Pandey  ·  PDPM IIITDM Jabalpur", size=20, color=RGBColor(0xEC, 0xD9, 0xB0))
    _text(s, Inches(0.8), Inches(6.35), Inches(11.5), Inches(0.5),
          "Questions welcome", size=16, italic=True, color=RGBColor(0xD9, 0xE4, 0xDB))
    _notes(s, "Next: validation with agronomists, adding Vishvavallabha which specifically covers plant "
              "disease, and an expert-built evaluation set. Thank you — happy to take questions.  [4:40–5:00]")

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    p = Presentation(out)
    print(f"wrote {out.relative_to(ROOT)}  |  slides={len(p.slides)}  |  "
          f"notes on all={all(sl.has_notes_slide and sl.notes_slide.notes_text_frame.text.strip() for sl in p.slides)}")
