"""Build the 11-slide, 5-minute seminar deck (16:9 .pptx) + the phone-readable
speaker script (.docx). One SCRIPT list feeds both, so slide notes and the Word
script never drift apart.

ICSSR International Seminar on Revitalising IKS, online session 11 Sep 2026, 7-9 PM.

Slides: 1 title · 2 the problem · 3 the six texts (what each contains) · 4 what the
texts actually say (real verses) · 5 end-to-end flow · 6 two models -> combiner ->
database -> answer · 7 symptom-based search (the contribution) · 8 the interface ·
9 trust guarantees · 10 results · 11 why it matters + next steps.

Plain language throughout: no model/library names on the slides.

Usage:  python seminar/build_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT_PPTX = ROOT / "seminar" / "ICSSR_Seminar_Deck_2026-09-11.pptx"
OUT_DOCX = ROOT / "seminar" / "SPEAKER_SCRIPT_2026-09-11.docx"
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

SERIF, SANS = "Georgia", "Calibri"
W, H = Inches(13.333), Inches(7.5)

# --------------------------------------------------------------------------- the script
# (slide name, timing, what to say)
SCRIPT: list[tuple[str, str, str]] = [
    ("Title", "0:00-0:15",
     "Good evening, everyone. I'm Ankit Pawar from IIITDM Jabalpur; this work is with my "
     "supervisor, Dr. Akshay Pandey. In five minutes I'll show you a system that lets a farmer "
     "consult India's classical agricultural texts from a photograph of a sick plant - built so "
     "that the AI can never put words in those texts' mouths."),
    ("The problem", "0:15-0:50",
     "Here is the real situation. A farmer sees dark, scabby patches on apple leaves. The advice "
     "available today comes from a chemical dealer or a generic chatbot - which may invent facts. "
     "Meanwhile six classical treatises hold centuries of organic remedies. Three locks keep them "
     "on the shelf: they survive in Sanskrit or rare translations; there is no searchable digital "
     "form; and they describe problems by symptom, never by a modern disease name."),
    ("The six texts", "0:50-1:20",
     "A word on the six texts, because each covers something different. Vrikshayurveda is the "
     "science of plant life - planting, nourishing and treating sick trees. Brihat Samhita is an "
     "encyclopaedia with chapters on rain, finding water, crop growth and treating trees. Krishi "
     "Parashara is a farming almanac - rain forecasting, the sowing calendar, seed storage. "
     "Upavanavinoda is garden and tree care - planting, grafting, watering. Kashyapiya "
     "Krishisukti is cultivation - choosing land, water, seeds, growing grain and vegetables. "
     "And Vishvavallabha covers water-finding, horticulture and plant disease - the one we add next."),
    ("What the texts actually say", "1:20-1:45",
     "Let me show you what is inside. This is Vrikshayurveda, verses 173 and 190 to 194, in the "
     "published translation. Notice the structure: a symptom - roots eaten by insects, leaves "
     "turning yellow and pale - and then a remedy - water the tree with cold water for seven days; "
     "a paste of milk, kunapa water and cow dung. Symptom, then remedy, verse-numbered. That is what "
     "makes these texts searchable - if we ask them in their own language."),
    ("How the system works - end to end", "1:45-2:20",
     "Here is the whole system in one flow. The farmer uploads a leaf photo and a soil photo, names "
     "the crop, and may add a suspected cause. The leaf is cut out of the background. One model "
     "identifies the disease; a second reads the soil. The diagnosis is rewritten as the symptoms "
     "the texts describe, and that searches our digital corpus of the texts. A language model then "
     "writes the advice using only the passages found, citing chapter and verse - or says plainly "
     "that the texts do not cover the case. The farmer also sees why: a heat-map of where the model "
     "looked, and the passages used."),
    ("Two models, one question, one database", "2:20-2:55",
     "Structurally it is two models, a combiner, and a database. Model one looks at the leaf photo "
     "and gives the disease name, healthy or diseased, and a confidence. Model two looks at the soil "
     "photo and gives the soil type, how moist it is, and its texture. The combiner takes both, plus "
     "the crop and the suspected cause, and turns them into one question written in the texts' own "
     "language - the symptom description. That question searches the database: the six texts cut "
     "into 327 numbered passages. The best passages go to the answer writer, which may use only "
     "those passages, must cite the verse, and otherwise refuses."),
    ("The contribution - search by symptom, not by name", "2:55-3:25",
     "Why is the combiner the contribution? Search the texts for 'apple scab' and you get almost "
     "nothing - relevance 0.01 to 0.04 - because that phrase does not exist in classical literature. "
     "Search instead for 'dark, rough, scabby patches spreading over the leaf' and relevance jumps to "
     "0.59 to 0.96. The texts prescribe by symptom, not by crop. And because a symptom carries across "
     "plants, the system can advise - cautiously, with a confidence figure - even on a crop it was "
     "never trained for."),
    ("What the farmer sees", "3:25-3:50",
     "This is the interface. On the left, the inputs: leaf photo, soil photo, the crop, an optional "
     "suspected cause. On the right, the result: the diagnosis with a confidence, a heat-map of the "
     "leaf region the model examined, the recommendation with its citation from Vrikshayurveda, and "
     "the retrieved passages themselves, so anyone can check the source. If the texts are silent, "
     "this panel says so honestly."),
    ("Trust by design", "3:50-4:20",
     "Four guarantees make this safe for indigenous knowledge. It cites the verse. It refuses when "
     "the texts are silent - in evaluation, one hundred percent honest refusal and zero fabricated "
     "citations. Its attention is verified: early on, the heat-map showed the model reading the "
     "background of field photos, and we retrained until it reads the lesion. And uncertain cases go "
     "to an agricultural expert for review - a human in the loop."),
    ("What exists today", "4:20-4:40",
     "This is a working prototype, not a proposal: a corpus of 327 passages from five digitised "
     "texts, with Vishvavallabha next. Searching by meaning clearly beats searching by keywords. The "
     "disease model separates healthy from diseased almost perfectly; the soil model reads soil type "
     "and moisture at about ninety and ninety-six percent. All numbers are preliminary until expert "
     "validation."),
    ("Why it matters - and next", "4:40-5:00",
     "Why this matters for revitalising IKS: a verse that is retrieved and cited in a farmer's "
     "advisory is preserved, transmitted and validated at once - and nothing is distorted, because "
     "everything traces to its source. Next: validation with agronomists, adding Vishvavallabha, and "
     "an expert-built evaluation set. Thank you - happy to take questions."),
]

QA = [
    ("Does it replace the agricultural expert?",
     "No. It retrieves and cites; uncertain cases are routed to an agronomist. It is an access "
     "tool, not an authority."),
    ("How do you know the AI isn't making things up?",
     "It can only compose from passages it actually retrieved, must cite them, and a check "
     "confirms each citation was really retrieved. In evaluation: zero fabricated citations."),
    ("What if the texts don't cover the disease?",
     "It says so. Refusing is a feature, not a failure. Coverage is the current limiter, so we "
     "are adding texts - Vishvavallabha next, which specifically covers plant disease."),
    ("Are the translations reliable?",
     "We use published scholarly translations (e.g. Asian Agri-History Foundation editions) and "
     "keep verse numbers, so any passage can be checked against the printed edition."),
    ("Which crops does it support?",
     "Trained on common crops; because the texts prescribe by symptom, it can advise cautiously "
     "on others, with a confidence figure and an explicit caution."),
    ("Can farmers actually use it today?",
     "The prototype is a web application. Farmer-facing deployment and field validation are the "
     "next phase."),
]

DO_NOT_SAY = [
    "Model or library names (EfficientNet, Llama, BM25...) - say 'image model', 'language model', 'search'.",
    "That all six texts are digitised - five are; Vishvavallabha is next.",
    "The over-refusal percentage. If asked: 'it refuses about half the time when the texts are thin - deliberately; coverage is the limiter.'",
    "That Sanskrit was machine-translated.",
    "Anything about the NITI manual experiment beyond 'a modern IKS-derived manual is included, tiered separately'.",
]


# --------------------------------------------------------------------------- helpers
def _rect(sl, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None, lw=1.25):
    s = sl.shapes.add_shape(shape, x, y, w, h)
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
    tb = sl.shapes.add_textbox(x, y, w, h)
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


def _box(sl, x, y, w, h, title, sub=None, *, fill=CARD, accent=GREEN, tsize=15, ssize=12):
    """Diagram node: rounded card, coloured top strip, title + optional sub-line."""
    _rect(sl, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(sl, x + Inches(0.15), y, w - Inches(0.3), Inches(0.08), accent)
    _text(sl, x + Inches(0.05), y + Inches(0.14), w - Inches(0.1),
          Inches(0.5) if sub else h - Inches(0.2), title, size=tsize, bold=True, color=GREEN,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP if sub else MSO_ANCHOR.MIDDLE)
    if sub:
        _text(sl, x + Inches(0.05), y + Inches(0.6), w - Inches(0.1), h - Inches(0.65), sub,
              size=ssize, color=INK, align=PP_ALIGN.CENTER)


def _arrow(sl, x, y, w=Inches(0.42), h=Inches(0.36), color=OCHRE, down=False, left=False):
    a = sl.shapes.add_shape(MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    if left:
        a.rotation = 180
    return a


def _base(prs, title, kicker):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(sl, 0, 0, W, H, PARCH)
    _rect(sl, 0, 0, W, Inches(0.14), GREEN)
    _text(sl, Inches(0.5), Inches(0.28), Inches(9), Inches(0.35), kicker.upper(), size=12,
          bold=True, color=OCHRE)
    _text(sl, Inches(0.5), Inches(0.55), Inches(12.3), Inches(0.8), title, size=30, bold=True,
          color=GREEN, font=SERIF)
    _text(sl, Inches(0.5), Inches(7.08), Inches(9), Inches(0.3),
          "Ankit Pawar · Dr. Akshay Pandey  |  PDPM IIITDM Jabalpur", size=10, color=MUTED)
    return sl


def _notes(sl, i):
    name, t, body = SCRIPT[i]
    sl.notes_slide.notes_text_frame.text = f"[{t}]  {body}"


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
    _text(sl, x + Inches(0.2), y + Inches(0.18), w - Inches(0.4), Inches(0.5), title, size=tsize,
          bold=True, color=GREEN, font=SERIF)
    _text(sl, x + Inches(0.2), y + Inches(0.75), w - Inches(0.4), h - Inches(0.9), body, size=bsize,
          color=INK, ls=1.12)


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
          "International Seminar on Revitalising Indigenous Knowledge Systems  ·  ICSSR  ·  11–12 September 2026",
          size=13, color=PALE)
    _text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.4),
          "Sub-theme: Digital Preservation, Artificial Intelligence, and Documentation of Indigenous Knowledge",
          size=13, italic=True, color=PALE)
    _text(s, Inches(0.8), Inches(1.85), Inches(11.7), Inches(1.4),
          "An IKS-Grounded Multimodal Advisory System", size=38, bold=True, color=WHITE, font=SERIF)
    _text(s, Inches(0.8), Inches(3.05), Inches(11.7), Inches(1.0),
          "Connecting Image-Based Plant Diagnosis with Classical Indian Agricultural Texts",
          size=22, color=CREAM, font=SERIF)
    _text(s, Inches(0.8), Inches(4.45), Inches(11.5), Inches(0.5),
          "वृक्षायुर्वेद  ·  बृहत्संहिता  ·  कृषिपराशर  ·  उपवनविनोद  ·  काश्यपीयकृषिसूक्ति  ·  विश्ववल्लभ",
          size=17, color=OCHRE)
    _text(s, Inches(0.8), Inches(6.1), Inches(11.5), Inches(0.5),
          "Ankit Pawar   ·   Dr. Akshay Pandey (Supervisor)", size=21, bold=True, color=WHITE)
    _text(s, Inches(0.8), Inches(6.55), Inches(11.5), Inches(0.5),
          "Department of Computer Science & Engineering, PDPM Indian Institute of Information "
          "Technology, Design and Manufacturing, Jabalpur", size=13, color=PALE)
    _notes(s, 0)

    # ===== 2. The problem =====
    s = _base(prs, "Centuries of plant-care knowledge — locked on the shelf", "The problem")
    _rect(s, Inches(0.5), Inches(1.5), Inches(4.3), Inches(5.3), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    s.shapes.add_picture(str(leaf), Inches(0.75), Inches(1.75), width=Inches(1.6))
    _text(s, Inches(2.5), Inches(1.75), Inches(2.2), Inches(1.7),
          ["A farmer sees", "dark, scabby patches", "on apple leaves."], size=16, color=INK)
    _text(s, Inches(0.75), Inches(3.55), Inches(3.9), Inches(0.5),
          "Where does advice come from today?", size=15, bold=True, color=GREEN)
    _bullets(s, Inches(0.75), Inches(4.0), Inches(3.9), Inches(2.7), [
        "A chemical dealer",
        "A generic chatbot — which may invent facts and cite sources that do not exist",
        "Not from the classical texts that describe exactly this problem",
    ], size=14)
    _text(s, Inches(5.1), Inches(1.5), Inches(7.8), Inches(0.4),
          "Six classical treatises hold the knowledge…", size=16, bold=True, color=GREEN)
    books = [["Vriksha-", "yurveda"], ["Brihat", "Samhita"], ["Krishi", "Parashara"],
             ["Upavana-", "vinoda"], ["Kashyapiya", "Krishisukti"], ["Vishva-", "vallabha"]]
    tones = [GREEN, GREEN2, OCHRE, RGBColor(0x8C, 0x5A, 0x14), RGBColor(0x3B, 0x6B, 0x4A), RGBColor(0xA3, 0x72, 0x25)]
    bw, bh = Inches(1.22), Inches(1.5)
    for i, b in enumerate(books):
        x = Inches(5.1) + i * (bw + Inches(0.09))
        _rect(s, x, Inches(1.95), bw, bh, tones[i])
        _rect(s, x, Inches(2.1), bw, Inches(0.04), CREAM)
        _text(s, x, Inches(2.2), bw, bh - Inches(0.3), b, size=12, bold=True, color=WHITE,
              font=SERIF, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, Inches(5.1), Inches(3.6), Inches(7.8), Inches(0.4),
          "…but three locks keep it on the shelf", size=16, bold=True, color=RED)
    locks = [("1  Language", "Sanskrit, or rare scattered translations"),
             ("2  No digital form", "Nothing structured or searchable"),
             ("3  Symptom, not disease name",
              "The texts say “dark scabby patches spreading” — never “apple scab”. Modern labels do not exist in them.")]
    y = Inches(4.05)
    for t, sub in locks:
        _rect(s, Inches(5.1), y, Inches(7.8), Inches(0.85), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _rect(s, Inches(5.1), y + Inches(0.15), Inches(0.08), Inches(0.55), RED)
        _text(s, Inches(5.3), y + Inches(0.06), Inches(2.6), Inches(0.75), t, size=15, bold=True, color=GREEN,
              anchor=MSO_ANCHOR.MIDDLE)
        _text(s, Inches(7.75), y + Inches(0.06), Inches(5.1), Inches(0.75), sub, size=13, color=INK,
              anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.93)
    _notes(s, 1)

    # ===== 3. The six texts =====
    s = _base(prs, "The six texts — and what each one actually contains", "The sources, in plain words")
    texts = [
        ("Vrikshayurveda", "Surapala, c. 10th century",
         "The science of plant life: how to choose land, plant and nourish trees, and treat sick ones — "
         "includes recipes such as kunapa water (fermented manure).", GREEN),
        ("Brihat Samhita", "Varahamihira, 6th century",
         "An encyclopaedia. We use its chapters on rain, finding underground water, growth of crops, "
         "and treatment of trees.", GREEN2),
        ("Krishi Parashara", "Parashara",
         "A practical farming almanac: forecasting rain, the sowing calendar, storing seed, care of cattle.", OCHRE),
        ("Upavanavinoda", "Sarngadhara",
         "Garden and orchard care: planting, grafting, watering, and protecting trees from harm.",
         RGBColor(0x8C, 0x5A, 0x14)),
        ("Kashyapiya Krishisukti", "Kashyapa, before 800 CE",
         "Cultivation from the ground up: choosing land, water and irrigation, seeds, sowing, growing "
         "grain and vegetables.", RGBColor(0x3B, 0x6B, 0x4A)),
        ("Vishvavallabha", "Chakrapani Mishra, 1577",
         "Finding water, harvesting rain, horticulture — and plant disease and pest management. "
         "Being added next.", RGBColor(0xA3, 0x72, 0x25)),
    ]
    cw, ch = Inches(4.0), Inches(2.45)
    for i, (name, who, what, col) in enumerate(texts):
        x = Inches(0.5) + (i % 3) * (cw + Inches(0.15))
        y = Inches(1.5) + (i // 3) * (ch + Inches(0.18))
        _rect(s, x, y, cw, ch, CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _rect(s, x + Inches(0.15), y, cw - Inches(0.3), Inches(0.09), col)
        _text(s, x + Inches(0.2), y + Inches(0.15), cw - Inches(0.4), Inches(0.45), name, size=19, bold=True, color=GREEN, font=SERIF)
        _text(s, x + Inches(0.2), y + Inches(0.58), cw - Inches(0.4), Inches(0.35), who, size=12, italic=True, color=MUTED)
        _text(s, x + Inches(0.2), y + Inches(0.95), cw - Inches(0.4), ch - Inches(1.05), what, size=14, color=INK, ls=1.12)
    _text(s, Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.3),
          "Five of the six are digitised in our corpus today; Vishvavallabha is the next addition.", size=11, italic=True, color=MUTED)
    _notes(s, 2)

    # ===== 4. What the texts actually say =====
    s = _base(prs, "Inside a text: symptom, then remedy, verse by verse",
              "Inside the corpus · Vrikshayurveda of Surapala (tr. Sadhale, AAHF 1996)")
    _rect(s, Inches(0.5), Inches(1.5), Inches(6.05), Inches(5.35), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(0.7), Inches(1.6), Inches(5.7), Inches(0.4), "THE SYMPTOM", size=12, bold=True, color=RED)
    _text(s, Inches(0.7), Inches(1.95), Inches(5.7), Inches(1.7),
          "173.  “If the trees are exposed to scorching heat, the roots are eaten away by insects "
          "and result in drying, yellowness, and excessive paleness of leaves.”",
          size=16, color=INK, font=SERIF, italic=True)
    _text(s, Inches(0.7), Inches(3.75), Inches(5.7), Inches(0.4), "THE REMEDY", size=12, bold=True, color=GREEN)
    _text(s, Inches(0.7), Inches(4.1), Inches(5.7), Inches(2.7),
          ["193.  “To remove insects from the roots and branches, water the trees with cold water for seven days.”",
           "194.  “The worms can be overcome by the paste of milk, kunapa water, and cow dung…”",
           "191.  “Watered by the decoction of milk, honey, yastimadhu and madhuka, trees suffering "
           "from pitta-type diseases get cured.”"],
          size=14, color=INK, font=SERIF, italic=True, ls=1.15)
    _text(s, Inches(6.9), Inches(1.5), Inches(6.0), Inches(0.5), "Why this structure matters",
          size=18, bold=True, color=GREEN, font=SERIF)
    _bullets(s, Inches(6.9), Inches(2.05), Inches(6.0), Inches(3.0), [
        "Every remedy is attached to an observed symptom — not to a crop, not to a disease name",
        "Every verse is numbered → every answer can be cited and checked in the printed edition",
        "Ingredients are ordinary farm materials: milk, cow dung, ash, honey, kunapa (fermented manure)",
        "The same symptom recurs across plants → the knowledge transfers across crops",
    ], size=15, gap=9)
    _rect(s, Inches(6.9), Inches(5.35), Inches(6.0), Inches(1.5), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(7.1), Inches(5.45), Inches(5.6), Inches(1.35),
          ["So the question is not “does the knowledge exist?”",
           "It is: how do we ask the texts in their own language — from a photograph?"],
          size=15, bold=True, color=GREEN, anchor=MSO_ANCHOR.MIDDLE)
    _notes(s, 3)

    # ===== 5. End-to-end flow =====
    s = _base(prs, "From a photograph to a cited recommendation", "End-to-end flow")
    r1y, nw, nh, gap = Inches(1.55), Inches(2.1), Inches(1.55), Inches(0.35)
    x = Inches(0.5)
    _box(s, x, r1y, nw, nh, "1  Farmer inputs", "leaf photo · soil photo · crop · suspected cause (optional)", accent=OCHRE)
    s.shapes.add_picture(str(leaf), x + Inches(0.55), r1y + Inches(1.05), width=Inches(0.45))
    s.shapes.add_picture(str(soil), x + Inches(1.15), r1y + Inches(1.05), width=Inches(0.45))
    _arrow(s, x + nw + Inches(0.02), r1y + Inches(0.6)); x += nw + gap
    _box(s, x, r1y, nw, nh, "2  Locate the leaf", "cut the leaf out of the background so the model cannot cheat")
    _arrow(s, x + nw + Inches(0.02), r1y + Inches(0.6)); x += nw + gap
    _box(s, x, r1y, nw, nh, "3  Diagnose", "disease model → “apple scab”\nsoil model → type · moisture · texture")
    _arrow(s, x + nw + Inches(0.02), r1y + Inches(0.6)); x += nw + gap
    _box(s, x, r1y, nw, nh, "4  Rephrase", "“dark, rough, scabby patches spreading over the leaf”", accent=OCHRE, fill=SOFT2)
    _arrow(s, x + nw + Inches(0.02), r1y + Inches(0.6)); x += nw + gap
    _box(s, x, r1y, nw, nh, "5  Search the texts", "327 verse-level passages\nsearch by meaning + by keyword")
    _arrow(s, x + nw / 2 - Inches(0.18), r1y + nh + Inches(0.12), w=Inches(0.36), h=Inches(0.42), down=True)
    r2y = r1y + nh + Inches(0.7)
    _box(s, x, r2y, nw, nh, "6  Write the advice", "language model uses ONLY the retrieved passages; cites chapter & verse")
    _arrow(s, x - Inches(0.44), r2y + Inches(0.6), left=True); x -= nw + gap
    _box(s, x, r2y, nw, nh, "7  Check & decide", "every citation verified against what was retrieved\n→ answer, or honest refusal", accent=RED)
    _arrow(s, x - Inches(0.44), r2y + Inches(0.6), left=True); x -= nw + gap
    _box(s, x, r2y, nw, nh, "8  Explain", "heat-map of the leaf region used + the passages themselves")
    _arrow(s, x - Inches(0.44), r2y + Inches(0.6), left=True); x -= nw + gap
    _box(s, x, r2y, nw, nh, "9  Farmer receives", "recommendation + source verse — or “the texts do not cover this”", accent=OCHRE)
    _arrow(s, x - Inches(0.44), r2y + Inches(0.6), left=True); x -= nw + gap
    _box(s, x, r2y, nw, nh, "10  Learn safely", "uncertain cases collected for an agricultural expert to review", fill=SOFT)
    _text(s, Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.6),
          "The language model never answers from its own memory — every sentence must come from a retrieved passage.",
          size=15, italic=True, color=GREEN)
    _notes(s, 4)

    # ===== 6. Two models -> combiner -> database -> answer =====
    s = _base(prs, "Two models, one combiner, one database", "System structure")
    cx, cw = Inches(0.5), Inches(3.1)
    _rect(s, cx, Inches(1.5), cw, Inches(2.35), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, cx + Inches(0.15), Inches(1.5), cw - Inches(0.3), Inches(0.08), GREEN)
    _text(s, cx + Inches(0.1), Inches(1.6), cw - Inches(0.2), Inches(0.4), "MODEL 1 · Plant disease", size=15, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _text(s, cx + Inches(0.15), Inches(1.95), cw - Inches(0.3), Inches(0.35), "Input: the leaf photo", size=12, italic=True, color=MUTED)
    _bullets(s, cx + Inches(0.1), Inches(2.25), cw - Inches(0.2), Inches(1.6), [
        "Disease name  → “apple scab”", "Healthy or diseased", "Confidence  → 0.86", "Where it looked (heat-map)"], size=13, gap=3)
    _rect(s, cx, Inches(4.05), cw, Inches(2.35), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, cx + Inches(0.15), Inches(4.05), cw - Inches(0.3), Inches(0.08), OCHRE)
    _text(s, cx + Inches(0.1), Inches(4.15), cw - Inches(0.2), Inches(0.4), "MODEL 2 · Soil", size=15, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _text(s, cx + Inches(0.15), Inches(4.5), cw - Inches(0.3), Inches(0.35), "Input: the soil photo", size=12, italic=True, color=MUTED)
    _bullets(s, cx + Inches(0.1), Inches(4.8), cw - Inches(0.2), Inches(1.6), [
        "Soil type  → alluvial", "Moisture  → moist", "Texture  → medium", "(visual only — no chemical claims)"], size=13, gap=3)
    _text(s, cx, Inches(6.5), cw, Inches(0.4), "+ crop name and suspected cause from the farmer", size=12, italic=True, color=GREEN)
    _arrow(s, cx + cw + Inches(0.05), Inches(2.5), w=Inches(0.45), h=Inches(0.36))
    _arrow(s, cx + cw + Inches(0.05), Inches(5.05), w=Inches(0.45), h=Inches(0.36))
    kx, kw = cx + cw + Inches(0.5), Inches(2.85)
    _rect(s, kx, Inches(1.5), kw, Inches(4.9), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, kx + Inches(0.15), Inches(1.5), kw - Inches(0.3), Inches(0.09), OCHRE)
    _text(s, kx + Inches(0.1), Inches(1.65), kw - Inches(0.2), Inches(0.5), "THE COMBINER", size=15, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _text(s, kx + Inches(0.15), Inches(2.15), kw - Inches(0.3), Inches(1.2),
          "Takes both results + crop + cause and writes ONE question in the language the texts use:", size=13, color=INK)
    _rect(s, kx + Inches(0.2), Inches(3.4), kw - Inches(0.4), Inches(1.55), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, kx + Inches(0.3), Inches(3.45), kw - Inches(0.6), Inches(1.45),
          "“dark, rough, scabby patches spreading over the leaf of a tree growing in moist alluvial soil”",
          size=13, italic=True, bold=True, color=OCHRE, font=SERIF, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, kx + Inches(0.15), Inches(5.1), kw - Inches(0.3), Inches(1.2),
          "Symptom leads; crop and soil are context. This is the key idea of the whole system.", size=12, italic=True, color=GREEN)
    _arrow(s, kx + kw + Inches(0.05), Inches(3.75), w=Inches(0.45), h=Inches(0.36))
    dx, dw = kx + kw + Inches(0.5), Inches(2.5)
    _rect(s, dx, Inches(1.5), dw, Inches(4.9), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, dx + Inches(0.15), Inches(1.5), dw - Inches(0.3), Inches(0.09), GREEN2)
    _text(s, dx + Inches(0.1), Inches(1.65), dw - Inches(0.2), Inches(0.5), "THE DATABASE", size=15, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _text(s, dx + Inches(0.15), Inches(2.15), dw - Inches(0.3), Inches(1.1),
          "The six texts, cut into 327 numbered passages — each tagged with book, chapter and verse.", size=13, color=INK)
    for i, b in enumerate(["Vrikshayurveda", "Brihat Samhita", "Krishi Parashara", "Upavanavinoda", "Kashyapiya", "Vishvavallabha → next"]):
        _rect(s, dx + Inches(0.2), Inches(3.3) + i * Inches(0.36), dw - Inches(0.4), Inches(0.3), CARD, line=LINE, lw=0.5)
        _text(s, dx + Inches(0.25), Inches(3.3) + i * Inches(0.36), dw - Inches(0.5), Inches(0.3), b, size=11, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, dx + Inches(0.15), Inches(5.55), dw - Inches(0.3), Inches(0.8),
          "Search by meaning + by keyword → the 5 best-matching passages", size=12, italic=True, color=GREEN)
    _arrow(s, dx + dw + Inches(0.05), Inches(3.75), w=Inches(0.45), h=Inches(0.36))
    ax = dx + dw + Inches(0.5)
    aw = Inches(12.8) - ax
    _rect(s, ax, Inches(1.5), aw, Inches(4.9), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _rect(s, ax + Inches(0.15), Inches(1.5), aw - Inches(0.3), Inches(0.09), RED)
    _text(s, ax + Inches(0.1), Inches(1.65), aw - Inches(0.2), Inches(0.5), "THE ANSWER", size=15, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    _bullets(s, ax + Inches(0.1), Inches(2.2), aw - Inches(0.2), Inches(3.4), [
        "Written only from those 5 passages",
        "Every line cites the verse",
        "Citations checked against what was found",
        "If nothing fits → “the texts do not cover this”",
        "Shown to the farmer with the heat-map and the passages",
    ], size=13, gap=5)
    _notes(s, 5)

    # ===== 7. The contribution: symptom-based search =====
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
    _text(s, rx + Inches(0.2), Inches(2.05), lw - Inches(0.4), Inches(0.9),
          "“dark, rough, scabby patches spreading over the leaf”", size=20, bold=True, color=INK, font=SERIF)
    _text(s, rx + Inches(0.2), Inches(2.85), lw - Inches(0.4), Inches(0.5), "→ strong matches — the texts speak this language.", size=14, color=MUTED)
    _text(s, rx + Inches(0.2), Inches(3.35), lw - Inches(0.4), Inches(0.9), "relevance  0.59 – 0.96", size=28, bold=True, color=GREEN)
    _rect(s, Inches(0.5), Inches(4.9), Inches(12.3), Inches(1.95), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(0.7), Inches(5.0), Inches(4.0), Inches(0.4), "How the rewriting is done", size=14, bold=True, color=GREEN)
    _text(s, Inches(0.7), Inches(5.35), Inches(5.6), Inches(1.4),
          "A language model is given the diagnosis and asked to describe the visible symptoms in plain, "
          "descriptive words — the way a cultivator would describe them. The crop name becomes context; the symptom leads.",
          size=13, color=INK)
    _text(s, Inches(6.7), Inches(5.0), Inches(6.0), Inches(0.4), "What this unlocks", size=14, bold=True, color=GREEN)
    _bullets(s, Inches(6.7), Inches(5.35), Inches(6.0), Inches(1.45), [
        "Searching by meaning beats searching by keywords: 0.94 vs 0.70",
        "A symptom carries across plants → cautious advice even for untrained crops, with a confidence figure",
        "Matches how the texts themselves prescribe — by symptom, not by crop",
    ], size=13, gap=4)
    _notes(s, 6)

    # ===== 8. Interface mock =====
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
    _rect(s, Inches(4.45), Inches(1.95), Inches(0.02), Inches(4.9), LINE)
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
    _text(s, rx + Inches(5.1), ry + Inches(0.68), Inches(2.85), Inches(0.75),
          "heat-map on the lesion — not the background", size=11, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    _rect(s, rx, ry + Inches(1.6), Inches(8.0), Inches(1.75), SOFT2, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(0.12), ry + Inches(1.65), Inches(7.7), Inches(0.35), "Recommendation (from the texts only)", size=11, bold=True, color=MUTED)
    _text(s, rx + Inches(0.12), ry + Inches(1.95), Inches(7.7), Inches(1.4),
          ["Cut away the affected parts and sprinkle the spots with water mixed with milk; feed the roots "
           "with kunapa water and cow-dung paste. Water with cold water for seven days.",
           "Source: [Vrikshayurveda, v. 193–194]   ·   [Vrikshayurveda, v. 191]"],
          size=12, color=INK, ls=1.12)
    _rect(s, rx, ry + Inches(3.45), Inches(8.0), Inches(1.35), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, rx + Inches(0.12), ry + Inches(3.5), Inches(7.7), Inches(0.35), "Retrieved passages (so anyone can verify)", size=11, bold=True, color=MUTED)
    _text(s, rx + Inches(0.12), ry + Inches(3.8), Inches(7.7), Inches(1.0),
          ["v.193  “To remove insects from the roots and branches … water the trees with cold water for seven days.”",
           "v.194  “The worms can be overcome by the paste of milk, kunapa water, and cow dung …”"],
          size=11, italic=True, color=INK, font=SERIF)
    _text(s, Inches(0.5), Inches(6.78), Inches(12.3), Inches(0.3),
          "Illustrative screen with real corpus passages. If the texts are silent, the result panel states: “The classical texts do not contain enough information for this case.”",
          size=10, italic=True, color=MUTED)
    _notes(s, 7)

    # ===== 9. Trust by design =====
    s = _base(prs, "Trust by design — four guarantees against fabrication", "Safety for indigenous knowledge")
    cw, ch, cy = Inches(2.95), Inches(3.55), Inches(1.5)
    cards = [
        ("Cites the verse", ["Every recommendation is composed only from retrieved passages.", "", "[Vrikshayurveda, v. 193]"], GREEN),
        ("Refuses when silent", ["If the texts do not cover a case, it says so instead of inventing.", "", "100% honest refusal", "0% fabricated citations"], OCHRE),
        ("Attention verified", ["The heat-map must sit on the lesion. Early on it sat on the background of field photos — we retrained until it did not.", "", "Explainability as a check, not decoration"], GREEN2),
        ("Human in the loop", ["Uncertain or out-of-scope cases are collected for an agronomist to review.", "", "No silent self-learning from unverified guesses"], PURPLE),
    ]
    for i, (t, body, col) in enumerate(cards):
        x = Inches(0.5) + i * (cw + Inches(0.17))
        _card(s, x, cy, cw, ch, t, body, col)
    _rect(s, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.5), SOFT, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
    _text(s, Inches(0.7), Inches(5.38), Inches(12), Inches(0.4), "Also: honest about uncertainty", size=14, bold=True, color=GREEN)
    _text(s, Inches(0.7), Inches(5.72), Inches(12), Inches(1.05),
          "For a plant the model was never trained on, it still advises — but shows a confidence figure and an explicit caution, "
          "and asks for more photographs when confidence is low. Wording is always “confidence”, never “accuracy”.",
          size=13, color=INK)
    _notes(s, 8)

    # ===== 10. Results =====
    s = _base(prs, "What exists today — prototype and results", "Results (preliminary, own evaluation)")
    _text(s, Inches(0.5), Inches(1.45), Inches(5.9), Inches(0.4), "The corpus", size=15, bold=True, color=GREEN)
    rows = [("Text", "Passages"), ("Brihat Samhita (12 chapters)", "140"), ("Vrikshayurveda (full)", "52"),
            ("Kashyapiya Krishisukti", "39"), ("Upavanavinoda", "15"), ("Krishi Parashara", "13"),
            ("Natural-farming manual (NITI Aayog, modern IKS-derived)", "68"), ("Total", "327")]
    ty = Inches(1.85)
    for i, (a, b) in enumerate(rows):
        fill = GREEN if i == 0 else (SOFT if i == len(rows) - 1 else (CARD if i % 2 else PARCH))
        _rect(s, Inches(0.5), ty, Inches(4.7), Inches(0.36), fill, line=LINE, lw=0.5)
        _rect(s, Inches(5.2), ty, Inches(1.2), Inches(0.36), fill, line=LINE, lw=0.5)
        col = WHITE if i == 0 else INK
        _text(s, Inches(0.55), ty, Inches(4.6), Inches(0.36), a, size=11, bold=(i in (0, len(rows) - 1)), color=col, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, Inches(5.2), ty, Inches(1.2), Inches(0.36), b, size=11, bold=(i in (0, len(rows) - 1)), color=col, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        ty += Inches(0.36)
    _text(s, Inches(0.5), ty + Inches(0.08), Inches(5.9), Inches(0.5),
          "Five classical texts digitised; Vishvavallabha (plant disease & pest management) next.", size=11, italic=True, color=MUTED)
    mx = Inches(6.8)
    _text(s, mx, Inches(1.45), Inches(6.0), Inches(0.4), "Measured behaviour", size=15, bold=True, color=GREEN)
    metrics = [
        ("0.01–0.04 → 0.59–0.96", "relevance of what is found: modern label vs symptom rewriting"),
        ("0.94 vs 0.70", "search quality: by meaning vs by keyword"),
        ("100% · 0%", "honest refusal on unanswerable queries · fabricated citations"),
        ("~90% · ~96%", "soil type · moisture recognition"),
        ("near-perfect", "healthy-vs-diseased separation; attention verified on the lesion"),
    ]
    my = Inches(1.85)
    for big, lab in metrics:
        _rect(s, mx, my, Inches(6.0), Inches(0.78), CARD, MSO_SHAPE.ROUNDED_RECTANGLE, line=LINE)
        _text(s, mx + Inches(0.12), my, Inches(3.05), Inches(0.78), big, size=14, bold=True, color=OCHRE, font=SERIF, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, mx + Inches(3.2), my, Inches(2.7), Inches(0.78), lab, size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        my += Inches(0.86)
    _text(s, mx, my + Inches(0.05), Inches(6.0), Inches(0.6),
          "Limiter today is coverage, not method: where the texts are thin, the system refuses rather than guesses. Hence the corpus expansion.",
          size=11, italic=True, color=MUTED)
    _notes(s, 9)

    # ===== 11. Why it matters + next =====
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, GREEN)
    _text(s, Inches(0.8), Inches(0.5), Inches(8), Inches(0.4), "WHY THIS MATTERS FOR REVITALISING IKS", size=12, bold=True, color=OCHRE)
    _text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(1.6),
          "A text that is cited every day is preserved — and alive.", size=38, bold=True, color=WHITE, font=SERIF)
    _bullets(s, Inches(0.8), Inches(2.7), Inches(11.5), Inches(1.9), [
        "Documentation and use reinforce each other — retrieval is preservation in practice",
        "Verifiable, not distorted — every answer traces to its source verse",
        "In the farmer’s hands, not only in archives — a template for AI on Indigenous knowledge beyond agriculture",
    ], size=18, color=WHITE, gap=8)
    _rect(s, Inches(0.8), Inches(4.65), Inches(11.7), Inches(0.05), OCHRE)
    _text(s, Inches(0.8), Inches(4.8), Inches(8), Inches(0.4), "NEXT STEPS", size=12, bold=True, color=OCHRE)
    _bullets(s, Inches(0.8), Inches(5.15), Inches(11.5), Inches(1.2), [
        "Validation of recommendations with agronomists   ·   add Vishvavallabha   ·   expert-built evaluation set",
    ], size=16, color=WHITE)
    _text(s, Inches(0.8), Inches(6.25), Inches(11.5), Inches(0.6), "Thank you", size=32, bold=True, color=WHITE, font=SERIF)
    _text(s, Inches(0.8), Inches(6.85), Inches(11.5), Inches(0.4),
          "Ankit Pawar  ·  Dr. Akshay Pandey  ·  PDPM IIITDM Jabalpur", size=14, color=CREAM)
    _notes(s, 10)

    prs.save(OUT_PPTX)
    for p in (leaf, soil):
        p.unlink(missing_ok=True)
    return OUT_PPTX


# --------------------------------------------------------------------------- speaker script (.docx)
def build_script() -> Path:
    from docx import Document
    from docx.shared import Pt as DPt

    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", DPt(14)     # phone-readable

    d.add_heading("Speaker script — ICSSR seminar, 11 Sep 2026 (5 minutes)", level=1)
    p = d.add_paragraph()
    p.add_run("Read the paragraph for each slide while that slide is shown. The bold line under "
              "each paragraph is the cue to click. Total ≈ 5 minutes at a normal pace. If the "
              "moderator is cutting time, skip Slide 4 and Slide 10 — never skip 5, 6, 7 or 9.").italic = True

    for i, (name, t, body) in enumerate(SCRIPT, 1):
        d.add_heading(f"Slide {i} — {name}   [{t}]", level=2)
        d.add_paragraph(body)
        nxt = "→ next slide" if i < len(SCRIPT) else "→ stop, smile, wait for questions"
        r = d.add_paragraph().add_run(nxt)
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

    d.save(OUT_DOCX)
    return OUT_DOCX


if __name__ == "__main__":
    pp = build_deck()
    dx = build_script()
    prs = Presentation(pp)
    words = sum(len(b.split()) for _, _, b in SCRIPT)
    print(f"deck   : {pp.relative_to(ROOT)}  ({len(prs.slides)} slides, notes on all="
          f"{all(sl.notes_slide.notes_text_frame.text.strip() for sl in prs.slides)})")
    print(f"script : {dx.relative_to(ROOT)}  ({words} words ≈ {words/145:.1f} min at 145 wpm)")
