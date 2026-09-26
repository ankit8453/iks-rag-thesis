"""Build the NCSTC 2026 participation brief for Dr. Akshay Pandey (.docx).

Purpose: BEFORE writing a proposal, Dr. Pandey needs to decide whether we apply,
under which component, and whether he will be Principal Investigator. This brief
answers "how can we participate and what is needed" in one readable page-set, and
ends with the specific decisions and institutional data only he can supply.

Not a proposal. A decision brief.

Source: docs/SchemeId_2344_NewCall2026NCSTCfinal.docx (NCSTC/DST Call 2026).
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT_DOCX = ROOT / "NCSTC_PARTICIPATION_BRIEF_2026-09-17.docx"

TITLE = "NCSTC (DST) Call for Proposals 2026 — Can We Apply, and What Would It Take?"
SUBTITLE = ("Prepared for Dr. Akshay Pandey  |  17 September 2026  |  "
            "Ankit Pawar, M.Tech (CSE), PDPM IIITDM Jabalpur")

ACCENT = RGBColor(0x1F, 0x4E, 0x79)


# --------------------------------------------------------------------------- #
# document helpers
# --------------------------------------------------------------------------- #
def style_doc(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Pt(54)
        s.left_margin = s.right_margin = Pt(54)


def para(doc, text="", *, bold=False, italic=False, size=11, align=None,
         space_after=6, color=None, indent=0):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    return p


def rich(doc, parts, *, size=11, space_after=6, indent=0):
    """A paragraph from [(text, bold), ...] so key terms can be emphasised."""
    p = doc.add_paragraph()
    for text, bold in parts:
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.bold = bold
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    return p


def heading(doc, text, *, size=13):
    para(doc, text, bold=True, size=size, space_after=4, color=ACCENT)


def bullet(doc, text, *, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    return p


def numbered(doc, text, *, bold_lead=None):
    p = doc.add_paragraph(style="List Number")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    return p


def table(doc, rows, *, widths=None, header=True):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            r.bold = header and ri == 0
            p.paragraph_format.space_after = Pt(2)
    return t


def rule(doc):
    para(doc, "_" * 92, size=8, space_after=8, color=RGBColor(0xBB, 0xBB, 0xBB))


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def build(doc: Document) -> None:
    style_doc(doc)
    para(doc, TITLE, bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, SUBTITLE, size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=10)
    rule(doc)

    # ---- 1. the ask ------------------------------------------------------- #
    heading(doc, "1.  Summary — what I am asking you to decide")
    para(doc, "Sir, you forwarded the NCSTC (DST) Call for Proposals 2026 and asked how we "
              "might participate. I have read the call in full. This note sets out where we "
              "fit, what we can honestly claim, and what is needed from the Institute. It is "
              "not a proposal — I would like your decision on four points first, because "
              "three of them only you can answer.")
    rich(doc, [("The deadline is 30 September 2026", True),
               (" — thirteen days from today. Submission is online at onlinedst.gov.in. "
                "Most of the remaining work is institutional paperwork rather than writing, "
                "so a decision within two or three days would still leave time.", False)])

    # ---- 2. nature of the call ------------------------------------------- #
    heading(doc, "2.  What kind of funding this is")
    rich(doc, [("NCSTC funds science communication, not research.", True),
               (" It supports taking scientific knowledge to students, communities and the "
                "public — toolkits, learning modules, demonstrations, outreach. It does not "
                "fund thesis work as such.", False)])
    para(doc, "This matters for how we frame the application. We cannot submit our M.Tech "
              "work as a research project. We would instead propose to take what we have "
              "already built and turn it into a teaching and outreach resource. In my "
              "reading that is an advantage rather than a limitation: most applicants "
              "propose to build something, whereas we would be proposing to disseminate "
              "something that already runs.")

    # ---- 3. component fit ------------------------------------------------- #
    heading(doc, "3.  Recommended component")
    rich(doc, [("Science and Technology Communication Tools → Value Added Learning "
                "Materials", True),
               (", defined in the call as: “Development of teaching-learning materials, "
                "specific toolkits and modules on application of AI in S&T to augment "
                "learning skills in STEM fields.”", False)])
    para(doc, "This is the closest match to what we have. Two other components are "
              "plausible and I mention them so the choice is on the record:")
    bullet(doc, "the IKS bullet asks for methods and tools to “integrate, popularise "
                "and disseminate Indian Knowledge Systems within the modern context”, "
                "which describes our work almost word for word — but that whole section is "
                "framed around Indian mathematical heritage, so it is a stretch;",
           bold_lead="Bhartiya Ganita Literacy — ")
    bullet(doc, "STEM students engage with communities to identify grassroots problems, "
                "guided by faculty mentors. A natural fit for an Institute project, and it "
                "is new this year so the field may be less crowded.",
           bold_lead="Samvaad — ")
    para(doc, "My recommendation is Value Added Learning Materials, with the IKS angle as "
              "the subject matter of the toolkit rather than the category we apply under. "
              "I would welcome your view.", italic=True)

    # ---- 4. what we would propose ---------------------------------------- #
    heading(doc, "4.  What we would propose to do")
    rich(doc, [("Working title: ", True),
               ("An AI Toolkit for Indian Agricultural Knowledge — teaching students how "
                "artificial intelligence can be grounded in classical Indian texts without "
                "distorting them.", False)])
    para(doc, "The teaching idea, in one sentence: students see a working system that reads "
              "classical Indian agricultural treatises, gives advice from a photograph of a "
              "diseased leaf, cites the exact verse it used — and openly refuses when the "
              "texts have no answer.")
    para(doc, "That refusal behaviour is the pedagogical core. Students today use AI "
              "chatbots that state falsehoods with complete confidence. Our system "
              "demonstrates the alternative, and demonstrates it on Indian heritage "
              "material, where a fabricated claim would be a form of distortion of the "
              "tradition itself.")
    para(doc, "Indicative deliverables (to be firmed up once the component is fixed):")
    bullet(doc, "a hands-on module in which students query the system, inspect the "
                "retrieved verse, and verify the citation against the printed text;")
    bullet(doc, "a workbook on grounded AI: retrieval, citation, and abstention, taught "
                "through the IKS corpus rather than through abstract examples;")
    bullet(doc, "an open, searchable digital corpus of the classical treatises, usable by "
                "other institutions;")
    bullet(doc, "outreach sessions for school- and college-level STEM students, and a "
                "demonstration for a farming community in the selected district.")

    # ---- 5. what already exists ------------------------------------------ #
    heading(doc, "5.  What we can already show (this is our strongest card)")
    para(doc, "The call asks for prior experience and for evidence that the work is "
              "feasible. We have a running system, not a plan:")
    table(doc, [
        ["Component", "Present status"],
        ["Digital corpus", "233 verse-level passages from five classical treatises "
                           "(Vrikshayurveda, Brihat Samhita, Krishi Parashara, "
                           "Upavanavinoda, Kashyapiya Krishisukti) plus a modern "
                           "natural-farming manual; searchable, with metadata"],
        ["Plant-disease recognition", "EfficientNet-B4 cascade; retrained on leaf crops so "
                                      "that its attention rests on the lesion, verified by "
                                      "Grad-CAM visual explanation"],
        ["Soil recognition", "Multi-task model: soil type 89.9%, moisture 95.8%"],
        ["Grounded advisory", "Hybrid retrieval with re-ranking; the correct treatise is "
                              "retrieved for every evaluation query; answers carry "
                              "verse-level citations"],
        ["Faithfulness", "Zero fabricated citations across all evaluation runs; the system "
                         "abstains where the texts are silent"],
        ["Working prototype", "End-to-end interface: leaf photo + soil photo → "
                              "diagnosis → cited advice → visual explanation"],
    ])
    para(doc)
    rich(doc, [("We also have a measured result worth presenting publicly. ", True),
               ("We tested the system on 22 modern crop-disease descriptions and found that "
                "the classical texts can genuinely answer only a few of them — the treatises "
                "classify plant disorders by cause in Ayurvedic terms, whereas modern "
                "diagnosis classifies by the appearance of the lesion. Rather than let the "
                "system invent remedies to fill that gap, it refuses. Quantifying where "
                "Indian classical knowledge does and does not map onto modern disease "
                "categories is, I think, a contribution in its own right, and it is exactly "
                "the kind of honest finding a science-communication project should carry to "
                "students.", False)])

    doc.add_page_break()

    # ---- 6. what is needed ------------------------------------------------ #
    heading(doc, "6.  What is needed — and who must provide it")
    rich(doc, [("The Principal Investigator must be a faculty member with an affiliated "
                "organisation. ", True),
               ("As an M.Tech student I cannot hold that role. If we apply, the proposal "
                "would be yours, with PDPM IIITDM Jabalpur as the proposer institution and "
                "with me contributing the technical work. The form asks for the PI's "
                "designation, organisation, PFMS Unique ID and institutional details.",
                False)])
    para(doc)
    table(doc, [
        ["Required by the form", "Who supplies it"],
        ["Principal Investigator details (designation, DOB, category, contact)",
         "Dr. Akshay Pandey"],
        ["Institute details: address, website, establishment year, PFMS Unique ID, name and "
         "designation of the Head of the Institution",
         "Institute administration / Dean R&D"],
        ["Prior experience in public engagement and science communication; list of ongoing "
         "and completed projects; five recent projects of a similar nature with monitoring "
         "and beneficiary-feedback mechanisms",
         "Dr. Akshay Pandey / Department"],
        ["Target area (village, block, district, state) and target group, with the basis on "
         "which they were selected and consultations held",
         "To be decided jointly"],
        ["Budget: total, recurring, non-recurring, with itemised justification",
         "To be decided jointly"],
        ["Objectives, need identification, methodology, work plan, timelines, key "
         "performance indicators, impact-assessment methodology, deliverables and "
         "sustainability",
         "I will draft these"],
        ["Infrastructure and resources already available; self-generated resources in cash "
         "and kind", "Institute / Department"],
    ])

    # ---- 7. risks --------------------------------------------------------- #
    para(doc)
    heading(doc, "7.  An honest assessment of our chances")
    bullet(doc, "the form asks for five recent projects of a similar nature, with evidence "
                "of monitoring and beneficiary feedback. If the Department has not run "
                "science-communication or outreach grants before, this section will be thin, "
                "and it is the section most likely to weigh against us. I would rather raise "
                "this now than after we have spent two weeks writing.",
           bold_lead="The main risk is track record: ")
    bullet(doc, "thirteen days is workable for the narrative sections, but the PFMS ID and "
                "institutional signatures depend on offices we do not control.",
           bold_lead="Time: ")
    bullet(doc, "a working prototype, a genuine IKS subject that matches national priority, "
                "a measured result about AI faithfulness, and a clear student audience.",
           bold_lead="In our favour: ")
    rich(doc, [("If the track-record requirement proves to be a barrier this year, an "
                "alternative is to apply jointly with a partner that has outreach "
                "experience — the form explicitly provides for a partner or collaborating "
                "institution.", False)])

    # ---- 8. decisions ----------------------------------------------------- #
    heading(doc, "8.  Decisions I need from you")
    numbered(doc, "Do we apply this year?", bold_lead="")
    numbered(doc, "Are you willing to be Principal Investigator, with IIITDM Jabalpur as "
                  "the proposer institution?", bold_lead="")
    numbered(doc, "Which component — Value Added Learning Materials (my recommendation), "
                  "the IKS bullet under Bhartiya Ganita Literacy, or Samvaad?",
             bold_lead="")
    numbered(doc, "Have you or the Department held any DST or science-communication "
                  "project previously that we can list?", bold_lead="")
    numbered(doc, "Which district or villages should we name as the target area, and what "
                  "budget scale do you have in mind?", bold_lead="")

    # ---- 9. plan ---------------------------------------------------------- #
    heading(doc, "9.  If you say yes — proposed schedule")
    table(doc, [
        ["Dates", "Work"],
        ["18–19 Sep", "Your decision on the five points above; request PFMS ID and "
                      "institutional details from the administration"],
        ["20–24 Sep", "I draft the full proposal in the prescribed format: summary, "
                      "objectives, need identification, target group, methodology, work "
                      "plan, timelines, KPIs, impact assessment, deliverables"],
        ["24–26 Sep", "Budget preparation with itemised justification; your review"],
        ["27–28 Sep", "Revision, institutional signatures, supporting annexures"],
        ["29 Sep", "Online submission at onlinedst.gov.in, one day before the deadline"],
    ])
    para(doc)
    para(doc, "I am ready to begin drafting as soon as you confirm the component. The "
              "technical and evaluation content is already written up and can be adapted "
              "quickly to the prescribed format.")
    para(doc)
    para(doc, "Ankit Pawar", bold=True, space_after=0)
    para(doc, "M.Tech (Computer Science & Engineering), PDPM IIITDM Jabalpur", size=10,
         italic=True)

    # ---- annexure --------------------------------------------------------- #
    doc.add_page_break()
    heading(doc, "Annexure — the call in brief")
    table(doc, [
        ["Item", "Detail"],
        ["Scheme", "National Council of Science and Technology Communication (NCSTC), "
                   "Department of Science and Technology, Government of India"],
        ["Call", "Call for Proposals 2026"],
        ["Last date", "30 September 2026"],
        ["Submission", "Online at https://onlinedst.gov.in (format from "
                       "https://dst.gov.in/whatsnew/announcement)"],
        ["Contact", "Dr. Rashmi Sharma, Head NCSTC, Technology Bhawan, New Mehrauli Road, "
                    "New Delhi — r.sharma72@nic.in, 011-29512324 Extn. 12018"],
        ["Eligibility note", "Industries cannot apply directly for grant-in-aid; the "
                             "proposer is to be Government, Not-for-Profit or Autonomous"],
    ])
    para(doc)
    para(doc, "Components invited under the call:", bold=True)
    for t in ["Empowering STEM Students for Community Engagement and Career Exploration "
              "through Samvaad",
              "Science and Technology Communication Tools — traditional scientific toys and "
              "games; advanced scientific communication tools; Value Added Learning "
              "Materials (our recommended fit)",
              "Bhartiya Ganita Literacy — Sulba Sutras; Bhartiya Ganita in computing and AI; "
              "integrating and disseminating Indian Knowledge Systems",
              "Children Science Congress — district, state and 33rd national level",
              "SciCom for Sustainable Future — occupational hazards; health and environment; "
              "risk communication; frontier domains including AI and robotics",
              "Hands-on Science Programme — science on wheels, mobile labs",
              "National Science Day celebration and other innovative ideas"]:
        bullet(doc, t)


def main() -> None:
    doc = Document()
    build(doc)
    doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
