"""Render COMPLETE_PROJECT_DOCUMENTATION.md to a clean, professional PDF.

Pipeline: Markdown -> styled standalone HTML -> PDF via headless Edge/Chrome.
The CSS mimics a formal Word/report look (blue section headings, ruled tables)
so the output is suitable to hand to a thesis supervisor.

Usage:  python scripts/md_to_pdf.py [input.md] [output.pdf]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = ROOT / "COMPLETE_PROJECT_DOCUMENTATION.md"

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  color: #222; font-size: 10.6pt; line-height: 1.5; margin: 0;
}
h1 { font-size: 21pt; color: #1f4e79; margin: 0 0 2px; font-weight: 700; }
h1 + h2 { border: none; margin-top: 2px; }
h2 {
  font-size: 14pt; color: #1f6fb2; font-weight: 700;
  margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 2px solid #1f6fb2;
  page-break-after: avoid;
}
h3 {
  font-size: 11.6pt; color: #2178b8; font-weight: 700;
  margin: 15px 0 5px; page-break-after: avoid;
}
p { margin: 6px 0; }
em { color: #555; }
strong { color: #1a1a1a; }
a { color: #1f6fb2; text-decoration: none; }
ul, ol { margin: 6px 0 6px 0; padding-left: 22px; }
li { margin: 3px 0; }
code {
  font-family: Consolas, "Courier New", monospace; font-size: 9.4pt;
  background: #f2f4f8; border: 1px solid #e1e6ef; border-radius: 3px; padding: 0 4px;
}
blockquote {
  margin: 8px 0; padding: 7px 14px; color: #475569;
  background: #f6f8fc; border-left: 4px solid #9db8d6; font-style: italic;
}
table {
  border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.8pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #c9d3e0; padding: 6px 9px; text-align: left; vertical-align: top;
}
th { background: #eaf1f9; color: #1f4e79; font-weight: 700; }
tr:nth-child(even) td { background: #f7f9fc; }
hr { border: none; border-top: 1px solid #d7dde6; margin: 18px 0; }
/* the italic subtitle block right under the title */
h1 ~ p em { color: #8a5a1a; }
"""


def main() -> int:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else md_path.with_suffix(".pdf")
    html_path = md_path.with_suffix(".render.html")

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "sane_lists", "attr_list", "md_in_html"],
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")

    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    browser = edge if edge.exists() else chrome
    if not browser.exists():
        print("No Edge/Chrome found for PDF printing.", file=sys.stderr)
        return 1

    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    print("Printing PDF via:", browser.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not pdf_path.exists():
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        print("PDF was not produced.", file=sys.stderr)
        return 1

    html_path.unlink(missing_ok=True)
    size_kb = pdf_path.stat().st_size / 1024
    print(f"OK -> {pdf_path}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
