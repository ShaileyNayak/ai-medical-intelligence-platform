"""Generate docs/Project_Report.pdf from Project_Report.md using reportlab."""

from __future__ import annotations

from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
except ImportError as exc:
    raise SystemExit("Install reportlab: pip install reportlab") from exc

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "Project_Report.md"
OUT = ROOT / "docs" / "Project_Report.pdf"


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h_style = ParagraphStyle("HCustom", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9, leading=12)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        safe = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if line.startswith("# "):
            story.append(Paragraph(safe[2:], title_style))
        elif line.startswith("## "):
            story.append(Paragraph(safe[3:], h_style))
        elif line.startswith("```"):
            continue
        else:
            story.append(Paragraph(safe, body))
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
