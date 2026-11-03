"""Build the dated Word protocol from the maintained Markdown specification.

Run with the bundled document Python runtime. Render with render_docx.py after building.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def text_runs(paragraph, text):
    for index, part in enumerate(text.split("`")):
        part = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", part)
        run = paragraph.add_run(part)
        if index % 2:
            run.font.name = "Consolas"
            run.font.size = Pt(9)


def build():
    doc = Document()
    for border in doc.styles.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.7)
    section.left_margin = section.right_margin = Inches(.8)
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = doc.styles[name]
        style.font.name = "Microsoft YaHei"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    doc.styles["Title"].font.size = Pt(23)
    doc.styles["Heading 1"].font.size = Pt(15)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(15)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(7)

    doc.add_paragraph("施工人员工效智能感知历史视频接口规范", "Title")
    doc.add_paragraph("2026 年 11 月 3 日修订版")
    lines = (ROOT / "docs/history_video_api.md").read_text(encoding="utf-8").splitlines()[1:]
    index = 0
    in_code = False
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            in_code = not in_code
            index += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            run = p.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
        elif line.startswith("## "):
            doc.add_paragraph(line[3:], "Heading 1")
        elif line.startswith("| "):
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                row = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Table Grid"
            table.autofit = False
            widths = (1.5, .7, 4.7) if len(rows[0]) == 3 else (2.8, 4.1)
            for column, width in zip(table.columns, widths):
                column.width = Inches(width)
            for row_index, values in enumerate(rows):
                cells = table.add_row().cells
                for cell, value, width in zip(cells, values, widths):
                    cell.width = Inches(width)
                    cell.paragraphs[0].paragraph_format.space_after = Pt(3)
                    text_runs(cell.paragraphs[0], value)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        run.bold = row_index == 0
            repeat = OxmlElement("w:tblHeader")
            table.rows[0]._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after = Pt(1)
            continue
        elif line.strip():
            style = "List Bullet" if line.startswith("- ") else "Normal"
            paragraph = doc.add_paragraph(style=style)
            text_runs(paragraph, line[2:] if style == "List Bullet" else line)
            if line.startswith("`POST "):
                paragraph.paragraph_format.keep_with_next = True
        index += 1
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("工效历史视频接口 1103   ·   ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    output = ROOT / "docs/施工人员工效智能感知历史视频接口规范-1103.docx"
    doc.save(output)
    print(output)


if __name__ == "__main__":
    build()
