"""Professional, print-ready PDF documentation report for analysed Facebook content.

Fully offline (fpdf2 + uharfbuzz for Arabic/RTL text shaping). Structure:

  cover page -> table of contents -> 1. executive summary -> 2. methodology ->
  3. taxonomy -> 4. findings at a glance (vector charts) -> 5. flagged items
  (one colour-coded record each) -> 6. review & sign-off ->
  Appendix A: full item inventory; Appendix B: analysis failures.

A Unicode TTF font with Arabic coverage is picked automatically (Arial /
Segoe UI / DejaVu, or GBV_PDF_FONT env var); without one the report falls
back to a latin-1 core font.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import datetime, timezone

from fpdf import FPDF
from fpdf.enums import TableCellFillMode
from fpdf.fonts import FontFace
from PIL import Image

from analyzer import CATEGORIES, SEVERITIES

__all__ = ["build_pdf_report", "save_pdf_report"]

TOOL = "fb_gbv_monitor"
FONT_FAMILY = "AppFont"

# ------------------------------------------------------------------ palette
PRIMARY = (38, 34, 84)        # deep indigo
ACCENT = (99, 74, 156)
TEXT = (39, 42, 48)
MUTED = (110, 115, 125)
FAINT = (150, 154, 162)
BORDER = (214, 217, 224)
BG_LIGHT = (246, 247, 250)
BG_SOFT = (240, 242, 246)
WHITE = (255, 255, 255)

SEV_COLORS = {
    "critical": (168, 28, 44),
    "high": (222, 92, 30),
    "medium": (224, 148, 20),
    "low": (110, 145, 70),
    "none": (130, 135, 143),
}

BAR_PALETTE = [
    (99, 74, 156), (42, 132, 137), (196, 88, 58), (86, 124, 88), (148, 100, 160),
    (66, 118, 176), (168, 116, 46), (110, 143, 96), (158, 76, 126), (64, 128, 154),
    (134, 108, 62), (152, 84, 84), (72, 104, 150), (114, 114, 114),
]

CAT_LABELS = {
    "sexist_hate_speech": "Sexist hate speech",
    "misogynistic_insults": "Misogynistic insults",
    "sexual_harassment": "Sexual harassment",
    "threats_of_violence": "Threats of violence",
    "incitement_to_violence": "Incitement to violence",
    "body_or_slut_shaming": "Body / slut-shaming",
    "doxxing_or_privacy_violation": "Doxxing / privacy violation",
    "stalking_or_intimidation": "Stalking / intimidation",
    "ncii_or_sextortion": "NCII / sextortion",
    "victim_blaming": "Victim-blaming",
    "silencing_or_discrediting": "Silencing / discrediting",
    "trivialising_gbv": "Trivialising GBV",
    "sexist_stereotyping": "Sexist stereotyping",
    "other_gbv": "Other GBV",
}
TARGET_LABELS = {
    "individual_woman": "Individual woman",
    "public_figure_woman": "Woman public figure",
    "women_as_group": "Women as a group",
    "girl_minor": "Girl / minor",
    "unclear": "Unclear",
    "none": "None specified",
}
LANG_LABELS = {
    "ar": "Arabic", "darija": "Tunisian Darija", "fr": "French",
    "en": "English", "mixed": "Mixed", "other": "Other",
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2028\u2029\u2066-\u2069]")


# ------------------------------------------------------------------ text utils
def _txt(s) -> str:
    return _CONTROL_RE.sub("", str(s if s is not None else "")).replace("\r", "").strip()


def _excerpt(s: str, n: int) -> str:
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _fmt_ts(ts: str) -> str:
    ts = _txt(ts)
    if not ts:
        return "n/a"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return ts


def _fin(s: str) -> str:
    if not s or s == "…":
        return "(the excerpt was empty)"
    return s


# ------------------------------------------------------------------ fonts
def _font_candidates() -> dict[str, str | None]:
    """Find a system TTF with Arabic + Latin coverage (GBV_PDF_FONT overrides)."""
    w = "C:/Windows/Fonts"
    m = "/System/Library/Fonts/Supplemental"
    d = "/usr/share/fonts/truetype/dejavu"
    reg = os.environ.get("GBV_PDF_FONT")
    if reg:
        bold = os.environ.get("GBV_PDF_FONT_BOLD", reg)
        ital = os.environ.get("GBV_PDF_FONT_ITALIC", reg)
        return {"": _first(reg), "B": _first(bold), "I": _first(ital), "BI": _first(bold)}
    a = _first(f"{w}/arial.ttf", f"{m}/Arial.ttf")
    if a:
        base = os.path.dirname(a)
        return {
            "": a,
            "B": _first(f"{base}/arialbd.ttf", a),
            "I": _first(f"{base}/ariali.ttf", a),
            "BI": _first(f"{base}/arialbi.ttf", a),
        }
    n = "/usr/share/fonts/truetype/noto"
    nt = _first(f"{n}/NotoSansArabic-Regular.ttf", f"{n}/NotoNaskhArabic-Regular.ttf")
    if nt:
        return {
            "": nt,
            "B": _first(f"{n}/NotoSansArabic-Bold.ttf", f"{n}/NotoNaskhArabic-Bold.ttf", nt),
            "I": nt,
            "BI": nt,
        }
    dv = _first(f"{d}/DejaVuSans.ttf")
    if dv:
        return {
            "": dv,
            "B": _first(f"{d}/DejaVuSans-Bold.ttf", dv),
            "I": _first(f"{d}/DejaVuSans-Oblique.ttf", dv),
            "BI": _first(f"{d}/DejaVuSans-BoldOblique.ttf", dv),
        }
    s = _first(f"{w}/segoeui.ttf")
    if s:
        return {
            "": s,
            "B": _first(f"{w}/segoeuib.ttf", s),
            "I": _first(f"{w}/segoeuii.ttf", s),
            "BI": _first(f"{w}/segoeuiz.ttf", s),
        }
    return {}


def _first(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


# ------------------------------------------------------------------ pdf
class _ReportPDF(FPDF):
    def __init__(self, title: str, generated: str, footer_note: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.rep_title = title
        self.rep_generated = generated
        self.footer_note = footer_note
        self.chrome = False
        self.unicode_font = True
        self.alias_nb_pages("{nb}")
        self.set_margins(18, 17, 18)
        self.set_auto_page_break(True, margin=17)
        self.set_subject("Documentation of gender-based cyberviolence / hate speech against women")
        self.set_creator(TOOL)
        self.set_author(f"{TOOL} — local AI analysis")

    def T(self, s) -> str:
        """Sanitise text; degrade to latin-1 if no Unicode font was found."""
        s = _txt(s)
        return s if self.unicode_font else s.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        if not self.chrome:
            return
        self.set_text_color(*MUTED)
        self.set_font(self.font_family, "", 7.8)
        self.set_y(8)
        self.cell(0, 4.6, self.T(self.rep_title), align="L", new_x="RIGHT", new_y="TOP")
        self.cell(0, 4.6, self.T(self.rep_generated), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BORDER)
        self.set_line_width(0.25)
        self.line(self.l_margin, 13.6, self.w - self.r_margin, 13.6)
        self.set_line_width(0.1)
        self.set_text_color(*TEXT)

    def footer(self):
        if not self.chrome:
            return
        self.set_y(-11)
        self.set_font(self.font_family, "", 7.3)
        self.set_text_color(*FAINT)
        self.cell(0, 5, self.T(self.footer_note), align="L", new_x="RIGHT", new_y="TOP")
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)

    # ------------------------------------------------------- layout helpers
    def lines_of(self, text: str, w: float = 0, size: float = 8.5, line_h: float = 0) -> float:
        """Exact height (mm) a multi_cell would take, measured via dry run."""
        if not text:
            return 0.0
        line_h = line_h or size * 0.53
        try:
            return float(self.multi_cell(w, line_h, text, dry_run=True, output="HEIGHT"))
        except Exception:
            approx = max(10.0, w / (size * 0.42))
            return max(1, math.ceil(len(text) / approx)) * line_h

    def ensure_room(self, needed_mm: float) -> None:
        if self.get_y() + needed_mm > self.page_break_trigger:
            self.add_page()

    def h1(self, number: str, title: str) -> None:
        label = f"{number} {title}"
        self.set_text_color(*PRIMARY)
        self.set_font(self.font_family, "B", 14.5)
        self.cell(0, 9, self.T(f"{number}   {title}"), new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.set_line_width(0.9)
        self.set_draw_color(*ACCENT)
        self.line(self.l_margin, y + 1.2, self.l_margin + 12, y + 1.2)
        self.set_line_width(0.25)
        self.set_draw_color(*BORDER)
        self.line(self.l_margin + 14, y + 1.2, self.w - self.r_margin, y + 1.2)
        self.set_line_width(0.1)
        self.start_section(label, level=0)
        if getattr(self, "_collect_toc", False):       # pass 1: record page numbers
            self._toc_entries.append((label, self.page_no()))
        self.set_y(y + 5.5)
        self.set_text_color(*TEXT)

    def h2(self, title: str) -> None:
        self.set_text_color(*ACCENT)
        self.set_font(self.font_family, "B", 10.5)
        self.set_x(self.l_margin)
        self.cell(0, 6.4, self.T(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)

    def para(self, text: str, size: float = 9, color: tuple = TEXT, style: str = "") -> None:
        self.set_font(self.font_family, style, size)
        self.set_text_color(*color)
        self.multi_cell(0, size * 0.53, self.T(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)

    def bullets(self, items: list[str]) -> None:
        self.set_font(self.font_family, "", 9)
        self.set_text_color(*TEXT)
        lh = 9 * 0.53
        for it in items:
            h = self.lines_of(it, self.epw - 6, 9)
            self.set_x(self.l_margin)
            self.cell(6, lh, self.T("•"), new_x="RIGHT", new_y="TOP")
            self.multi_cell(self.epw - 6, lh, self.T(it), new_x="LMARGIN", new_y="NEXT")
            self.ln(1.2)
        self.ln(0.8)

    def bullet_one(self, text: str) -> None:
        self.set_font(self.font_family, "", 9)
        self.set_text_color(*TEXT)
        lh = 9 * 0.53
        self.set_x(self.l_margin)
        self.cell(6, lh, self.T("–"), new_x="RIGHT", new_y="TOP")
        self.multi_cell(self.epw - 6, lh, self.T(text), new_x="LMARGIN", new_y="NEXT")

    def shaded_block(self, text: str, bg: tuple, size: float = 8.6, bar: tuple | None = None) -> None:
        """Multi-line text on a filled panel (kept on a single page)."""
        pad_v, pad_h = 2.6, 3.2
        lh = size * 0.53
        h = self.lines_of(text, self.epw - 2 * pad_h, size, lh) + 2 * pad_v
        self.set_fill_color(*bg)
        self.rect(self.l_margin, self.get_y(), self.epw, h, style="F")
        if bar:
            self.set_fill_color(*bar)
            self.rect(self.l_margin, self.get_y(), 1.5, h, style="F")
        self.set_y(self.get_y() + pad_v)
        self.set_font(self.font_family, "", size)
        self.set_text_color(*TEXT)
        self.set_x(self.l_margin + pad_h)
        self.multi_cell(self.epw - 2 * pad_h, lh, self.T(text), new_x="LMARGIN", new_y="NEXT")
        self.set_y(self.get_y() + pad_v)

    def key_value(self, pairs: list[tuple[str, str]], label_w: float = 46, size: float = 9,
                  gap: float = 1.4) -> None:
        lh = size * 0.53
        for label, value in pairs:
            self.set_y(self.get_y())
            self.set_font(self.font_family, "B", size * 0.86)
            self.set_text_color(*MUTED)
            self.set_x(self.l_margin)
            self.cell(label_w, lh, self.T(label), new_x="RIGHT", new_y="TOP")
            self.set_font(self.font_family, "", size)
            self.set_text_color(*TEXT)
            self.set_x(self.l_margin + label_w)
            self.multi_cell(self.epw - label_w, lh, self.T(value), new_x="LMARGIN", new_y="NEXT")
            if gap:
                self.ln(gap)

    # ------------------------------------------------------------- widgets
    def stat_box(self, x: float, y: float, w_mm: float, h_mm: float, value: str,
                 label: str, color: tuple) -> None:
        light = tuple(int(c + (255 - c) * 0.9) for c in color)
        self.set_fill_color(*light)
        self.rect(x, y, w_mm, h_mm, style="F", round_corners=True, corner_radius=1.6)
        self.set_xy(x + 3, y + 3)
        self.set_font(self.font_family, "B", 15)
        self.set_text_color(*color)
        self.cell(w_mm - 6, 7.5, self.T(value), align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + 3, y + 11.5)
        self.set_font(self.font_family, "B", 6.6)
        self.set_text_color(*MUTED)
        self.cell(w_mm - 6, 3.6, self.T(label.upper()), align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)

    def hbar_chart(self, pairs: list[tuple[str, int]], x: float, w_mm: float, *,
                   label_w: float = 26, colors: list | None = None,
                   row_h: float = 6.0, size: float = 7.6) -> None:
        """Horizontal bar chart drawn with vector rectangles."""
        if not pairs:
            return
        vmax = max(v for _, v in pairs) or 1
        bar_x = x + label_w + 1.5
        bar_max = w_mm - label_w - 11
        y = self.get_y()
        for i, (label, count) in enumerate(pairs):
            yy = y + i * (row_h + 1.0)
            self.set_font(self.font_family, "", size)
            self.set_text_color(*TEXT)
            self.set_xy(x, yy)
            self.cell(label_w, row_h, self.T(_fit_text(self, label, label_w - 1)),
                      align="L", new_x="RIGHT", new_y="TOP")
            fill = colors[i % len(colors)] if colors else ACCENT
            if count:
                bw = max(bar_max * count / vmax, 0.4)
                self.set_fill_color(*fill)
                self.rect(bar_x, yy + 1.9, bw, row_h - 3.8, style="F", round_corners=True, corner_radius=0.8)
            self.set_font(self.font_family, "B", size)
            self.set_text_color(*MUTED)
            self.set_xy(bar_x + bar_max + 1, yy)
            self.cell(9, row_h, str(count), align="L", new_x="RIGHT", new_y="TOP")
        self.set_y(y + len(pairs) * (row_h + 1.0) + 1)
        self.set_text_color(*TEXT)

    def two_col_charts(self, left: tuple[str, list], right: tuple[str, list], *,
                       left_colors: list | None = None, right_colors: list | None = None) -> None:
        col_w = (self.epw - 6) / 2
        top = self.get_y()
        for j, (title, pairs, colors) in enumerate([
            (left[0], left[1], left_colors), (right[0], right[1], right_colors)
        ]):
            x = self.l_margin + j * (col_w + 6)
            self.set_xy(x, top)
            self.set_font(self.font_family, "B", 9)
            self.set_text_color(*PRIMARY)
            self.cell(col_w, 5.5, self.T(title), new_x="LMARGIN", new_y="NEXT")
            self.set_y(top + 6.4)
            self.set_x(x)
            self.hbar_chart(pairs, x, col_w, colors=colors)
        max_rows = max(len(left[1]), len(right[1]))
        self.set_y(top + 6.4 + max(max_rows, 1) * 7.0 + 3)
        self.set_text_color(*TEXT)


def _fit_text(pdf: _ReportPDF, s: str, max_w: float) -> str:
    if pdf.get_string_width(s) <= max_w:
        return s
    while s and pdf.get_string_width(s + "…") > max_w:
        s = s[:-1]
    return s + "…"


# ------------------------------------------------------------------ report sections
def _cover(pdf: _ReportPDF, meta: dict, rows: list[dict], model: str, pseudonymized: bool,
           generated: str) -> None:
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, 210, 58, style="F")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 58, 210, 2.2, style="F")

    pdf.set_y(16)
    pdf.set_x(18)
    pdf.set_text_color(*WHITE)
    pdf.set_font(py_font_family(pdf), "B", 20)
    pdf.multi_cell(165, 9.6, pdf.T("Monitoring Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(py_font_family(pdf), "", 12.2)
    pdf.multi_cell(168, 6.6,
                   pdf.T("Gender-Based Cyberviolence & Hate Speech Against Women"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    pdf.set_font(py_font_family(pdf), "", 8.6)
    pdf.set_text_color((214, 205, 236))
    pdf.multi_cell(168, 4.8,
                   pdf.T(f"Public Facebook content · classified locally with an LM Studio model · "
                         f"generated by {TOOL}"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*TEXT)

    pdf.set_y(74)
    pdf.h2("Report summary")
    total = len(rows)
    flagged = [r for r in rows if r.get("is_violation")]
    rate = (len(flagged) / total * 100) if total else 0.0
    post_n = int(meta.get("post_count") or 0)
    com_n = int(meta.get("comment_count") or 0)
    pdf.key_value([
        ("Source page", meta.get("source_url", "n/a")),
        ("Collected (UTC)", _fmt_ts(meta.get("collected_at_utc"))),
        ("Report generated (UTC)", generated),
        ("Items analysed", f"{total}  ({post_n} posts, {com_n} comments)"),
        ("Flagged by the model", f"{len(flagged)}  ({rate:.1f}% of all items)"),
        ("Classification model", model),
        ("Author pseudonymisation", "Enabled" if pseudonymized else "DISABLED — real names included"),
        ("Data protection", "All analysis ran locally; no content was uploaded."),
    ], label_w=58, gap=2.6)

    # status pill
    pdf.ln(4)
    if flagged:
        worst = max(flagged, key=lambda r: SEVERITIES.index(r["severity"]))
        sev = worst["severity"]
        pill_txt = f"Model flagged {len(flagged)} item(s) — highest severity: {sev.upper()}"
        pill_bg, pill_tx = SEV_COLORS[sev], WHITE
    else:
        pill_txt = "No items were flagged by the model"
        pill_bg, pill_tx = SEV_COLORS["none"], WHITE
    pdf.set_x(18)
    pw = min(pdf.get_string_width(pill_txt) + 14, pdf.epw)
    p_y, p_h = pdf.get_y(), 9.5
    pdf.set_fill_color(*pill_bg)
    pdf.rect(18, p_y, pw, p_h, style="F", round_corners=True, corner_radius=3)
    pdf.set_xy(18, p_y + 2)
    pdf.set_font(py_font_family(pdf), "B", 10)
    pdf.set_text_color(*pill_tx)
    pdf.cell(pw, 6, pdf.T(pill_txt), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*TEXT)

    # confidentiality block pinned to the bottom of the cover (no auto-break here)
    pdf.set_auto_page_break(False)
    pdf.set_y(269)
    pdf.shaded_block(
        "CONFIDENTIAL — DOCUMENTATION MATERIAL. This report contains records of alleged online abuse. "
        "Flags below are an automated first-pass indication produced by a local AI model: every item must be "
        "reviewed and confirmed by a human analyst before it is usable as evidence in publications or "
        "complaints.",
        (252, 238, 238), size=8.2, bar=SEV_COLORS["critical"],
    )
    pdf.set_auto_page_break(True, margin=17)


def py_font_family(pdf: FPDF) -> str:
    return pdf.font_family or FONT_FAMILY


def _toc_page(pdf: _ReportPDF, entries: list[tuple[str, int]]) -> None:
    pdf.add_page()
    pdf.set_font(py_font_family(pdf), "B", 15)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 9, pdf.T("Contents"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.9)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y + 1.2, pdf.l_margin + 12, y + 1.2)
    pdf.set_line_width(0.1)
    pdf.set_y(y + 7)
    _draw_toc(pdf, entries)


def _draw_toc(pdf: _ReportPDF, entries: list[tuple[str, int]]) -> None:
    if not entries:
        return
    lh = 7.6 if len(entries) <= 12 else 6.6
    size = 10.3 if len(entries) <= 12 else 9.2
    pdf.set_text_color(*TEXT)
    pdf.set_line_width(0.2)
    y = pdf.get_y() + 2
    for name, page_no in entries:
        pdf.set_font(py_font_family(pdf), "B", size)
        x0 = pdf.l_margin
        page_w = pdf.w - pdf.r_margin - x0
        num = str(page_no)
        num_w = max(pdf.get_string_width(num) + 4, 8)
        avail = page_w - num_w - 3
        pdf.set_y(y)
        pdf.set_x(x0)
        pdf.cell(avail, lh, pdf.T(_fit_text(pdf, name, avail - 3)), new_x="RIGHT", new_y="TOP")
        pdf.set_draw_color(*BORDER)
        pdf.dashed_line(pdf.get_x() + 1.5, y + lh * 0.77, pdf.w - pdf.r_margin - num_w - 2, y + lh * 0.77)
        link = None
        try:
            link = pdf.add_link(page=page_no)
        except Exception:
            pass
        pdf.cell(page_w - avail, lh, num, align="R", link=link, new_x="LMARGIN", new_y="NEXT")
        y += lh + 0.6


def _render_toc(pdf: _ReportPDF, outline: list) -> None:
    pdf.set_text_color(*TEXT)
    y = pdf.get_y() + 2
    pdf.set_line_width(0.2)
    for sec in outline:
        name, page_no, level = sec.name, sec.page_number, getattr(sec, "level", 0)
        indent = 5 * (level or 0)
        pdf.set_font(py_font_family(pdf), "B" if level == 0 else "", 10.3)
        x0 = pdf.l_margin + indent
        page_w = pdf.w - pdf.r_margin - x0
        num = str(page_no)
        num_w = max(pdf.get_string_width(num) + 4, 8)
        avail = page_w - num_w - 3
        pdf.set_y(y)
        pdf.set_x(x0)
        pdf.cell(avail, 7, pdf.T(_fit_text(pdf, name, avail - 3)), new_x="RIGHT", new_y="TOP")
        pdf.set_draw_color(*BORDER)
        pdf.dashed_line(pdf.get_x() + 1.5, y + 5.4, pdf.w - pdf.r_margin - num_w - 2, y + 5.4)
        link = None
        try:
            link = pdf.add_link(page=page_no)
        except Exception:
            pass
        pdf.set_font(py_font_family(pdf), "B", 10.3)
        pdf.cell(page_w - avail, 7, num, align="R", link=link, new_x="LMARGIN", new_y="NEXT")
        y += 8.0


def _executive_summary(pdf: _ReportPDF, rows: list[dict]) -> None:
    total = len(rows)
    flagged = [r for r in rows if r["is_violation"]]
    errors = [r for r in rows if r.get("analysis_error")]
    hi = [r for r in flagged if r["severity"] in ("high", "critical")]
    rate = (len(flagged) / total * 100) if total else 0.0
    uniq = len({r["author"] for r in flagged})
    mean_conf = (sum(r["confidence"] for r in flagged) / len(flagged)) if flagged else 0.0

    pdf.h1("1.", "Executive summary")
    if total == 0:
        pdf.shaded_block("No items were analysed for this report.", BG_LIGHT, bar=FAINT)
        return

    boxes = [
        (str(total), "items analysed", PRIMARY),
        (str(len(flagged)), "flagged by model", ACCENT),
        (f"{rate:.1f}%", "flag share", (54, 118, 124)),
        (str(len(hi)), "high / critical", SEV_COLORS["high"]),
        (str(uniq), "authors involved", (146, 98, 42)),
        (str(len(errors)), "analysis errors", SEV_COLORS["critical"]),
    ]
    bw, bh, gap = (pdf.epw - 2 * 5) / 3, 22, 5
    top = pdf.get_y() + 1
    for i, (v, lab, col) in enumerate(boxes):
        pdf.stat_box(pdf.l_margin + (i % 3) * (bw + gap), top + (i // 3) * (bh + 5), bw, bh, v, lab, col)
    pdf.set_y(top + 2 * (bh + 5) + 3)

    post_n = sum(1 for r in rows if r["kind"] == "post")
    com_n = sum(1 for r in rows if r["kind"] == "comment")
    if flagged:
        worst = max(flagged, key=lambda r: SEVERITIES.index(r["severity"]))
        authors_note = (f", spread over {uniq} distinct author(s)." if uniq > 1
                        else ", all from a single author.")
        summary = (
            f"The model reviewed {post_n} post(s) and {com_n} comment(s) from the source page and flagged "
            f"{len(flagged)} of them ({rate:.1f}%) as gender-based cyberviolence or hate speech against "
            f"women, with a mean confidence of {mean_conf:.2f} / 1.00. The most severe flag is "
            f"{worst['severity'].upper()}{authors_note}"
            + (f" In addition, {len(errors)} item(s) could not be analysed (see Appendix B)."
               if errors else "")
        )
    else:
        summary = (
            f"The model reviewed {post_n} post(s) and {com_n} comment(s) from the source page and none were "
            f"flagged as gender-based cyberviolence or hate speech against women. This does not certify the "
            f"page as safe — dialects, sarcasm and sampling can hide abuse."
            + (f" {len(errors)} item(s) could not be analysed (see Appendix B)." if errors else "")
        )
    pdf.para(summary, size=9.4)
    pdf.ln(3)

    top_cats = Counter(c for r in flagged for c in r["categories"]).most_common(3)
    if top_cats:
        pdf.h2("Most prominent concerns")
        for c, n in top_cats:
            share = (n / len(flagged) * 100) if flagged else 0
            pdf.bullet_one(f"{CAT_LABELS.get(c, c)} — {n} flagged item(s) ({share:.0f}% of flags) · "
                           f"{_txt(CATEGORIES.get(c, ''))}")
        pdf.ln(1.5)

    if errors:
        pdf.ln(1.5)
        pdf.shaded_block(
            f"Quality notice: {len(errors)} item(s) could not be analysed (model/API errors). They are listed "
            "in Appendix B and must not be treated as 'clean'.",
            (255, 248, 234), bar=SEV_COLORS["medium"])


def _methodology(pdf: _ReportPDF, meta: dict, rows: list[dict], model: str, pseudonymized: bool) -> None:
    pdf.h1("2.", "Methodology, scope and limitations")
    pdf.h2("How the content was collected")
    pdf.bullets([
        f"Public content was collected from the page <{meta.get('source_url', 'n/a')}> on "
        f"{_fmt_ts(meta.get('collected_at_utc'))} using a dedicated research account; collection ran in a "
        "headless browser session and captured the visible post and comment texts.",
        "Each item carries a SHA-256 hash of its original text and a collection timestamp, so any excerpt in "
        "this report can be re-verified against the raw JSON/CSV export that accompanies it.",
        f"Author names are {'pseudonymised (irreversible codes derived from the name) in all outputs' if pseudonymized else 'NOT pseudonymised — this file contains real identity data and must be stored securely and minimised before wider use'}.",
    ])
    pdf.h2("How the items were classified")
    pdf.bullets([
        f"Every item was classified by the language model “{model}”, running entirely on the analyst's "
        "machine (LM Studio). No text was transmitted to any cloud service.",
        "The model judged each item against the taxonomy in section 3: whether it targets a woman or women "
        "because of gender, on a scale none → low → medium → high → critical, with a per-item confidence "
        "score, a target group, a language guess and a short verbatim evidence excerpt.",
        "Flagged items are sorted by severity, then by confidence. Statistics and charts in section 4 cover "
        "flagged items only.",
    ])
    pdf.h2("Limitations — read before acting on any finding")
    pdf.bullets([
        "Model output is an automated first-pass flag, not a legal or factual determination. A human must "
        "review every flagged item before it is used in reports, publications, takedown requests or "
        "complaints.",
        "Dialects (esp. Arabic, Tunisian Darija and Arabizi), sarcasm, coded language and missing context "
        "can cause false positives and false negatives. Quotes are excerpts, not complete items.",
        "Sampling stops at the configured post limit and comment threads may be incomplete; the absence of "
        "flags is not proof of absence of GBV.",
    ])
    errs = [r for r in rows if r.get("analysis_error")]
    if errs:
        pdf.ln(1)
        pdf.shaded_block(f"{len(errs)} item(s) failed classification and are listed in Appendix B; they are "
                         "excluded from all statistics.", (255, 248, 234), bar=SEV_COLORS["medium"])


def _taxonomy(pdf: _ReportPDF) -> None:
    pdf.h1("3.", "Classification taxonomy")
    pdf.para("Categories below were used during classification (editable in analyzer.py). Severities: "
             "none = not a violation; then low, medium, high; critical = explicit threats of rape or death, "
             "doxxing intended to enable harm, or sextortion.", size=9)
    pdf.ln(2)
    head = FontFace(emphasis="BOLD", color=WHITE, fill_color=PRIMARY)
    pdf.set_font(py_font_family(pdf), "", 8.3)
    pdf.set_draw_color(*BORDER)
    pdf.set_fill_color(*WHITE)  # ambient fill leaks into non-zebra rows otherwise
    with pdf.table(
        col_widths=(pdf.epw * 0.30, pdf.epw * 0.70),
        text_align=("LEFT", "LEFT"),
        line_height=4.3,
        headings_style=head,
        cell_fill_color=BG_LIGHT,
        cell_fill_mode=TableCellFillMode.ROWS,
    ) as table:
        hr = table.row()
        hr.cell("Category")
        hr.cell("Definition")
        for cid, desc in CATEGORIES.items():
            r = table.row()
            r.cell(pdf.T(CAT_LABELS.get(cid, cid)))
            r.cell(pdf.T(_txt(desc)))


def _at_a_glance(pdf: _ReportPDF, rows: list[dict]) -> None:
    flagged = [r for r in rows if r["is_violation"]]
    pdf.h1("4.", "Findings at a glance")
    if not flagged:
        pdf.shaded_block(
            "The model flagged no items in this dataset, so there is nothing to chart. Appendix A still "
            "lists every item that was analysed.", BG_LIGHT, bar=FAINT)
        return

    pdf.h2("By category of abuse")
    cat_counts = Counter(c for r in flagged for c in r["categories"])
    cats = sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    pdf.hbar_chart([(CAT_LABELS.get(c, c), n) for c, n in cats],
                   pdf.l_margin, pdf.epw, label_w=64, colors=BAR_PALETTE, row_h=6.8, size=8.2)
    pdf.ln(5)

    sev_order = ["critical", "high", "medium", "low"]
    sev_counts = Counter(r["severity"] for r in flagged)
    tgt_counts = Counter(r["target"] for r in flagged)
    lang_counts = Counter(r["language"] for r in flagged)
    kind_counts = Counter(r["kind"] for r in flagged)

    pdf.two_col_charts(
        ("By severity", [(s, sev_counts.get(s, 0)) for s in sev_order]),
        ("By target group", sorted(((TARGET_LABELS.get(t, t), n) for t, n in tgt_counts.items()),
                                   key=lambda kv: -kv[1])),
        left_colors=[SEV_COLORS[s] for s in sev_order],
    )
    pdf.ln(4)
    pdf.two_col_charts(
        ("By language", sorted(((LANG_LABELS.get(l, l), n) for l, n in lang_counts.items()),
                               key=lambda kv: -kv[1])
         or [("n/a", 0)]),
        ("Flagged items by type", [("Posts", kind_counts.get("post", 0)),
                                   ("Comments", kind_counts.get("comment", 0))]),
    )

    pdf.ln(2)
    n_from = len({r["author"] for r in flagged})
    conf_hi = sum(1 for r in flagged if r["confidence"] >= 0.75)
    pdf.para(
        f"Distribution: {len(flagged)} flagged item(s) from {n_from} author(s); {conf_hi} carry a "
        "confidence of 0.75 or higher. Prioritise the severe items (high / critical) for human review — "
        "they follow in section 5.", size=9)


def _finding_card(pdf: _ReportPDF, rank: int, r: dict) -> None:
    sev = r["severity"]
    color = SEV_COLORS[sev]
    text_w = pdf.epw
    size, lh = 8.4, 8.4 * 0.53

    # ---- resolve screenshot (if the item was scraped with one)
    shot_path = (r.get("screenshot") or "").replace("/", os.sep)
    if shot_path and not os.path.isabs(shot_path):
        shot_path = os.path.abspath(shot_path)
    shot_w = shot_h = 0.0
    if shot_path and os.path.isfile(shot_path):
        try:
            with Image.open(shot_path) as im:
                px_w, px_h = im.size
            if px_w and px_h:
                shot_w = text_w
                shot_h = shot_w * px_h / px_w
                max_h = 95.0  # keep cards readable
                if shot_h > max_h:
                    shot_h = max_h
                    shot_w = max_h * px_w / px_h
        except Exception:
            pass

    # ---- measure the card height so it never splits across pages
    cat_s = ", ".join(CAT_LABELS.get(c, c) for c in r["categories"]) or "—"
    ev = _fin(_excerpt(_txt(r["evidence"]), 300))
    ex = _fin(_excerpt(_txt(r["explanation"]), 600))
    h_banner = 9.5
    h_meta = sum(max(pdf.lines_of(v, text_w - 42, size), 5.4) + 1.4 for v in
                 (cat_s, "x", r["author"])) + 4
    h_shot = (6.0 + shot_h + 4.0) if shot_w else 0
    h_evid = (7.5 + pdf.lines_of(ev, text_w - 10, size) + 4.6 + 2.2) if r["evidence"] else 0
    h_expl = (7.0 + pdf.lines_of(ex, text_w - 4, size) + 2.0) if r["explanation"] else 0
    h_info = 8.0
    total_h = h_banner + h_meta + h_shot + h_evid + h_expl + h_info + 8
    pdf.ensure_room(min(total_h, pdf.page_break_trigger - pdf.t_margin))
    pdf.ln(1.5)

    x0, y0 = pdf.l_margin, pdf.get_y()

    # ---- severity banner
    pdf.set_fill_color(*color)
    pdf.rect(x0, y0, text_w, 9, style="F", round_corners=True, corner_radius=1.2)
    pdf.set_xy(x0 + 4, y0 + 1.6)
    pdf.set_font(py_font_family(pdf), "B", 9)
    pdf.set_text_color(*WHITE)
    pdf.cell(text_w * 0.45, 6, pdf.T(f"#{rank:02d}   ·   {r['id']}   ·   {r['kind'].upper()}"),
             align="L", new_x="RIGHT", new_y="TOP")
    pdf.set_x(x0 + 4 + text_w * 0.45)
    pdf.cell(text_w * 0.55 - 4, 6, pdf.T(f"{sev.upper()}   ·   confidence {r['confidence']:.0%}"),
             align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*TEXT)

    # ---- metadata rows (zebra)
    kind_bit = "original post" if r["kind"] == "post" else f"comment on {r.get('parent_id') or '?'}"
    meta_rows = [
        ("Categories", cat_s),
        ("Target / lang / type",
         f"{TARGET_LABELS.get(r['target'], r['target'])} · {LANG_LABELS.get(r['language'], r['language'])} · {kind_bit}"),
        ("Author / posted",
         f"{r['author']}" + (f"  ·  displayed time: {r['time_text']}" if r.get("time_text") else "")),
    ]
    y = y0 + 10.5
    for i, (lab, val) in enumerate(meta_rows):
        bg = BG_SOFT if i % 2 else BG_LIGHT
        vh = pdf.lines_of(val, text_w - 42, size)
        rh = max(vh, 5.4) + 1.6
        pdf.set_fill_color(*bg)
        pdf.rect(x0, y, text_w, rh, style="F")
        pdf.set_y(y + 1.0)
        pdf.set_x(x0 + 3.5)
        pdf.set_font(py_font_family(pdf), "B", size * 0.82)
        pdf.set_text_color(*MUTED)
        pdf.cell(34, lh, pdf.T(lab.upper()), new_x="RIGHT", new_y="TOP")
        pdf.set_font(py_font_family(pdf), "", size)
        pdf.set_text_color(*TEXT)
        pdf.set_x(x0 + 39.5)
        pdf.multi_cell(text_w - 39.5, lh, pdf.T(val), new_x="LMARGIN", new_y="NEXT")
        y += rh
    pdf.set_y(y + 3)

    # ---- screenshot as captured at collection time
    if shot_w:
        if pdf.get_y() + shot_h + 14 > pdf.page_break_trigger:
            pdf.add_page()
        pdf.set_font(pdf.font_family, "B", size)
        pdf.set_text_color(*MUTED)
        pdf.set_x(x0)
        pdf.cell(text_w, lh, pdf.T("SCREENSHOT AS CAPTURED AT COLLECTION"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.2)
        ix = x0 + (text_w - shot_w) / 2
        try:
            pdf.image(shot_path, x=ix, w=shot_w)
        except Exception:
            pass
        pdf.set_font(pdf.font_family, "", 6.8)
        pdf.set_text_color(*FAINT)
        pdf.multi_cell(0, 3.2, pdf.T(f"{r['screenshot']} — visual context only; the SHA-256 below covers the text content."),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*TEXT)
        pdf.ln(2)
    else:
        pdf.ln(0.5)

    # ---- evidence + explanation
    if r["evidence"]:
        pdf.set_font(py_font_family(pdf), "B", size)
        pdf.set_text_color(*color)
        pdf.set_x(x0)
        pdf.cell(text_w, lh, pdf.T("EVIDENCE (verbatim excerpt)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.2)
        pdf.shaded_block(ev, (241, 242, 245), size=size, bar=color)
        pdf.ln(2.2)
    if r["explanation"]:
        pdf.set_font(py_font_family(pdf), "B", size)
        pdf.set_text_color(*MUTED)
        pdf.set_x(x0)
        pdf.cell(text_w, lh, pdf.T("WHY THE MODEL FLAGGED IT"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.9)
        pdf.para(ex, size=size)
        pdf.ln(2.2)

    # ---- provenance footer + review box
    prov = []
    if r.get("permalink"):
        prov.append(f"Permalink: {r['permalink']}")
    prov.append(f"Collected: {_fmt_ts(r.get('collected_at'))}")
    pdf.set_font(py_font_family(pdf), "", size * 0.86)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 3.8, pdf.T("  ·  ".join(prov)), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(py_font_family(pdf), "", 8.6)
    pdf.cell(0, 5,
             pdf.T("Human review:   [ ] pending    [v] verified    [x] rejected    [!] excerpt corrected"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(py_font_family(pdf), "", 6.6)
    pdf.set_text_color(*FAINT)
    pdf.multi_cell(0, 3.2, pdf.T(f"SHA-256: {r.get('sha256', 'n/a')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*TEXT)

    yb = pdf.get_y() + 2.5
    pdf.set_draw_color(*BORDER)
    pdf.set_line_width(0.25)
    pdf.line(x0, yb, pdf.w - pdf.r_margin, yb)
    pdf.set_line_width(0.1)
    pdf.set_y(yb + 5)


def _detailed_findings(pdf: _ReportPDF, rows: list[dict]) -> None:
    flagged = [r for r in rows if r["is_violation"]]
    sev_rank = {s: i for i, s in enumerate(SEVERITIES)}
    flagged.sort(key=lambda r: (-sev_rank[r["severity"]], -r["confidence"]))
    pdf.h1("5.", f"Flagged items — detailed records ({len(flagged)})")
    if not flagged:
        pdf.shaded_block("No items were flagged in this dataset.", BG_LIGHT, bar=FAINT)
        return
    pdf.para(f"{len(flagged)} record(s), most severe first. The banner colour encodes severity "
             "(deep red = critical, orange = high, amber = medium, olive = low). Each record is designed to "
             "be copied or photographed into an incident file as-is: verify the evidence excerpt and the "
             "permalink against the original post, then mark the review line under the record.", size=9)
    pdf.ln(1)
    for rank, r in enumerate(flagged, 1):
        _finding_card(pdf, rank, r)


def _sig_row(pdf: _ReportPDF, label1: str, label2: str, w1: float) -> None:
    w = pdf.epw
    y0 = pdf.get_y()
    pdf.set_draw_color(*MUTED)
    pdf.set_line_width(0.3)
    pdf.set_font(py_font_family(pdf), "", 9)
    pdf.set_text_color(*TEXT)
    pdf.set_x(pdf.l_margin)
    pdf.cell(w1 - 2, 8, pdf.T(label1), new_x="RIGHT", new_y="TOP")
    pdf.line(pdf.l_margin + w1 - 2 + 1, y0 + 6.5, pdf.l_margin + w1 + w * 0.22, y0 + 6.5)
    pdf.set_x(pdf.l_margin + w1 + 1)
    pdf.cell(w - w1 - 2, 8, pdf.T(label2), new_x="LMARGIN", new_y="TOP")
    pdf.line(pdf.w - pdf.r_margin - 32, y0 + 6.5, pdf.w - pdf.r_margin, y0 + 6.5)
    pdf.set_y(y0 + 13)


def _signoff(pdf: _ReportPDF) -> None:
    pdf.h1("9.", "Human review and sign-off")
    pdf.para("Automated flags gain evidentiary value only once reviewed. For each record in section 5, mark "
             "the review line and note any corrections, then sign below to certify the review was performed.",
             size=9.4)
    pdf.ln(3)
    pdf.shaded_block(
        "Review status legend for section 5 records:\n"
        "[ ]  Not reviewed yet      [v]  Verified — matches the cited source      "
        "[x]  Rejected — false positive\n"
        "[!]  Verified but excerpt corrected or context added (state below which record ID was adjusted)",
        BG_LIGHT, bar=ACCENT, size=8.6)
    pdf.ln(5)
    _sig_row(pdf, "Reviewed by (name & role):", "Date:", pdf.epw * 0.60)
    _sig_row(pdf, "Certified by (supervisor):", "Date:", pdf.epw * 0.60)
    pdf.ln(4)
    pdf.set_font(py_font_family(pdf), "B", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, pdf.T("Review notes:"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*BORDER)
    pdf.set_line_width(0.25)
    for _ in range(4):
        pdf.set_y(pdf.get_y() + 9)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_line_width(0.1)
    pdf.set_text_color(*TEXT)


def _gallery(pdf: _ReportPDF, rows: list[dict]) -> None:
    """Appendix C: thumbnail grid of every scraped item that has a screenshot."""
    shots = []
    for r in sorted(rows, key=lambda x: (len(x["id"]), x["id"])):
        p = (r.get("screenshot") or "")
        if not p:
            continue
        fp = os.path.join(os.getcwd(), p)
        if os.path.isfile(fp) and os.path.getsize(fp):
            shots.append((r, fp))
    pdf.h1("C.", "Visual record of collected items")
    if not shots:
        pdf.shaded_block("No screenshots were captured for this run "
                         "(enable “Screenshot each post” before scraping).", BG_LIGHT, bar=FAINT)
        return
    pdf.para(f"Element screenshots captured by the scraper at collection time for {len(shots)} item(s), "
             "in item order. Screenshots are visual context for the human reviewer: they show the content "
             "as it appeared, including images the text export cannot carry. The integrity anchors remain "
             "the SHA-256 text hashes in section A.", size=9)
    pdf.ln(2)

    from PIL import Image as PILImage
    cols, gap = 3, 4.5
    cell_w = (pdf.epw - (cols - 1) * gap) / cols
    max_h = 85.0
    i = 0
    while i < len(shots):
        batch = shots[i:i + cols]
        fits = []
        for r, path in batch:
            try:
                with PILImage.open(path) as im:
                    pw_, ph_ = im.size
            except Exception:
                pw_, ph_ = 0, 0
            if not pw_ or not ph_:
                fits.append(None)
                continue
            w, h = cell_w, cell_w * ph_ / pw_
            if h > max_h:
                h, w = max_h, max_h * pw_ / ph_
            fits.append((r, path, w, h))
        row_h = max((f[3] for f in fits if f), default=0) or 8
        need = 4 + row_h + 9
        if pdf.get_y() + need > pdf.page_break_trigger:
            pdf.add_page()
        top = pdf.get_y()
        for j, f in enumerate(fits):
            if not f:
                continue
            r, path, w, h = f
            cx = pdf.l_margin + j * (cell_w + gap) + (cell_w - w) / 2
            try:
                pdf.image(path, x=cx, y=top, w=w, h=h)
            except Exception:
                pass
            pdf.set_draw_color(*BORDER)
            pdf.set_line_width(0.25)
            pdf.rect(cx, top, w, h)
            pdf.set_line_width(0.1)
        pdf.set_y(top + row_h + 1.6)
        for j, f in enumerate(fits):
            if not f:
                continue
            r = f[0]
            sev_color = SEV_COLORS.get(r["severity"], FAINT)
            cx = pdf.l_margin + j * (cell_w + gap)
            pdf.set_xy(cx, top + row_h + 1.6)
            pdf.set_font(pdf.font_family, "B", 8)
            pdf.set_text_color(*sev_color)
            pdf.cell(9, 4.4, pdf.T(f"{r['id']}"), new_x="RIGHT", new_y="TOP")
            pdf.set_font(pdf.font_family, "", 7.2)
            pdf.set_text_color(*MUTED)
            pdf.cell(cell_w - 9, 4.4, pdf.T(r["kind"]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(top + row_h + 8)
        i += cols


def _appendices(pdf: _ReportPDF, rows: list[dict]) -> None:
    total = len(rows)
    sev_rank = {s: i for i, s in enumerate(SEVERITIES)}
    ordered = sorted(rows, key=lambda r: (0 if r["is_violation"] else 1,
                                          -sev_rank[r["severity"]], r["id"]))

    pdf.h1("A.", "Full item inventory")
    pdf.para(f"All {total} item(s) analysed: flagged items first (by descending severity), then items the "
             "model did not flag. Full texts and hashes are in the companion CSV / JSON exports.", size=9)
    pdf.ln(1.5)
    head = FontFace(emphasis="BOLD", color=WHITE, fill_color=PRIMARY)
    pdf.set_font(py_font_family(pdf), "", 7.6)
    pdf.set_draw_color(*BORDER)
    pdf.set_fill_color(*WHITE)  # ambient fill leaks into non-zebra rows otherwise
    fixed = 12 + 12.5 + 15 + 34 + 10 + 9.5
    with pdf.table(
        col_widths=(12, 12.5, 15, 34, 10, 9.5, pdf.epw - fixed),
        text_align=("LEFT", "LEFT", "LEFT", "LEFT", "CENTER", "CENTER", "LEFT"),
        line_height=3.9,
        headings_style=head,
        cell_fill_color=BG_LIGHT,
        cell_fill_mode=TableCellFillMode.ROWS,
    ) as table:
        hr = table.row()
        for c in ("ID", "Kind", "Severity", "Categories", "Lang", "Conf", "Text excerpt"):
            hr.cell(c)
        for r in ordered:
            row = table.row()
            row.cell(pdf.T(r["id"]))
            row.cell(pdf.T(r["kind"]))
            row.cell(pdf.T(r["severity"]))
            row.cell(pdf.T(", ".join(CAT_LABELS.get(c, c) for c in r["categories"]) or "—"))
            row.cell(pdf.T(r["language"]))
            row.cell(pdf.T(f"{r['confidence']:.2f}"))
            row.cell(pdf.T(_excerpt(_txt(r["text"]), 150)))

    errors = [r for r in rows if r.get("analysis_error")]
    if errors:
        pdf.ln(8)
        pdf.h1("B.", "Analysis failures")
        pdf.para("These items could not be classified (model or API errors). They are NOT counted as clean.",
                 size=9)
        pdf.ln(1.5)
        pdf.set_fill_color(*WHITE)  # ambient fill leaks into non-zebra rows otherwise
        with pdf.table(
            col_widths=(12, 12.5, 62, pdf.epw - 86.5),
            text_align=("LEFT", "LEFT", "LEFT", "LEFT"),
            line_height=3.9,
            headings_style=head,
            cell_fill_color=(255, 246, 244),
            cell_fill_mode=TableCellFillMode.ROWS,
        ) as table:
            hr = table.row()
            for c in ("ID", "Kind", "Error", "Text excerpt"):
                hr.cell(c)
            for r in errors:
                row = table.row()
                row.cell(pdf.T(r["id"]))
                row.cell(pdf.T(r["kind"]))
                row.cell(pdf.T(_excerpt(_txt(r["analysis_error"]), 160)))
                row.cell(pdf.T(_excerpt(_txt(r["text"]), 120)))


# ------------------------------------------------------------------ entry point
def _setup_pdf() -> _ReportPDF:
    fonts = _font_candidates()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf = _ReportPDF(
        title="GBV & Hate-Speech Monitoring Report",
        generated=f"{TOOL} · {now}",
        footer_note="Auto-generated first-pass documentation — human review required before evidentiary use.",
    )
    if fonts.get(""):
        for style, path in fonts.items():
            if path:
                pdf.add_font(FONT_FAMILY, style, path)
        pdf.set_font(FONT_FAMILY, "", 10)
    else:
        pdf.unicode_font = False
        pdf.set_font("helvetica", "", 10)
    try:
        pdf.set_text_shaping(True)
    except Exception:  # uharfbuzz not installed
        pdf.unicode_font = False
    return pdf


def _assemble(pdf: _ReportPDF, meta: dict, rows: list[dict], model: str, pseudonymized: bool,
              generated: str, collect_toc: bool, extra: dict | None = None) -> None:
    pdf._collect_toc = collect_toc
    pdf._toc_entries = [] if collect_toc else getattr(pdf, "_toc_entries", [])
    pdf.add_page()
    _cover(pdf, meta, rows, model, pseudonymized, generated)
    _toc_page(pdf, pdf._toc_entries)
    pdf.chrome = True

    pdf.add_page()
    _executive_summary(pdf, rows)
    pdf.add_page()
    _methodology(pdf, meta, rows, model, pseudonymized)
    pdf.add_page()
    _taxonomy(pdf)
    pdf.add_page()
    _at_a_glance(pdf, rows)
    pdf.add_page()
    _detailed_findings(pdf, rows)
    extra = extra or {}
    if extra.get("osint"):
        pdf.add_page()
        _osint_dossier(pdf, extra["osint"])
    if extra.get("longitudinal") and not extra["longitudinal"].get("first_run"):
        pdf.add_page()
        _longitudinal(pdf, extra["longitudinal"])
    if extra.get("memory") or extra.get("quality") or extra.get("category_quality"):
        pdf.add_page()
        _memory_quality(pdf, extra)
    pdf.add_page()
    _signoff(pdf)
    pdf.add_page()
    _appendices(pdf, rows)
    pdf.add_page()
    _gallery(pdf, rows)


def _osint_dossier(pdf: _ReportPDF, o: dict) -> None:
    """Section 6: automated, local OSINT dossier (clusters, dups, entities, cadence, net)."""
    pdf.h1("6.", "OSINT dossier (automated, local)")
    pdf.para("This dossier is computed by the reporter from the collected dataset with local "
             "models (embeddings, classification): it produces analytical leads for human "
             "investigation — not conclusions. All items cited below are anchored in Appendix A.",
             size=9)
    pdf.ln(2)

    clusters = o.get("clusters") or []
    if clusters:
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Theme clusters (semantic grouping of the corpus)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_fill_color(*WHITE)
        rows = [[str(i + 1), c["label"][:70], str(c["size"])] for i, c in enumerate(clusters[:10])]
        with pdf.table(col_widths=(10, pdf.epw - 24, 14),
                       text_align=("RIGHT", "LEFT", "RIGHT"), line_height=4.4, padding=1.2) as table:
            hr = table.row()
            hr.cell("#")
            hr.cell("Representative opening (cluster label)")
            hr.cell("Items")
            for r in rows:
                tr = table.row()
                for cell in r:
                    tr.cell(cell)
    pdf.ln(2)

    dups = o.get("near_dups") or []
    if dups:
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Possible template reproduction (near-duplicate items)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for d in dups[:8]:
            pdf.shaded_block(f"Item pair {d['a']} ↔ {d['b']} — semantic similarity {d['sim']:.3f}. "
                             "Same-narrative reposting/copy flows: check the two authors for relations.",
                             BG_LIGHT, bar=FAINT)
        pdf.ln(1)

    ents = o.get("entities") or []
    if ents:
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Most frequent named entities (LLM extraction)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_fill_color(*WHITE)
        with pdf.table(col_widths=(24, pdf.epw - 48, 24), text_align=("LEFT", "LEFT", "RIGHT"),
                       line_height=4.4, padding=1.2) as table:
            hr = table.row()
            hr.cell("Type")
            hr.cell("Name")
            hr.cell("Mentions")
            for e in ents[:12]:
                tr = table.row()
                tr.cell(str(e["type"]))
                tr.cell(e["name"])
                tr.cell(str(e["count"]))
    pdf.ln(2)

    cad = o.get("cadence") or {}
    if cad:
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Posting cadence (page-displayed timestamps)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)
        max_n = max(cad.values()) or 1
        bar_w = pdf.epw * 0.62
        for label, n in cad.items():
            y = pdf.get_y()
            pdf.set_x(pdf.l_margin)
            pdf.set_font(pdf.font_family, "", 8.5)
            pdf.set_text_color(*TEXT)
            pdf.cell(26, 5, pdf.T(label[:24]), new_x="RIGHT", new_y="TOP")
            w = bar_w * n / max_n
            pdf.set_fill_color(*PRIMARY if n == max_n else FAINT)
            pdf.rect(pdf.l_margin + 28, y + 0.8, w, 3.8, style="F")
            pdf.set_xy(pdf.l_margin + 30 + w, y)
            pdf.cell(12, 5, pdf.T(str(n)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    amp = o.get("amplifiers") or {}
    if amp:
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Amplification network (commenters → authors)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_fill_color(*WHITE)
        left = amp.get("most_active_commenters", [])
        right = amp.get("most_commented_authors", [])
        with pdf.table(col_widths=(38, 40, 38, 40), line_height=4.4, padding=1.2,
                       text_align=("LEFT", "RIGHT", "LEFT", "RIGHT")) as table:
            hr = table.row()
            hr.cell("Commenter")
            hr.cell("Comments given")
            hr.cell("Most commented")
            hr.cell("Comments received")
            for i in range(max(len(left), len(right))):
                l = left[i] if i < len(left) else {}
                r = right[i] if i < len(right) else {}
                tr = table.row()
                tr.cell(l.get("node", "—"))
                tr.cell(str(l.get("comments_given", "—")))
                tr.cell(r.get("node", "—"))
                tr.cell(str(r.get("comments_received", "—")))
        if amp.get("pagerank_leaders") or amp.get("betweenness_leaders"):
            pdf.ln(1)
            pdf.set_font(pdf.font_family, "", 8.5)
            pdf.set_text_color(*TEXT)
            pdf.multi_cell(0, 4.2, pdf.T("Influence leaders beyond raw volume (PageRank = structural influence; "
                                         "betweenness = gatekeeping between communities; ◉ = entity node):"))
            pdf.ln(1)
            pdf.set_fill_color(*WHITE)
            pr = {p["node"]: p["pagerank"] for p in amp.get("pagerank_leaders", [])}
            bw = {b["node"]: b["betweenness"] for b in amp.get("betweenness_leaders", [])}
            with pdf.table(col_widths=(pdf.epw - 60, 30, 30), line_height=4.4, padding=1.2,
                           text_align=("LEFT", "RIGHT", "RIGHT")) as table:
                hr = table.row()
                hr.cell("Node")
                hr.cell("PageRank")
                hr.cell("Betweenness")
                for node in list(dict.fromkeys(list(pr) + list(bw)))[:8]:
                    tr = table.row()
                    tr.cell(node)
                    tr.cell(str(pr.get(node, "—")))
                    tr.cell(str(bw.get(node, "—")))
        if o.get("graphml_path"):
            pdf.ln(1)
            pdf.set_font(pdf.font_family, "", 7.6)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 4, pdf.T(f"Full graph: {o['graphml_path']} (GraphML — open in Gephi/yEd)."))
        # template fingerprints (copy-paste reproduction across authors)
        tpls = o.get("templates") or []
        if tpls:
            pdf.ln(1)
            pdf.set_font(pdf.font_family, "B", 10)
            pdf.cell(0, 6, pdf.T("Template fingerprint clusters (word-shingle overlap)"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_fill_color(*WHITE)
            with pdf.table(col_widths=(pdf.epw - 108, 16, 26, 28), line_height=4.4, padding=1.2,
                           text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as table:
                hr = table.row()
                hr.cell("Template (opening words)")
                hr.cell("Copies")
                hr.cell("Authors")
                hr.cell("Items")
                for t in tpls[:10]:
                    tr = table.row()
                    tr.cell(t.get("label", "")[:70])
                    tr.cell(str(t.get("size", "")))
                    tr.cell(str(t.get("accounts", "—")))
                    tr.cell(", ".join(t.get("members", [])[:4]))
            pdf.ln(1)
            pdf.set_font(pdf.font_family, "", 7.6)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 4, pdf.T("Distinct-author template groups are the strongest simple coordination "
                                       "signal computable locally; verified byte-identical posts across different "
                                       "watched pages are stronger still.")) 
        cross = o.get("cross_page") or []
        if cross:
            pdf.ln(1.5)
            pdf.set_font(pdf.font_family, "B", 10)
            pdf.cell(0, 6, pdf.T("Identical posts across watched pages"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_fill_color(*WHITE)
            with pdf.table(col_widths=(34, 18, 18, pdf.epw - 70), line_height=4.4, padding=1.2,
                           text_align=("LEFT", "RIGHT", "RIGHT", "LEFT")) as table:
                hr = table.row()
                hr.cell("sha256 (prefix)")
                hr.cell("Pages")
                hr.cell("Items")
                hr.cell("Page labels")
                for x in cross[:10]:
                    tr = table.row()
                    tr.cell(x["sha256"][:18] + "…")
                    tr.cell(str(x["pages"]))
                    tr.cell(str(x["n_items"]))
                    tr.cell(" · ".join(x.get("page_labels", [])))
        burst = o.get("burst")
        if burst:
            pdf.ln(2)
            pdf.shaded_block(f"BURST WARNING: {burst['latest']} flagged items in the newest run vs baseline "
                             f"{burst['baseline_mean']} over {burst['baseline_runs']} prior runs (+{burst['sigma']}σ). "
                             "Sudden coordinated posting waves deserve a manual look.", BG_LIGHT, bar=PRIMARY)
        pdf.ln(1)
    pdf.shaded_block("These leads are analytical hypotheses for the human investigator; they carry "
                     "no evidentiary weight until reviewed (section 9).", BG_LIGHT, bar=FAINT)


def _memory_quality(pdf: _ReportPDF, extra: dict) -> None:
    """Section: verdict memory + human/AI agreement + per-category calibration."""
    pdf.h1("8.", "Verdict memory & review quality")
    mem = extra.get("memory") or {}
    if mem.get("seeded"):
        pdf.para(f"Verdict memory: {mem['seeded']} item(s) inherited a human review decision from an "
                 "identical or near-identical earlier item (cosine ≥ 0.97 over local embeddings). "
                 "These verdicts are stamped 'verdict-memory' with an inherited_from anchor in the case "
                 "dataset — they reduce repeat model calls and keep rulings consistent across recurring posts.",
                 size=9)
        pdf.ln(2)
    q = extra.get("quality") or {}
    if q.get("n"):
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.cell(0, 6, pdf.T("Human–AI agreement on reviewed items (Cohen's kappa)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_fill_color(*WHITE)
        with pdf.table(col_widths=(70, pdf.epw - 70), line_height=4.4, padding=1.2,
                       text_align=("LEFT", "RIGHT")) as table:
            hr = table.row()
            hr.cell("Metric")
            hr.cell("Value")
            for label, val in [
                ("Reviewed items", str(q.get("n", ""))),
                ("Kappa (agreement beyond chance)", str(q.get("kappa", "—"))),
                ("True positives (flagged → confirmed)", str(q.get("true_pos", 0))),
                ("False positives (flagged → rejected)", str(q.get("false_pos", 0))),
                ("False negatives (clean → confirmed)", str(q.get("false_neg", 0))),
                ("True negatives (clean → rejected)", str(q.get("true_neg", 0))),
            ]:
                tr = table.row()
                tr.cell(label)
                tr.cell(val)
        if q.get("kappa") is not None and q["kappa"] < 0.4:
            pdf.ln(1)
            pdf.shaded_block("Kappa below 0.4: treat machine flags as hints only and re-read explanations "
                             "before using them externally.", BG_LIGHT, bar=PRIMARY)
        cq = extra.get("category_quality") or []
        if cq:
            pdf.ln(2)
            pdf.set_font(pdf.font_family, "B", 10)
            pdf.cell(0, 6, pdf.T("Per-category calibration (human precision per label)"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_fill_color(*WHITE)
            with pdf.table(col_widths=(pdf.epw - 110, 37, 36, 37), line_height=4.4, padding=1.2,
                           text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as table:
                hr = table.row()
                hr.cell("Category")
                hr.cell("Confirmed")
                hr.cell("Rejected")
                hr.cell("Human precision")
                for c in cq[:12]:
                    tr = table.row()
                    tr.cell(c["category"])
                    tr.cell(str(c["confirmed"]))
                    tr.cell(str(c["rejected"]))
                    tr.cell(f"{c['human_precision']:.2f}")
        pdf.ln(2)
        pdf.para("Every statistic here is computed from the review decisions recorded in the case dataset "
                 "— the same review decisions that gate external use of flags.", size=8.4)


def _longitudinal(pdf: _ReportPDF, lon: dict) -> None:
    """Section: what changed since the previous run of this page."""
    pdf.h1("7.", "Longitudinal comparison vs previous run")
    pdf.para("Delta against the most recent completed run of the same page. Deltas are the "
             "core of monitoring: they point at what is new, not what was already reviewed.",
             size=9)
    pdf.ln(2)
    cells = [
        ("Items first seen this run", str(lon.get("new_items", 0))),
        ("Flags added since previous run", str(lon.get("new_flags", 0))),
        ("Flags removed (content/re-verdict)", str(lon.get("dropped_flags", 0))),
        ("Severity changes on continuing items", str(lon.get("severity_flips", 0))),
    ]
    pdf.set_fill_color(*WHITE)
    with pdf.table(col_widths=(pdf.epw - 26, 26), line_height=5.2, padding=1.6,
                   text_align=("LEFT", "RIGHT")) as table:
        for k, v in cells:
            tr = table.row()
            tr.cell(k)
            tr.cell(v)
    pdf.ln(2)
    pdf.shaded_block("Read this section together with the flagged-items section above: items "
                     "flagged in earlier runs that were confirmed/rejected by a human are "
                     "listed in the case dataset review log.", BG_LIGHT, bar=FAINT)


def build_pdf_report(meta: dict, rows: list[dict], model: str, pseudonymized: bool,
                     extra: dict | None = None) -> bytes:
    """Build the PDF report and return its bytes. Fully offline.

    Two passes: pass 1 records the physical page of each section, pass 2
    renders the document with a table of contents pointing to those pages.
    """
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pass1 = _setup_pdf()                     # throwaway: collect section pages
    _assemble(pass1, meta, rows, model, pseudonymized, generated, collect_toc=True, extra=extra)

    pass2 = _setup_pdf()
    pass2._toc_entries = pass1._toc_entries
    _assemble(pass2, meta, rows, model, pseudonymized, generated, collect_toc=False, extra=extra)

    return bytes(pass2.output())


def save_pdf_report(meta: dict, rows: list[dict], model: str, pseudonymized: bool, path: str,
                    extra: dict | None = None) -> str:
    """Build the PDF report and write it to `path` (returns the path)."""
    data = build_pdf_report(meta, rows, model, pseudonymized, extra)
    with open(path, "wb") as f:
        f.write(data)
    return path
