"""
pdf_export.py  —  Generate PDF chat export using ReportLab Platypus.
Uses BaseDocTemplate + PageTemplate so header/footer are stable.

Place this file in: backend/
No file-path dependencies — receives data as dicts from chat_db.
"""

import io
import re
import time

from reportlab.lib.pagesizes   import A4
from reportlab.lib.units       import cm
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus        import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, HRFlowable, Table, TableStyle,
    KeepTogether, NextPageTemplate, PageBreak
)
from reportlab.platypus.flowables import Flowable

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE  = colors.HexColor("#5b21b6")
LPURPLE = colors.HexColor("#ede9fe")
BORDER  = colors.HexColor("#c4b5fd")
CYAN    = colors.HexColor("#0891b2")
AMBER   = colors.HexColor("#d97706")
GREEN   = colors.HexColor("#059669")
INDIGO  = colors.HexColor("#4338ca")
DARK    = colors.HexColor("#1a1a2e")
GREY    = colors.HexColor("#6b7280")
WHITE   = colors.white
BLACK   = colors.black
LIGHT   = colors.HexColor("#f5f3ff")


# ── Header / Footer drawn on every page ────────────────────────────────────────

def _make_page_drawing(session_title: str):
    """Return an onPage callable that draws header + footer."""
    def draw(canvas, doc):
        W, H = A4
        canvas.saveState()

        # Header bar
        canvas.setFillColor(PURPLE)
        canvas.rect(0, H - 1.3*cm, W, 1.3*cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(1.5*cm, H - 0.88*cm, "Research Paper RAG Assistant")

        canvas.setFont("Helvetica", 9)
        short_title = session_title[:55] + ("…" if len(session_title) > 55 else "")
        canvas.drawRightString(W - 1.5*cm, H - 0.88*cm, short_title)

        # Footer line
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(1.5*cm, 1.2*cm, W - 1.5*cm, 1.2*cm)

        # Footer text
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.5*cm, 0.65*cm,
                          f"Generated {time.strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(W - 1.5*cm, 0.65*cm,
                               f"Page {doc.page}")

        canvas.restoreState()
    return draw


# ── Styles ─────────────────────────────────────────────────────────────────────

def _styles():
    S = {}
    base = getSampleStyleSheet()

    S["doc_title"] = ParagraphStyle(
        "DocTitle",
        fontSize=20, textColor=PURPLE, fontName="Helvetica-Bold",
        spaceAfter=8, spaceBefore=4,
    )
    S["meta_key"] = ParagraphStyle(
        "MetaKey",
        fontSize=9, textColor=PURPLE, fontName="Helvetica-Bold",
    )
    S["meta_val"] = ParagraphStyle(
        "MetaVal",
        fontSize=9, textColor=DARK, fontName="Helvetica",
    )
    S["q_badge"] = ParagraphStyle(
        "QBadge",
        fontSize=9, textColor=GREY, fontName="Helvetica-Bold",
        spaceBefore=4, spaceAfter=2,
    )
    S["query"] = ParagraphStyle(
        "Query",
        fontSize=12, textColor=DARK, fontName="Helvetica-Bold",
        spaceAfter=6, leading=17,
    )
    S["section_bar_rag"]   = ParagraphStyle(
        "BarRAG",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        backColor=PURPLE, borderPad=(3, 8, 3, 8),
        spaceBefore=10, spaceAfter=4,
    )
    S["section_bar_ai"]    = ParagraphStyle(
        "BarAI",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        backColor=CYAN, borderPad=(3, 8, 3, 8),
        spaceBefore=10, spaceAfter=4,
    )
    S["section_bar_extra"] = ParagraphStyle(
        "BarExtra",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        backColor=AMBER, borderPad=(3, 8, 3, 8),
        spaceBefore=10, spaceAfter=4,
    )
    S["section_bar_about"] = ParagraphStyle(
        "BarAbout",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        backColor=GREEN, borderPad=(3, 8, 3, 8),
        spaceBefore=10, spaceAfter=4,
    )
    S["section_bar_cmd"]   = ParagraphStyle(
        "BarCmd",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        backColor=INDIGO, borderPad=(3, 8, 3, 8),
        spaceBefore=10, spaceAfter=4,
    )
    S["body"] = ParagraphStyle(
        "Body",
        fontSize=10, textColor=DARK, fontName="Helvetica",
        leading=16, spaceAfter=5,
    )
    S["sub_heading"] = ParagraphStyle(
        "SubH",
        fontSize=10, textColor=PURPLE, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=3,
    )
    S["source"] = ParagraphStyle(
        "Src",
        fontSize=9, textColor=INDIGO, fontName="Helvetica-Oblique",
        spaceAfter=2,
    )
    S["metric_hdr"] = ParagraphStyle(
        "MHdr",
        fontSize=8, textColor=GREY, fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["metric_val"] = ParagraphStyle(
        "MVal",
        fontSize=13, textColor=PURPLE, fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["metric_sub"] = ParagraphStyle(
        "MSub",
        fontSize=7, textColor=GREY, fontName="Helvetica",
        alignment=TA_CENTER,
    )
    return S


# ── Helpers ────────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Minimal XML escape for ReportLab Paragraph."""
    if not text:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _md_to_rl(text: str) -> str:
    """Convert basic markdown to ReportLab XML tags."""
    if not text:
        return ""
    text = _escape(text)
    # bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # italic *text*
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _para(text, style) -> Paragraph | None:
    t = _md_to_rl(str(text)).strip()
    if not t:
        return None
    return Paragraph(t, style)


def _hr(color=BORDER, thickness=0.5, space=4):
    return HRFlowable(
        width="100%", thickness=thickness,
        color=color, spaceAfter=space, spaceBefore=space
    )


def _bar(label: str, style_name: str, S: dict) -> Paragraph:
    return Paragraph(label, S[style_name])


def _text_block(text: str, S: dict) -> list:
    """Split text into paragraphs, recognising ### headings."""
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append(Spacer(1, 4))
            continue
        if stripped.startswith("###"):
            p = _para(stripped.lstrip("#").strip(), S["sub_heading"])
        elif stripped.startswith("##"):
            p = _para(stripped.lstrip("#").strip(), S["sub_heading"])
        else:
            p = _para(stripped, S["body"])
        if p:
            out.append(p)
    return out


def _sources_block(sources: list, S: dict) -> list:
    if not sources:
        return []
    out = [Spacer(1, 4)]
    for s in sources:
        out.append(Paragraph(
            f"<b>{_escape(s.get('paper_id',''))}</b>"
            f" — {_escape(s.get('section',''))}"
            f" (score: {s.get('score', 0):.3f})",
            S["source"]
        ))
    out.append(Spacer(1, 4))
    return out


def _metrics_block(metrics: dict, S: dict) -> list:
    if not metrics:
        return []

    def _v(key, sub="score"):
        v = metrics.get(key, {})
        if isinstance(v, dict):
            val = v.get(sub)
            return f"{val:.2f}" if isinstance(val, float) else "N/A"
        return "N/A"

    def _l(key):
        v = metrics.get(key, {})
        return v.get("label", "") if isinstance(v, dict) else ""

    rec_v = metrics.get("recall", {})
    rec_s = rec_v.get("score") if isinstance(rec_v, dict) else None
    rec   = f"{rec_s:.2f}" if rec_s is not None else "N/A"

    hdrs  = ["Confidence", "Precision@7", "Recall@7",
             "Semantic Sim.", "Faithfulness"]
    vals  = [_v("confidence"), _v("precision"), rec,
             _v("semantic_similarity"), _v("faithfulness")]
    subs  = [_l("confidence"), "", "", "", _l("faithfulness")]

    def row(items, style):
        return [Paragraph(i, style) for i in items]

    t = Table(
        [row(hdrs, S["metric_hdr"]),
         row(vals, S["metric_val"]),
         row(subs, S["metric_sub"])],
        colWidths=[3.2*cm] * 5
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), LPURPLE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [Spacer(1, 6), t, Spacer(1, 8)]


# ── Per-message flowables ──────────────────────────────────────────────────────

def _message_flowables(msg: dict, idx: int, S: dict) -> list:
    story = []

    # Separator
    story.append(Spacer(1, 8))
    story.append(_hr(color=PURPLE, thickness=1.2, space=6))

    # Q header line
    ts    = msg.get("timestamp", "")
    mode  = msg.get("output_mode", "") or "—"
    paper = msg.get("paper_id",    "") or "All Papers"
    story.append(Paragraph(
        f"<b>Q{idx}</b>  ·  {_escape(ts)}  ·  Mode: {_escape(mode)}"
        f"  ·  Paper: {_escape(paper)}",
        S["q_badge"]
    ))

    # Question text
    p = _para(msg.get("query", ""), S["query"])
    if p:
        story.append(p)
    story.append(Spacer(1, 4))

    # ── Single combined answer ─────────────────────────────────────────────────
    if msg.get("single_answer"):
        story.append(_bar("COMBINED ANSWER", "section_bar_rag", S))
        story.extend(_text_block(msg["single_answer"], S))

    # ── RAG answer ────────────────────────────────────────────────────────────
    if msg.get("rag_answer"):
        story.append(_bar("ANSWER FROM RESEARCH PAPERS (RAG)",
                          "section_bar_rag", S))
        story.extend(_text_block(msg["rag_answer"], S))
        story.extend(_sources_block(msg.get("sources", []), S))

    # ── Command / compare answer ───────────────────────────────────────────────
    if msg.get("command_answer"):
        story.append(_bar("COMMAND RESULT", "section_bar_cmd", S))
        story.extend(_text_block(msg["command_answer"], S))

    # ── AI context ────────────────────────────────────────────────────────────
    if msg.get("ai_context"):
        story.append(_bar("AI ANALYSIS & CONTEXT", "section_bar_ai", S))
        story.extend(_text_block(msg["ai_context"], S))

    # ── Additional knowledge ──────────────────────────────────────────────────
    if msg.get("additional"):
        story.append(_bar("ADDITIONAL KNOWLEDGE & CONNECTIONS",
                          "section_bar_extra", S))
        story.extend(_text_block(msg["additional"], S))

    # ── About paper ───────────────────────────────────────────────────────────
    if msg.get("about_paper"):
        story.append(_bar("ABOUT THIS PAPER", "section_bar_about", S))
        story.extend(_text_block(msg["about_paper"], S))

    # ── Metrics ───────────────────────────────────────────────────────────────
    story.extend(_metrics_block(msg.get("metrics"), S))

    return story


# ── Cover page ─────────────────────────────────────────────────────────────────

def _cover_flowables(session_info: dict, n_messages: int, S: dict) -> list:
    story = []
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(
        _escape(session_info.get("title", "Chat Export")),
        S["doc_title"]
    ))
    story.append(_hr(color=PURPLE, thickness=2, space=8))

    rows = [
        ["Session ID",   session_info.get("session_id", "")[:36]],
        ["Paper Filter", session_info.get("paper_filter", "") or "All Papers"],
        ["Created",      session_info.get("created_at",  "")],
        ["Last Updated", session_info.get("updated_at",  "")],
        ["Messages",     str(n_messages)],
        ["Exported",     time.strftime("%Y-%m-%d %H:%M:%S")],
    ]
    tdata = [
        [Paragraph(k, S["meta_key"]), Paragraph(_escape(str(v)), S["meta_val"])]
        for k, v in rows
    ]
    t = Table(tdata, colWidths=[4*cm, 12.5*cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LPURPLE, WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))
    story.append(_hr(color=PURPLE, thickness=1, space=8))
    return story


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_pdf(session_info: dict, messages: list) -> bytes:
    """
    Build and return PDF bytes for the given session + messages.
    Guaranteed to work with Python 3.12 + ReportLab 4.x.
    """
    buf   = io.BytesIO()
    S     = _styles()
    W, H  = A4
    title = session_info.get("title", "Chat Export")

    on_page = _make_page_drawing(title)

    # Content frame — leaves room for header (top) and footer (bottom)
    content_frame = Frame(
        x1=1.5*cm, y1=1.8*cm,
        width=W - 3.0*cm,
        height=H - 1.3*cm - 1.8*cm,
        id="content"
    )
    page_tpl = PageTemplate(
        id="main",
        frames=[content_frame],
        onPage=on_page
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        pageTemplates=[page_tpl],
        title=title,
        author="Research RAG Assistant",
        leftMargin=0, rightMargin=0,
        topMargin=0,  bottomMargin=0,
    )

    # Build full story
    story = _cover_flowables(session_info, len(messages), S)

    if not messages:
        story.append(Paragraph("No messages in this export.", S["body"]))
    else:
        for i, msg in enumerate(messages, 1):
            story.extend(_message_flowables(msg, i, S))

    doc.build(story)
    return buf.getvalue()