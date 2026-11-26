"""Multi-page PDF for a VP of Sales: cover, one page per chart (image + how to read + insight + action),
decay simulation, top priority accounts, scoring rules. Built with reportlab and in-memory images only."""
import io
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

import scoring

INK, SLATE, ACCENT, LIGHT = (
    colors.HexColor("#0F172A"),
    colors.HexColor("#64748B"),
    colors.HexColor("#2563EB"),
    colors.HexColor("#EFF6FF"),
)
_REPLACE = {
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2192": "->",
    "\u2011": "-",
    "\u2010": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u202f": " ",
}


def safe(text) -> str:
    """The built-in PDF fonts only cover Windows-1252, so swap anything outside it before drawing."""
    text = str(text)
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return escape(text.encode("cp1252", "replace").decode("cp1252"))


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            textColor=INK,
            alignment=0,
            spaceAfter=4,
        ),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=11, textColor=SLATE, leading=15),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=ACCENT,
            spaceBefore=8,
            spaceAfter=2,
        ),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=10.5, leading=15, textColor=INK),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=8.5, leading=11, textColor=SLATE),
        "bullet": ParagraphStyle(
            "bu",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            leftIndent=12,
            bulletIndent=0,
            textColor=INK,
        ),
    }


def _img(png, width):
    w, h = PILImage.open(io.BytesIO(png)).size
    return Image(io.BytesIO(png), width=width, height=width * h / w)


def _table(rows, col_widths, header=True, align_right_from=1):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def _fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.1f}" if v != int(v) or abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def build_pdf(report, meta, ai_texts=None) -> bytes:
    """report = analytics.build_report output. ai_texts = optional {chart_key: {'insight':..., 'action':...}}."""
    ai_texts = ai_texts or {}
    st_ = _styles()
    buf = io.BytesIO()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SLATE)
        canvas.drawString(18 * mm, 10 * mm, safe(f"Lead scoring report  |  {meta['label']}  |  generated {generated}"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="B2B Lead Scoring Report",
        author="Lead Scoring Studio",
    )
    doc.addPageTemplates(
        [PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=footer)]
    )
    W = doc.width
    story = []

    # ---- cover
    k = report["kpis"]
    story += [
        Paragraph("Lead Scoring Report", st_["title"]),
        Paragraph(
            safe(
                f"{meta['label']}  |  evaluation date {pd.Timestamp(meta['eval_date']).date()}  |  "
                f"email decay {'on, %d-day half-life' % meta['half_life'] if meta['decay'] else 'off'}"
            ),
            st_["sub"],
        ),
        Spacer(1, 8 * mm),
    ]
    kp = [
        ["Employer accounts", f"{k['employers']:,}", "Priority MQLs", f"{k['priority']:,}"],
        ["MQLs", f"{k['mql']:,}", "Average score", f"{k['avg_score']:.1f}"],
    ]
    if "sqls" in k:
        kp.append(["SQLs", f"{k['sqls']:,}", "Overall SQL rate", f"{k['sql_rate']:.1f}%"])
    t = Table(kp, colWidths=[W * 0.28, W * 0.22, W * 0.28, W * 0.22])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("TEXTCOLOR", (0, 0), (0, -1), SLATE),
                ("TEXTCOLOR", (2, 0), (2, -1), SLATE),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.white),
            ]
        )
    )
    story += [t, Spacer(1, 7 * mm), Paragraph("Key takeaways", st_["h1"])]
    for c in report["charts"]:
        if c["key"] in ("tiers", "sql_rate", "score_hist", "renewal", "decay"):
            text = ai_texts.get(c["key"], {}).get("insight", c["insight"])
            story.append(Paragraph(safe(text), st_["bullet"], bulletText="\u2022"))
            story.append(Spacer(1, 1.5 * mm))
    for w in report["warnings"]:
        story.append(Paragraph(safe("Note: " + w), st_["small"]))
    story.append(PageBreak())

    # ---- one page per chart
    for c in report["charts"]:
        txt = ai_texts.get(c["key"], {})
        story += [
            Paragraph(safe(c["title"]), st_["h1"]),
            _img(c["png"], W),
            Spacer(1, 2 * mm),
            Paragraph("How to read this chart", st_["h2"]),
            Paragraph(safe(c["how"]), st_["body"]),
            Paragraph("Insight", st_["h2"]),
            Paragraph(safe(txt.get("insight", c["insight"])), st_["body"]),
            Paragraph("Recommended action", st_["h2"]),
            Paragraph(safe(txt.get("action", c["action"])), st_["body"]),
        ]
        if c["key"] == "decay":
            sim = report["simulation"]
            rows = [list(sim.columns)] + [
                [safe(v) if isinstance(v, str) else _fmt(v) for v in r] for r in sim.itertuples(index=False)
            ]
            story += [
                Spacer(1, 4 * mm),
                Paragraph(safe(report["sim_sentence"]), st_["body"]),
                Spacer(1, 2 * mm),
                _table(rows, [W * 0.4, W * 0.2, W * 0.2, W * 0.2]),
            ]
        story.append(PageBreak())

    # ---- top priority accounts
    top = report["top10"]
    story += [
        Paragraph("Top 10 Priority MQL accounts", st_["h1"]),
        Paragraph("Highest composite score first, ties broken by estimated premium.", st_["small"]),
        Spacer(1, 3 * mm),
    ]
    if len(top):
        cols = {
            "account_id": "Account",
            "est_premium": "Premium ($)",
            "total_score": "Score",
            "account_engagement": "Engagement",
            "engagement": "Engagement",
            "days_to_renewal": "Days to renewal",
            "archetype": "Archetype",
            "is_sql": "SQL",
        }
        keep_cols = [c for c in cols if c in top.columns]
        if keep_cols:
            sub_top = top[keep_cols]
            rows = [[cols[c] for c in keep_cols]] + [
                [_fmt(v) if not isinstance(v, str) else safe(v) for v in r]
                for r in sub_top.itertuples(index=False)
            ]
            story.append(_table(rows, [W / len(keep_cols)] * len(keep_cols), align_right_from=1))
        else:
            story.append(Paragraph("Matching columns not available for Top 10 table.", st_["body"]))
    else:
        story.append(Paragraph("No Priority MQL accounts in this data.", st_["body"]))
    story.append(PageBreak())

    # ---- rules and caveats
    story += [
        Paragraph("How the scores are calculated", st_["h1"]),
        Paragraph(
            "Every number in this report is calculated in Python from the data. Rules marked 'assumed' are choices made "
            "where the model documentation was silent; review them before relying on the results.",
            st_["small"],
        ),
        Spacer(1, 3 * mm),
    ]
    rows = [["Area", "Rule", "Status"]] + [[safe(a), safe(r), safe(s)] for a, r, s in scoring.RULES_TABLE]
    rows = [[Paragraph(x, st_["small"]) if i else x for x in row] for i, row in enumerate(rows)]
    story += [
        _table(rows, [W * 0.22, W * 0.62, W * 0.16], align_right_from=9),
        Spacer(1, 5 * mm),
        Paragraph("Caveats", st_["h2"]),
        Paragraph(
            safe(
                "Scores, tiers and archetypes are rule-based signals of fit and engagement. They do not guarantee conversion or "
                "revenue. "
                + (
                    "This report was built from synthetic data. "
                    if "synthetic" in meta["label"].lower()
                    else "This report was built from uploaded data. "
                )
                + "Insight text is computed from the numbers shown."
            ),
            st_["body"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()
