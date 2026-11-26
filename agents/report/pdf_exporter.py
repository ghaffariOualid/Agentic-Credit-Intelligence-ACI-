"""Générateur PDF bancaire professionnel pour CreditReport (F06 — ReportLab)."""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.contracts.domain import CreditCase, CreditReport

ATLAS_SLATE = colors.HexColor("#182534")
TERRACOTTA = colors.HexColor("#B54A23")
SAND_BG = colors.HexColor("#FAF6EE")
SAND_BORDER = colors.HexColor("#E6D7C3")
EMERALD = colors.HexColor("#1F6F54")


def build_credit_report_pdf(case: CreditCase, report: CreditReport) -> bytes:
    """Génère les octets PDF d'un rapport d'analyse versionné et sourcé (F06)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"ACI Rapport {case.case_id} v{report.version}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ACITitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.white,
        spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "ACISub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#E6D7C3"),
    )
    h2_style = ParagraphStyle(
        "ACIH2",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=ATLAS_SLATE,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ACIBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=ATLAS_SLATE,
    )
    disclaimer_style = ParagraphStyle(
        "ACIDisclaimer",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=TERRACOTTA,
        leading=12,
    )

    story = []

    # Bandeau en-tête bancaire Atlas Slate + ligne terre cuite
    header_tbl = Table(
        [
            [
                Paragraph(
                    f"<b>Agentic Credit Intelligence — Rapport d'analyse v{report.version}</b>",
                    title_style,
                )
            ],
            [
                Paragraph(
                    f"Dossier {case.case_id} · {case.company_name} ({case.sector}) · "
                    f"Généré le {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
                    sub_style,
                )
            ],
        ],
        colWidths=[174 * mm],
    )
    header_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ATLAS_SLATE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (0, 0), 10),
                ("BOTTOMPADDING", (0, 1), (0, 1), 10),
                ("LINEBELOW", (0, 1), (-1, 1), 3, TERRACOTTA),
            ]
        )
    )
    story.append(header_tbl)
    story.append(Spacer(1, 6 * mm))

    # 1. Synthèse demande & indicateurs financiers déterministes (F03)
    m = report.financial_metrics
    story.append(Paragraph("1. Indicateurs financiers déterministes (F03)", h2_style))
    metrics_rows = [
        ["Indicateur", "Valeur", "Référence / Contrôle"],
        ["Montant demandé", f"{case.requested_amount:,.2f} {case.currency.value}", f"{case.facility_type} ({case.tenor_months} mois)"],
        ["Chiffre d'affaires", f"{m.revenue:,.2f} {m.currency.value}", f"Exercice {m.fiscal_year}"],
        ["EBITDA", f"{m.ebitda:,.2f} {m.currency.value}", f"Marge EBITDA : {m.ebitda_margin * 100:.1f} %" if m.ebitda_margin is not None else "N/A"],
        ["Résultat net", f"{m.net_income:,.2f} {m.currency.value}", f"Marge nette : {m.net_margin * 100:.1f} %" if m.net_margin is not None else "N/A"],
        ["Dette totale / Capitaux propres", f"{m.debt_to_equity:.2f}" if m.debt_to_equity is not None else "N/A", "Calcul déterministe"],
        ["Capacité de remboursement (DSCR)", f"{m.dscr:.2f}x" if m.dscr is not None else "N/A", "Seuil interne >= 1,20x"],
        ["Liquidité générale", f"{m.current_ratio:.2f}" if m.current_ratio is not None else "N/A", "Seuil >= 1,00"],
        ["Cohérence bilan vs dettes", "Conforme" if m.is_debt_consistent else "Écart détecté", f"Écart : {m.debt_reconciliation_diff or 0:,.2f} {m.currency.value}"],
    ]
    tbl_m = Table(metrics_rows, colWidths=[64 * mm, 50 * mm, 60 * mm])
    tbl_m.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SAND_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), ATLAS_SLATE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, SAND_BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FCF9F5")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(tbl_m)
    story.append(Spacer(1, 5 * mm))

    # 2. Constats de risque sourcés (F04)
    story.append(Paragraph("2. Constats de risque et preuves documentaires (F04)", h2_style))
    for f in report.risk_findings:
        ev_str = "; ".join(f"{e.document_name} ({e.page_or_sheet})" for e in f.evidence)
        story.append(
            Paragraph(
                f"<b>[{f.severity.value.upper()}] {f.category} ({f.code})</b> : {f.statement}<br/>"
                f"<i>Preuve : {ev_str} | Incertitude : {int(f.uncertainty * 100)} %</i>",
                body_style,
            )
        )
        story.append(Spacer(1, 2.5 * mm))

    # 3. Journal de contrôle humain HITL (F07)
    story.append(Paragraph("3. Validation & contrôle humain (F05 / F07)", h2_style))
    verif_status = "VALIDÉ" if report.verification.passed else "BLOQUÉ / EN ATTENTE"
    story.append(Paragraph(f"Statut de vérification obligatoire : <b>{verif_status}</b>", body_style))
    if report.hitl_decisions:
        for d in report.hitl_decisions:
            story.append(
                Paragraph(
                    f"• Analyste <b>{d.reviewer_id}</b> ({d.timestamp.strftime('%Y-%m-%d %H:%M UTC')}) — "
                    f"Action : <b>{d.action.value}</b>. Commentaire : {d.comment or 'Aucun'}. "
                    f"Modifications : {d.field_overrides or 'Aucune'}.",
                    body_style,
                )
            )
    else:
        story.append(Paragraph("• Aucune intervention de correction requise sur ce dossier.", body_style))

    story.append(Spacer(1, 6 * mm))

    # 4. Disclaimer obligatoire (F06)
    disc_tbl = Table([[Paragraph(f"<b>AVERTISSEMENT :</b> {report.disclaimer}", disclaimer_style)]], colWidths=[174 * mm])
    disc_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF1EC")),
                ("BOX", (0, 0), (-1, -1), 0.8, TERRACOTTA),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(disc_tbl)

    doc.build(story)
    return buf.getvalue()
