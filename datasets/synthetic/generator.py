"""Générateur reproductible de dossiers de crédit synthétiques annotés (Lot 1).

Produit :
- Les 3 dossiers de référence des maquettes (CR-0261 Atlas Industrie, CR-0260 Nova Services,
  CR-0259 Rivage Logistique)
- Des dossiers variés couvrant les cas sains, les incohérences bilan/dettes, les cas limites
  (EBITDA négatif, liquidité tendue, unités mixtes kMAD/MMAD) et les faibles confiances OCR.
- La vérité terrain déterministe (FinancialMetrics + VerificationResult) pour l'évaluation.
- Des pièces justificatives synthétiques (Bilan.txt, Resultat.txt, Dettes.xlsx).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from pydantic import BaseModel

from agents.financial.calculator import (
    compute_financial_metrics,
    verify_statement_and_metrics,
)
from services.contracts.domain import (
    CaseStatus,
    CreditCase,
    Currency,
    DocumentRef,
    DocumentStatus,
    DocumentType,
    EvidenceRef,
    FinancialMetrics,
    FinancialStatement,
    SourcedAmount,
    UnitScale,
    VerificationResult,
)


class SyntheticAnnotatedCase(BaseModel):
    """Un dossier synthétique complet avec sa vérité terrain annotée."""

    scenario_type: str
    case: CreditCase
    statement: FinancialStatement
    expected_metrics: FinancialMetrics
    expected_verification: VerificationResult


def _amt(
    value: float,
    doc_id: str,
    doc_name: str,
    page: str,
    excerpt: str,
    *,
    currency: Currency = Currency.MAD,
    scale: UnitScale = UnitScale.UNIT,
    period: str = "FY2025",
    confidence: float = 0.98,
) -> SourcedAmount:
    return SourcedAmount(
        value=value,
        currency=currency,
        scale=scale,
        period=period,
        evidence=EvidenceRef(
            document_id=doc_id,
            document_name=doc_name,
            page_or_sheet=page,
            excerpt=excerpt,
            confidence=confidence,
        ),
    )


def _default_docs(case_id: str, debt_needs_review: bool = False) -> list[DocumentRef]:
    return [
        DocumentRef(
            document_id=f"{case_id}-DOC1",
            filename="Bilan.pdf",
            doc_type=DocumentType.BILAN,
            status=DocumentStatus.VALIDATED,
            page_or_sheet_count=6,
        ),
        DocumentRef(
            document_id=f"{case_id}-DOC2",
            filename="Resultat.pdf",
            doc_type=DocumentType.RESULTAT,
            status=DocumentStatus.VALIDATED,
            page_or_sheet_count=3,
        ),
        DocumentRef(
            document_id=f"{case_id}-DOC3",
            filename="Dettes.xlsx",
            doc_type=DocumentType.DETTES,
            status=DocumentStatus.NEEDS_REVIEW if debt_needs_review else DocumentStatus.VALIDATED,
            page_or_sheet_count=2,
        ),
        DocumentRef(
            document_id=f"{case_id}-DOC4",
            filename="Releves.pdf",
            doc_type=DocumentType.RELEVES,
            status=DocumentStatus.VALIDATED,
            page_or_sheet_count=10,
        ),
    ]


def build_flagship_cases() -> list[SyntheticAnnotatedCase]:
    """Construit les 3 dossiers emblématiques issus des maquettes UI 01 et 02."""
    cases: list[SyntheticAnnotatedCase] = []

    # 1. CR-0261 / Atlas Industrie — Incohérence bilan vs Dettes.xlsx (Maquette 02 & 03)
    c1 = CreditCase(
        case_id="CR-0261",
        company_name="Atlas Industrie",
        sector="Industrie manufacturière",
        facility_type="Financement équipement",
        requested_amount=2_000_000.0,
        currency=Currency.MAD,
        tenor_months=60,
        status=CaseStatus.NEEDS_REVIEW,
        extraction_quality=0.97,
        documents=_default_docs("CR-0261", debt_needs_review=True),
    )
    s1 = FinancialStatement(
        case_id="CR-0261",
        fiscal_year="FY2025",
        currency=Currency.MAD,
        revenue=_amt(12.0, "CR-0261-DOC2", "Resultat.pdf", "p. 2", "Chiffre d'affaires: 12,0 M MAD", scale=UnitScale.MILLION),
        ebitda=_amt(1_950.0, "CR-0261-DOC2", "Resultat.pdf", "p. 2", "EBE / EBITDA: 1 950 kMAD", scale=UnitScale.THOUSAND),
        net_income=_amt(850.0, "CR-0261-DOC2", "Resultat.pdf", "p. 2", "Résultat net: 850 kMAD", scale=UnitScale.THOUSAND),
        total_debt=_amt(3_550_000.0, "CR-0261-DOC1", "Bilan.pdf", "p. 4", "Dettes financières totales: 3 550 000 MAD"),
        equity=_amt(5_000_000.0, "CR-0261-DOC1", "Bilan.pdf", "p. 4", "Capitaux propres: 5 000 000 MAD"),
        cash_and_equivalents=_amt(920_000.0, "CR-0261-DOC4", "Releves.pdf", "p. 8", "Solde trésorerie clôture: 920 000 MAD"),
        current_assets=_amt(4_100_000.0, "CR-0261-DOC1", "Bilan.pdf", "p. 4", "Actif circulant: 4 100 000 MAD"),
        current_liabilities=_amt(2_950_000.0, "CR-0261-DOC1", "Bilan.pdf", "p. 4", "Passif circulant: 2 950 000 MAD"),
        annual_debt_service=_amt(980_000.0, "CR-0261-DOC3", "Dettes.xlsx", "T1", "Échéances annuelles: 980 000 MAD"),
        # Écart volontaire : 3 920 000 vs 3 550 000 au bilan (+10,42 %)
        debt_schedule_total=_amt(3_920_000.0, "CR-0261-DOC3", "Dettes.xlsx", "T1", "Total tableau dettes: 3 920 000 MAD"),
    )
    m1 = compute_financial_metrics(s1, requested_amount=c1.requested_amount)
    v1 = verify_statement_and_metrics(s1, m1)
    cases.append(
        SyntheticAnnotatedCase(
            scenario_type="debt_reconciliation_mismatch",
            case=c1,
            statement=s1,
            expected_metrics=m1,
            expected_verification=v1,
        )
    )

    # 2. CR-0260 / Nova Services — Dossier sain et cohérent
    c2 = CreditCase(
        case_id="CR-0260",
        company_name="Nova Services",
        sector="Services B2B",
        facility_type="Crédit de trésorerie",
        requested_amount=750_000.0,
        currency=Currency.MAD,
        tenor_months=36,
        status=CaseStatus.COMPLETED,
        extraction_quality=0.98,
        documents=_default_docs("CR-0260"),
    )
    s2 = FinancialStatement(
        case_id="CR-0260",
        fiscal_year="FY2025",
        currency=Currency.MAD,
        revenue=_amt(8_400_000.0, "CR-0260-DOC2", "Resultat.pdf", "p. 2", "Chiffre d'affaires: 8 400 000 MAD"),
        ebitda=_amt(1_680_000.0, "CR-0260-DOC2", "Resultat.pdf", "p. 2", "EBITDA: 1 680 000 MAD"),
        net_income=_amt(920_000.0, "CR-0260-DOC2", "Resultat.pdf", "p. 2", "Résultat net: 920 000 MAD"),
        total_debt=_amt(1_400_000.0, "CR-0260-DOC1", "Bilan.pdf", "p. 3", "Dettes financières: 1 400 000 MAD"),
        equity=_amt(3_500_000.0, "CR-0260-DOC1", "Bilan.pdf", "p. 3", "Capitaux propres: 3 500 000 MAD"),
        cash_and_equivalents=_amt(1_100_000.0, "CR-0260-DOC4", "Releves.pdf", "p. 5", "Trésorerie: 1 100 000 MAD"),
        current_assets=_amt(3_200_000.0, "CR-0260-DOC1", "Bilan.pdf", "p. 3", "Actif circulant: 3 200 000 MAD"),
        current_liabilities=_amt(1_600_000.0, "CR-0260-DOC1", "Bilan.pdf", "p. 3", "Passif circulant: 1 600 000 MAD"),
        annual_debt_service=_amt(480_000.0, "CR-0260-DOC3", "Dettes.xlsx", "T1", "Service annuel dette: 480 000 MAD"),
        debt_schedule_total=_amt(1_400_000.0, "CR-0260-DOC3", "Dettes.xlsx", "T1", "Total dettes: 1 400 000 MAD"),
    )
    m2 = compute_financial_metrics(s2, requested_amount=c2.requested_amount)
    v2 = verify_statement_and_metrics(s2, m2)
    cases.append(
        SyntheticAnnotatedCase(
            scenario_type="healthy_standard",
            case=c2,
            statement=s2,
            expected_metrics=m2,
            expected_verification=v2,
        )
    )

    # 3. CR-0259 / Rivage Logistique — Liquidité tendue & faible confiance OCR sur un champ
    c3 = CreditCase(
        case_id="CR-0259",
        company_name="Rivage Logistique",
        sector="Transport & Logistique",
        facility_type="Financement flotte",
        requested_amount=1_250_000.0,
        currency=Currency.MAD,
        tenor_months=48,
        status=CaseStatus.ANALYZING,
        extraction_quality=0.89,
        documents=_default_docs("CR-0259"),
    )
    s3 = FinancialStatement(
        case_id="CR-0259",
        fiscal_year="FY2025",
        currency=Currency.MAD,
        revenue=_amt(15_200_000.0, "CR-0259-DOC2", "Resultat.pdf", "p. 2", "CA: 15 200 000 MAD"),
        ebitda=_amt(1_850_000.0, "CR-0259-DOC2", "Resultat.pdf", "p. 2", "EBITDA: 1 850 000 MAD"),
        net_income=_amt(310_000.0, "CR-0259-DOC2", "Resultat.pdf", "p. 2", "Résultat net: 310 000 MAD"),
        total_debt=_amt(6_800_000.0, "CR-0259-DOC1", "Bilan.pdf", "p. 4", "Dettes financières: 6 800 000 MAD"),
        equity=_amt(4_200_000.0, "CR-0259-DOC1", "Bilan.pdf", "p. 4", "Capitaux propres: 4 200 000 MAD"),
        cash_and_equivalents=_amt(450_000.0, "CR-0259-DOC4", "Releves.pdf", "p. 9", "Trésorerie: 450 000 MAD", confidence=0.62),
        current_assets=_amt(2_800_000.0, "CR-0259-DOC1", "Bilan.pdf", "p. 4", "Actif circulant: 2 800 000 MAD"),
        current_liabilities=_amt(3_300_000.0, "CR-0259-DOC1", "Bilan.pdf", "p. 4", "Passif circulant: 3 300 000 MAD"),
        annual_debt_service=_amt(1_700_000.0, "CR-0259-DOC3", "Dettes.xlsx", "T1", "Service dette: 1 700 000 MAD"),
        debt_schedule_total=_amt(6_800_000.0, "CR-0259-DOC3", "Dettes.xlsx", "T1", "Total dettes: 6 800 000 MAD"),
    )
    m3 = compute_financial_metrics(s3, requested_amount=c3.requested_amount)
    v3 = verify_statement_and_metrics(s3, m3)
    cases.append(
        SyntheticAnnotatedCase(
            scenario_type="low_ocr_confidence_and_tight_liquidity",
            case=c3,
            statement=s3,
            expected_metrics=m3,
            expected_verification=v3,
        )
    )

    return cases


def generate_synthetic_corpus(
    extra_count: int = 12,
    seed: int = 42,
) -> list[SyntheticAnnotatedCase]:
    """Génère un corpus annoté déterministe (3 dossiers phares + `extra_count` dossiers variés)."""
    rng = random.Random(seed)
    corpus = build_flagship_cases()

    sectors = [
        "Agroalimentaire",
        "BTP & Matériaux",
        "Textile & Confection",
        "Énergie & Environnement",
        "Distribution spécialisée",
        "Chimie & Plasturgie",
    ]
    scenarios = [
        "healthy_standard",
        "healthy_mixed_scales",
        "debt_reconciliation_mismatch",
        "tight_dscr_and_liquidity",
        "negative_ebitda_edge_case",
        "low_ocr_confidence",
    ]

    for idx in range(extra_count):
        case_num = 270 + idx
        case_id = f"CR-{case_num:04d}"
        scenario = scenarios[idx % len(scenarios)]
        sector = sectors[idx % len(sectors)]
        company_name = f"{sector.split()[0]} Maghreb #{idx + 1}"

        revenue = round(rng.uniform(5_000_000, 45_000_000), 2)
        ebitda_ratio = -0.04 if scenario == "negative_ebitda_edge_case" else rng.uniform(0.10, 0.24)
        ebitda = round(revenue * ebitda_ratio, 2)
        net_income = round(ebitda * rng.uniform(0.35, 0.60), 2)
        equity = round(revenue * rng.uniform(0.25, 0.55), 2)
        total_debt = round(equity * rng.uniform(0.35, 1.45), 2)
        cash = round(revenue * rng.uniform(0.04, 0.14), 2)
        current_assets = round(revenue * rng.uniform(0.22, 0.40), 2)
        current_liabilities = (
            round(current_assets * rng.uniform(1.10, 1.35), 2)
            if scenario == "tight_dscr_and_liquidity"
            else round(current_assets * rng.uniform(0.55, 0.85), 2)
        )
        annual_debt_service = (
            round(max(abs(ebitda), 500_000) * rng.uniform(0.90, 1.15), 2)
            if scenario == "tight_dscr_and_liquidity"
            else round(total_debt * rng.uniform(0.18, 0.28), 2)
        )
        schedule_debt = (
            round(total_debt * rng.uniform(1.06, 1.18), 2)
            if scenario == "debt_reconciliation_mismatch"
            else total_debt
        )
        requested_amount = round(revenue * rng.uniform(0.08, 0.22), 2)

        scale = UnitScale.THOUSAND if scenario == "healthy_mixed_scales" else UnitScale.UNIT
        div = int(scale.value)
        conf_debt = 0.61 if scenario == "low_ocr_confidence" else 0.96

        case = CreditCase(
            case_id=case_id,
            company_name=company_name,
            sector=sector,
            facility_type="Financement croissance",
            requested_amount=requested_amount,
            currency=Currency.MAD,
            tenor_months=rng.choice([36, 48, 60, 84]),
            status=CaseStatus.DRAFT,
            extraction_quality=0.96 if conf_debt > 0.75 else 0.84,
            documents=_default_docs(
                case_id, debt_needs_review=(scenario == "debt_reconciliation_mismatch")
            ),
        )
        stmt = FinancialStatement(
            case_id=case_id,
            fiscal_year="FY2025",
            currency=Currency.MAD,
            revenue=_amt(round(revenue / div, 4), f"{case_id}-DOC2", "Resultat.pdf", "p. 2", f"CA: {revenue:,.2f}", scale=scale),
            ebitda=_amt(round(ebitda / div, 4), f"{case_id}-DOC2", "Resultat.pdf", "p. 2", f"EBITDA: {ebitda:,.2f}", scale=scale),
            net_income=_amt(round(net_income / div, 4), f"{case_id}-DOC2", "Resultat.pdf", "p. 2", f"RN: {net_income:,.2f}", scale=scale),
            total_debt=_amt(round(total_debt / div, 4), f"{case_id}-DOC1", "Bilan.pdf", "p. 4", f"Dette: {total_debt:,.2f}", scale=scale, confidence=conf_debt),
            equity=_amt(round(equity / div, 4), f"{case_id}-DOC1", "Bilan.pdf", "p. 4", f"CP: {equity:,.2f}", scale=scale),
            cash_and_equivalents=_amt(round(cash / div, 4), f"{case_id}-DOC4", "Releves.pdf", "p. 8", f"Cash: {cash:,.2f}", scale=scale),
            current_assets=_amt(round(current_assets / div, 4), f"{case_id}-DOC1", "Bilan.pdf", "p. 4", f"AC: {current_assets:,.2f}", scale=scale),
            current_liabilities=_amt(round(current_liabilities / div, 4), f"{case_id}-DOC1", "Bilan.pdf", "p. 4", f"PC: {current_liabilities:,.2f}", scale=scale),
            annual_debt_service=_amt(round(annual_debt_service / div, 4), f"{case_id}-DOC3", "Dettes.xlsx", "T1", f"SD: {annual_debt_service:,.2f}", scale=scale),
            debt_schedule_total=_amt(round(schedule_debt / div, 4), f"{case_id}-DOC3", "Dettes.xlsx", "T1", f"Total dettes: {schedule_debt:,.2f}", scale=scale),
        )
        metrics = compute_financial_metrics(stmt, requested_amount=requested_amount)
        verif = verify_statement_and_metrics(stmt, metrics)

        corpus.append(
            SyntheticAnnotatedCase(
                scenario_type=scenario,
                case=case,
                statement=stmt,
                expected_metrics=metrics,
                expected_verification=verif,
            )
        )

    return corpus


def export_case_documents(annotated: SyntheticAnnotatedCase, out_dir: Path) -> None:
    """Exporte des pièces justificatives synthétiques (Bilan.txt, Resultat.txt, Dettes.xlsx)."""
    case_dir = out_dir / annotated.case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    m = annotated.expected_metrics
    s = annotated.statement

    (case_dir / "Bilan.txt").write_text(
        f"BILAN SYNTHÉTIQUE — {annotated.case.company_name} ({annotated.case.case_id})\n"
        f"Exercice : {s.fiscal_year} | Devise : {s.currency.value}\n"
        f"Page 4\n"
        f"Capitaux propres : {m.equity:,.2f} {s.currency.value}\n"
        f"Dettes financières totales : {m.total_debt:,.2f} {s.currency.value}\n"
        f"Actif circulant : {s.current_assets.normalized_value():,.2f} {s.currency.value}\n"
        f"Passif circulant : {s.current_liabilities.normalized_value():,.2f} {s.currency.value}\n",
        encoding="utf-8",
    )

    (case_dir / "Resultat.txt").write_text(
        f"COMPTE DE RÉSULTAT SYNTHÉTIQUE — {annotated.case.company_name} ({annotated.case.case_id})\n"
        f"Exercice : {s.fiscal_year} | Devise : {s.currency.value}\n"
        f"Page 2\n"
        f"Chiffre d'affaires : {m.revenue:,.2f} {s.currency.value}\n"
        f"EBITDA : {m.ebitda:,.2f} {s.currency.value}\n"
        f"Résultat net : {m.net_income:,.2f} {s.currency.value}\n",
        encoding="utf-8",
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "T1"
    ws.append(["Ligne de dette", "Établissement", "Devise", "Capital restant dû", "Échéance annuelle"])
    schedule_total = (
        s.debt_schedule_total.normalized_value()
        if s.debt_schedule_total is not None
        else m.total_debt
    )
    annual_service = s.annual_debt_service.normalized_value()
    part1 = round(schedule_total * 0.65, 2)
    part2 = round(schedule_total - part1, 2)
    ws.append(["Crédit équipement LT", "Banque A", s.currency.value, part1, round(annual_service * 0.65, 2)])
    ws.append(["Crédit moyen terme", "Banque B", s.currency.value, part2, round(annual_service * 0.35, 2)])
    ws.append(["TOTAL", "", s.currency.value, schedule_total, annual_service])
    wb.save(case_dir / "Dettes.xlsx")


def write_corpus_artifacts(
    output_dir: Path | None = None,
    extra_count: int = 12,
    seed: int = 42,
) -> Path:
    """Génère et écrit `corpus.json` ainsi que les documents des 3 dossiers phares."""
    target_dir = output_dir or (Path(__file__).resolve().parent / "generated")
    target_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_synthetic_corpus(extra_count=extra_count, seed=seed)
    payload: list[dict[str, Any]] = [item.model_dump(mode="json") for item in corpus]

    corpus_path = target_dir / "corpus.json"
    corpus_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for flagship in corpus[:3]:
        export_case_documents(flagship, target_dir / "documents")

    return corpus_path


if __name__ == "__main__":
    path = write_corpus_artifacts()
    print(f"Corpus synthétique généré avec succès : {path}")
