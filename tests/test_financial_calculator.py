"""Tests du moteur de calculs financiers déterministes et des contrôles F03/F05."""

from agents.financial.calculator import (
    compute_financial_metrics,
    verify_statement_and_metrics,
)
from datasets.synthetic.generator import build_flagship_cases
from services.contracts import (
    Currency,
    EvidenceRef,
    FinancialStatement,
    SourcedAmount,
    UnitScale,
)


def test_atlas_industrie_flagship_metrics_and_blocking_verification() -> None:
    atlas = build_flagship_cases()[0]
    assert atlas.case.case_id == "CR-0261"

    metrics = compute_financial_metrics(atlas.statement, requested_amount=atlas.case.requested_amount)
    # Valeurs alignées avec la Maquette UI 02 (page 6)
    assert metrics.revenue == 12_000_000.0
    assert metrics.net_income == 850_000.0
    assert metrics.debt_to_equity == 0.71
    assert metrics.is_debt_consistent is False
    assert metrics.debt_reconciliation_diff == 370_000.0

    verif = verify_statement_and_metrics(atlas.statement, metrics)
    assert verif.passed is False
    assert verif.requires_hitl is True
    assert "Dette totale incohérente" in (verif.suspension_reason or "")


def test_nova_services_healthy_case_passes_verification() -> None:
    nova = build_flagship_cases()[1]
    assert nova.case.case_id == "CR-0260"

    metrics = compute_financial_metrics(nova.statement, requested_amount=nova.case.requested_amount)
    assert metrics.is_debt_consistent is True
    assert metrics.debt_to_equity == 0.4
    assert metrics.current_ratio == 2.0
    assert metrics.dscr == 3.5

    verif = verify_statement_and_metrics(nova.statement, metrics)
    assert verif.passed is True
    assert verif.requires_hitl is False
    assert verif.blocking_errors == []


def test_negative_equity_and_negative_ebitda_edge_cases() -> None:
    ev = EvidenceRef(document_id="D1", document_name="Bilan.pdf", page_or_sheet="p. 1", confidence=0.95)
    base = dict(currency=Currency.MAD, scale=UnitScale.UNIT, period="FY2025", evidence=ev)
    stmt = FinancialStatement(
        case_id="CR-EDGE",
        fiscal_year="FY2025",
        currency=Currency.MAD,
        revenue=SourcedAmount(value=5_000_000, **base),
        ebitda=SourcedAmount(value=-250_000, **base),
        net_income=SourcedAmount(value=-600_000, **base),
        total_debt=SourcedAmount(value=3_000_000, **base),
        equity=SourcedAmount(value=-100_000, **base),
        cash_and_equivalents=SourcedAmount(value=150_000, **base),
        current_assets=SourcedAmount(value=900_000, **base),
        current_liabilities=SourcedAmount(value=1_500_000, **base),
        annual_debt_service=SourcedAmount(value=500_000, **base),
    )

    metrics = compute_financial_metrics(stmt)
    assert metrics.debt_to_equity is None
    assert metrics.net_debt_to_ebitda is None
    assert metrics.current_ratio == 0.6
    assert metrics.dscr == -0.5
    assert len(metrics.warnings) >= 3

    verif = verify_statement_and_metrics(stmt, metrics)
    assert verif.passed is False
    assert verif.requires_hitl is True
