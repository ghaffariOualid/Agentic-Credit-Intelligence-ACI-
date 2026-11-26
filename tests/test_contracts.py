"""Tests des contrats Pydantic métier et RL (F02, F04, F06, Page 12)."""

import pytest
from pydantic import ValidationError

from services.contracts import (
    Currency,
    EvidenceRef,
    FinancialStatement,
    ModelTier,
    RiskFinding,
    RiskSeverity,
    RLTransition,
    RouterAction,
    RouterState,
    SourcedAmount,
    UnitScale,
)


def _ev(name: str = "Bilan.pdf", page: str = "p. 4") -> EvidenceRef:
    return EvidenceRef(document_id="DOC-1", document_name=name, page_or_sheet=page)


def test_sourced_amount_normalization() -> None:
    amt_m = SourcedAmount(
        value=12.0, currency=Currency.MAD, scale=UnitScale.MILLION, period="FY2025", evidence=_ev()
    )
    amt_k = SourcedAmount(
        value=850.0, currency=Currency.MAD, scale=UnitScale.THOUSAND, period="FY2025", evidence=_ev()
    )
    assert amt_m.normalized_value() == 12_000_000.0
    assert amt_k.normalized_value() == 850_000.0


def test_financial_statement_rejects_currency_mismatch() -> None:
    base = dict(currency=Currency.MAD, scale=UnitScale.UNIT, period="FY2025", evidence=_ev())
    with pytest.raises(ValidationError, match="Devise incohérente"):
        FinancialStatement(
            case_id="CR-ERR-CUR",
            fiscal_year="FY2025",
            currency=Currency.MAD,
            revenue=SourcedAmount(value=10_000_000, currency=Currency.EUR, period="FY2025", evidence=_ev()),
            ebitda=SourcedAmount(value=1_500_000, **base),
            net_income=SourcedAmount(value=700_000, **base),
            total_debt=SourcedAmount(value=2_000_000, **base),
            equity=SourcedAmount(value=4_000_000, **base),
            cash_and_equivalents=SourcedAmount(value=500_000, **base),
            current_assets=SourcedAmount(value=2_000_000, **base),
            current_liabilities=SourcedAmount(value=1_500_000, **base),
            annual_debt_service=SourcedAmount(value=400_000, **base),
        )


def test_financial_statement_rejects_period_mismatch() -> None:
    base = dict(currency=Currency.MAD, scale=UnitScale.UNIT, period="FY2025", evidence=_ev())
    with pytest.raises(ValidationError, match="Exercice incohérent"):
        FinancialStatement(
            case_id="CR-ERR-PER",
            fiscal_year="FY2025",
            currency=Currency.MAD,
            revenue=SourcedAmount(value=10_000_000, currency=Currency.MAD, period="FY2024", evidence=_ev()),
            ebitda=SourcedAmount(value=1_500_000, **base),
            net_income=SourcedAmount(value=700_000, **base),
            total_debt=SourcedAmount(value=2_000_000, **base),
            equity=SourcedAmount(value=4_000_000, **base),
            cash_and_equivalents=SourcedAmount(value=500_000, **base),
            current_assets=SourcedAmount(value=2_000_000, **base),
            current_liabilities=SourcedAmount(value=1_500_000, **base),
            annual_debt_service=SourcedAmount(value=400_000, **base),
        )


def test_risk_finding_requires_at_least_one_evidence() -> None:
    with pytest.raises(ValidationError):
        RiskFinding(
            code="LIQ-01",
            severity=RiskSeverity.HIGH,
            category="Liquidité",
            statement="Liquidité insuffisante.",
            evidence=[],
        )


def test_rl_transition_contract_roundtrip() -> None:
    state = RouterState(
        task_type="financial_extraction",
        context_tokens=4500,
        estimated_complexity=0.65,
        gpu_utilization=0.72,
        vram_utilization=0.68,
        queue_depth=12,
        remaining_budget_usd=0.85,
    )
    action = RouterAction(
        selected_tier=ModelTier.MEDIUM,
        run_verification=True,
        masked_actions=["skip_verification"],
    )
    transition = RLTransition(
        episode_id="EP-001",
        state=state,
        action=action,
        reward=0.51,
        next_state=None,
        done=True,
        model_version="Qwen2.5-7B-Instruct-AWQ",
        policy_version="router-v03",
        quality=0.96,
        latency_ms=2300.0,
        gpu_seconds=2.3,
        masked_actions=["skip_verification"],
        measurement_context={"gpu": "nvidia-t4-16gb"},
    )
    restored = RLTransition.model_validate_json(transition.model_dump_json())
    assert restored == transition
