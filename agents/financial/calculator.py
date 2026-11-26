"""Moteur de calculs financiers déterministes et contrôles de cohérence (F03 / F05).

Conformément au cahier des charges ACI V1.0 :
- Tous les calculs financiers sont exécutés par des fonctions déterministes pures (jamais par LLM).
- Les unités (UNIT / THOUSAND / MILLION) et exercices sont normalisés et contrôlés.
- Aucune décision automatique d'octroi ou de refus de crédit n'est produite.
"""

from __future__ import annotations

from services.contracts.domain import (
    FinancialMetrics,
    FinancialStatement,
    VerificationResult,
)


def _safe_div(numerator: float, denominator: float, ndigits: int = 4) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, ndigits)


def compute_financial_metrics(
    statement: FinancialStatement,
    requested_amount: float | None = None,
    debt_tolerance_pct: float = 0.01,
) -> FinancialMetrics:
    """Calcule de manière reproductible les ratios financiers et contrôles de cohérence.

    Args:
        statement: États financiers extraits et validés par contrat Pydantic.
        requested_amount: Montant de financement demandé (dans la même devise, en unité de base).
        debt_tolerance_pct: Tolérance relative maximale (défaut 1 %) entre dette au bilan
            et total du tableau détaillé des dettes.
    """
    revenue = round(statement.revenue.normalized_value(), 2)
    ebitda = round(statement.ebitda.normalized_value(), 2)
    net_income = round(statement.net_income.normalized_value(), 2)
    total_debt = round(statement.total_debt.normalized_value(), 2)
    equity = round(statement.equity.normalized_value(), 2)
    cash = round(statement.cash_and_equivalents.normalized_value(), 2)
    current_assets = round(statement.current_assets.normalized_value(), 2)
    current_liabilities = round(statement.current_liabilities.normalized_value(), 2)
    annual_debt_service = round(statement.annual_debt_service.normalized_value(), 2)

    warnings: list[str] = []

    net_debt = round(total_debt - cash, 2)
    free_cash_flow_proxy = round(ebitda - annual_debt_service, 2)

    # Dette / Capitaux propres
    if equity <= 0:
        debt_to_equity = None
        warnings.append("Capitaux propres négatifs ou nuls : ratio Dette/Capitaux non significatif.")
    else:
        debt_to_equity = _safe_div(total_debt, equity)

    # Dette nette / EBITDA
    if ebitda <= 0:
        net_debt_to_ebitda = None
        warnings.append("EBITDA négatif ou nul : ratio Dette nette/EBITDA non significatif.")
    else:
        net_debt_to_ebitda = _safe_div(net_debt, ebitda)

    # Marges
    if revenue <= 0:
        net_margin = None
        ebitda_margin = None
        warnings.append("Chiffre d'affaires nul ou négatif : marges non calculables.")
    else:
        net_margin = _safe_div(net_income, revenue)
        ebitda_margin = _safe_div(ebitda, revenue)

    # Liquidité générale
    if current_liabilities <= 0:
        current_ratio = None
        warnings.append("Passif circulant nul ou négatif : ratio de liquidité non calculable.")
    else:
        current_ratio = _safe_div(current_assets, current_liabilities)
        if current_ratio is not None and current_ratio < 1.0:
            warnings.append(
                f"Liquidité générale inférieure à 1,0 ({current_ratio:.2f}) : tension potentielle de trésorerie."
            )

    # Capacité de remboursement (DSCR)
    if annual_debt_service <= 0:
        dscr = None
        warnings.append("Service annuel de la dette nul ou négatif : DSCR non calculable.")
    else:
        dscr = _safe_div(ebitda, annual_debt_service)
        if dscr is not None and dscr < 1.2:
            warnings.append(
                f"DSCR inférieur au seuil de vigilance 1,20 ({dscr:.2f}) : capacité de remboursement tendue."
            )

    # Ratio montant demandé / CA
    requested_to_revenue = (
        _safe_div(requested_amount, revenue)
        if (requested_amount is not None and revenue > 0)
        else None
    )

    # Rapprochement bilan vs tableau des dettes (Dettes.xlsx)
    debt_reconciliation_diff: float | None = None
    debt_reconciliation_diff_pct: float | None = None
    is_debt_consistent = True

    if statement.debt_schedule_total is not None:
        schedule_debt = round(statement.debt_schedule_total.normalized_value(), 2)
        debt_reconciliation_diff = round(abs(total_debt - schedule_debt), 2)
        if total_debt > 0:
            debt_reconciliation_diff_pct = round(debt_reconciliation_diff / total_debt, 4)
        else:
            debt_reconciliation_diff_pct = 1.0 if debt_reconciliation_diff > 0 else 0.0

        if debt_reconciliation_diff_pct > debt_tolerance_pct:
            is_debt_consistent = False
            warnings.append(
                f"Incohérence entre le bilan ({total_debt:,.2f} {statement.currency.value}) "
                f"et le tableau des dettes ({schedule_debt:,.2f} {statement.currency.value}) : "
                f"écart de {debt_reconciliation_diff_pct * 100:.2f} %."
            )

    return FinancialMetrics(
        case_id=statement.case_id,
        fiscal_year=statement.fiscal_year,
        currency=statement.currency,
        revenue=revenue,
        ebitda=ebitda,
        net_income=net_income,
        total_debt=total_debt,
        equity=equity,
        net_debt=net_debt,
        free_cash_flow_proxy=free_cash_flow_proxy,
        debt_to_equity=debt_to_equity,
        net_debt_to_ebitda=net_debt_to_ebitda,
        net_margin=net_margin,
        ebitda_margin=ebitda_margin,
        current_ratio=current_ratio,
        dscr=dscr,
        requested_to_revenue=requested_to_revenue,
        debt_reconciliation_diff=debt_reconciliation_diff,
        debt_reconciliation_diff_pct=debt_reconciliation_diff_pct,
        is_debt_consistent=is_debt_consistent,
        warnings=warnings,
    )


def verify_statement_and_metrics(
    statement: FinancialStatement,
    metrics: FinancialMetrics,
    min_evidence_confidence: float = 0.75,
) -> VerificationResult:
    """Contrôles bloquants du Verification Agent (F05).

    Échec bloquant + déclenchement HITL si une preuve fiable ou un contrôle obligatoire manque.
    """
    blocking_errors: list[str] = []
    missing_evidence_fields: list[str] = []

    sourced_fields = {
        "revenue": statement.revenue,
        "ebitda": statement.ebitda,
        "net_income": statement.net_income,
        "total_debt": statement.total_debt,
        "equity": statement.equity,
        "cash_and_equivalents": statement.cash_and_equivalents,
        "current_assets": statement.current_assets,
        "current_liabilities": statement.current_liabilities,
        "annual_debt_service": statement.annual_debt_service,
    }
    if statement.debt_schedule_total is not None:
        sourced_fields["debt_schedule_total"] = statement.debt_schedule_total

    for field_name, amt in sourced_fields.items():
        if amt.evidence.confidence < min_evidence_confidence:
            missing_evidence_fields.append(field_name)

    if missing_evidence_fields:
        blocking_errors.append(
            f"Confiance documentaire insuffisante (< {min_evidence_confidence:.2f}) "
            f"pour : {', '.join(missing_evidence_fields)}."
        )

    if not metrics.is_debt_consistent:
        blocking_errors.append("Dette totale incohérente avec les pièces sources.")

    if metrics.debt_to_equity is None:
        blocking_errors.append("Ratio Dette/Capitaux non calculable (capitaux propres <= 0).")

    if metrics.dscr is None:
        blocking_errors.append("DSCR non calculable (service de la dette <= 0).")

    passed = len(blocking_errors) == 0
    requires_hitl = not passed
    suspension_reason = blocking_errors[0] if blocking_errors else None

    return VerificationResult(
        passed=passed,
        blocking_errors=blocking_errors,
        missing_evidence_fields=missing_evidence_fields,
        requires_hitl=requires_hitl,
        suspension_reason=suspension_reason,
    )
