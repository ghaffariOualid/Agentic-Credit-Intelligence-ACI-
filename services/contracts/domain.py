"""Contrats métier Pydantic (F01 à F08) — ACI Spécification V1.0.

Règles structurelles imposées par le cahier des charges :
- Chaque valeur financière extraite est associée à une preuve documentaire (F02).
- Chaque constat de risque cite au moins une preuve et affiche son incertitude (F04).
- Aucune décision automatique d'octroi ou de refus de crédit (F06).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Currency(str, Enum):
    MAD = "MAD"
    EUR = "EUR"
    USD = "USD"


class UnitScale(int, Enum):
    UNIT = 1
    THOUSAND = 1_000
    MILLION = 1_000_000


class DocumentType(str, Enum):
    BILAN = "bilan"
    RESULTAT = "resultat"
    DETTES = "dettes"
    RELEVES = "releves"
    AUTRE = "autre"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class CaseStatus(str, Enum):
    DRAFT = "draft"
    ANALYZING = "analyzing"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKING = "blocking"


class HITLAction(str, Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    REQUEST_CORRECTION = "request_correction"


class EvidenceRef(BaseModel):
    """Référence documentaire vérifiable (ex. Bilan.pdf, p. 4 ou Dettes.xlsx, T1)."""

    document_id: str = Field(..., min_length=1)
    document_name: str = Field(..., min_length=1)
    page_or_sheet: str = Field(..., min_length=1, description="Ex: 'p. 4', 'T1', 'Feuille 1'")
    excerpt: str = Field(default="", description="Extrait textuel ou cellule source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DocumentRef(BaseModel):
    """Pièce justificative associée à un dossier de crédit (F01)."""

    document_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    page_or_sheet_count: int = Field(default=1, ge=1)
    storage_uri: str | None = None


class SourcedAmount(BaseModel):
    """Montant financier strictement typé et relié à sa preuve source (F02)."""

    value: float = Field(..., description="Valeur brute telle qu'extraite dans l'échelle indiquée")
    currency: Currency = Currency.MAD
    scale: UnitScale = UnitScale.UNIT
    period: str = Field(..., min_length=4, description="Exercice comptable, ex: 'FY2025'")
    evidence: EvidenceRef

    def normalized_value(self) -> float:
        """Retourne la valeur normalisée en unités monétaires de base."""
        return float(self.value) * int(self.scale.value)


class FinancialStatement(BaseModel):
    """Données financières extraites d'un dossier (F02)."""

    case_id: str = Field(..., min_length=1)
    fiscal_year: str = Field(..., min_length=4, description="Ex: 'FY2025'")
    currency: Currency = Currency.MAD

    revenue: SourcedAmount = Field(..., description="Chiffre d'affaires")
    ebitda: SourcedAmount = Field(..., description="Excédent brut d'exploitation (EBITDA)")
    net_income: SourcedAmount = Field(..., description="Résultat net")
    total_debt: SourcedAmount = Field(..., description="Dette financière totale au bilan")
    equity: SourcedAmount = Field(..., description="Capitaux propres")
    cash_and_equivalents: SourcedAmount = Field(..., description="Trésorerie et équivalents")
    current_assets: SourcedAmount = Field(..., description="Actif circulant")
    current_liabilities: SourcedAmount = Field(..., description="Passif circulant")
    annual_debt_service: SourcedAmount = Field(
        ..., description="Service annuel de la dette (intérêts + principal)"
    )
    debt_schedule_total: SourcedAmount | None = Field(
        default=None,
        description="Total issu du tableau détaillé des dettes (Dettes.xlsx) pour rapprochement",
    )

    @model_validator(mode="after")
    def validate_currency_and_period_consistency(self) -> "FinancialStatement":
        """Vérifie l'homogénéité des devises et exercices avant calcul (F03)."""
        amounts = [
            ("revenue", self.revenue),
            ("ebitda", self.ebitda),
            ("net_income", self.net_income),
            ("total_debt", self.total_debt),
            ("equity", self.equity),
            ("cash_and_equivalents", self.cash_and_equivalents),
            ("current_assets", self.current_assets),
            ("current_liabilities", self.current_liabilities),
            ("annual_debt_service", self.annual_debt_service),
        ]
        if self.debt_schedule_total is not None:
            amounts.append(("debt_schedule_total", self.debt_schedule_total))

        for field_name, amt in amounts:
            if amt.currency != self.currency:
                raise ValueError(
                    f"Devise incohérente pour '{field_name}': {amt.currency.value} != {self.currency.value}"
                )
            if amt.period != self.fiscal_year:
                raise ValueError(
                    f"Exercice incohérent pour '{field_name}': {amt.period} != {self.fiscal_year}"
                )
        return self


class CreditCaseCreate(BaseModel):
    """Payload de création d'un dossier (POST /api/v1/cases)."""

    case_id: str = Field(..., min_length=1, description="Ex: 'CR-0261'")
    company_name: str = Field(..., min_length=1)
    sector: str = Field(default="Industrie")
    facility_type: str = Field(default="Financement équipement")
    requested_amount: float = Field(..., gt=0)
    currency: Currency = Currency.MAD
    tenor_months: int = Field(..., gt=0, le=360)
    documents: list[DocumentRef] = Field(default_factory=list)


class CreditCase(CreditCaseCreate):
    """Dossier de crédit complet avec statut opérationnel."""

    status: CaseStatus = CaseStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_quality: float | None = Field(default=None, ge=0.0, le=1.0)


class FinancialMetrics(BaseModel):
    """Résultats déterministes du moteur financier (F03)."""

    case_id: str
    fiscal_year: str
    currency: Currency

    # Montants normalisés (en unité de base)
    revenue: float
    ebitda: float
    net_income: float
    total_debt: float
    equity: float
    net_debt: float
    free_cash_flow_proxy: float

    # Ratios financiers
    debt_to_equity: float | None = Field(default=None, description="Dette totale / Capitaux propres")
    net_debt_to_ebitda: float | None = Field(default=None, description="Dette nette / EBITDA")
    net_margin: float | None = Field(default=None, description="Résultat net / CA")
    ebitda_margin: float | None = Field(default=None, description="EBITDA / CA")
    current_ratio: float | None = Field(default=None, description="Actif circulant / Passif circulant")
    dscr: float | None = Field(
        default=None, description="Debt Service Coverage Ratio (EBITDA / Service annuel dette)"
    )
    requested_to_revenue: float | None = Field(
        default=None, description="Montant demandé / Chiffre d'affaires"
    )

    # Contrôles déterministes de cohérence
    debt_reconciliation_diff: float | None = Field(
        default=None, description="Écart absolu entre bilan (total_debt) et tableau des dettes"
    )
    debt_reconciliation_diff_pct: float | None = Field(
        default=None, description="Écart relatif en % entre bilan et tableau des dettes"
    )
    is_debt_consistent: bool = True
    warnings: list[str] = Field(default_factory=list)


class RiskFinding(BaseModel):
    """Constat de risque ou de conformité (F04) — preuve obligatoire."""

    code: str = Field(..., min_length=1)
    severity: RiskSeverity
    category: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1)
    evidence: list[EvidenceRef] = Field(..., min_length=1)
    uncertainty: float = Field(default=0.1, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Résultat du Verification Agent (F05) — échec bloquant si contrôle manquant."""

    passed: bool
    blocking_errors: list[str] = Field(default_factory=list)
    missing_evidence_fields: list[str] = Field(default_factory=list)
    requires_hitl: bool = False
    suspension_reason: str | None = None


class HITLDecision(BaseModel):
    """Journalisation d'une intervention humaine (F07)."""

    reviewer_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: HITLAction
    comment: str = Field(default="")
    field_overrides: dict[str, Any] = Field(default_factory=dict)


class CreditReport(BaseModel):
    """Rapport d'analyse versionné (F06) — sans décision automatique de crédit."""

    case_id: str
    version: int = Field(default=1, ge=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    financial_metrics: FinancialMetrics
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    verification: VerificationResult
    hitl_decisions: list[HITLDecision] = Field(default_factory=list)
    disclaimer: str = Field(
        default="Document d'aide à l'analyse financière — aucune décision automatique de crédit."
    )
