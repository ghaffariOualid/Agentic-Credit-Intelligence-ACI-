"""Modèles d'état et d'événements du runtime LangGraph (Page 7 — Maquette UI 03)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from services.contracts.domain import (
    CreditCase,
    CreditReport,
    FinancialMetrics,
    FinancialStatement,
    HITLDecision,
    RiskFinding,
    VerificationResult,
)
from services.contracts.intelligence import RLTransition, RouterDecision


class AgentStepEvent(BaseModel):
    """Trace structurée d'un nœud d'exécution (affichée dans le graphe et streamée en SSE)."""

    node: str
    status: Literal["completed", "interrupted", "pending", "failed"]
    model: str = "deterministic"
    tier: str = "deterministic"
    backend: str = "local_deterministic"
    tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    policy: str = "router-v01-static"
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunRecord(BaseModel):
    """Vue persistée/exposée d'une exécution multi-agents (`Run #R-...`)."""

    run_id: str
    case_id: str
    workflow_version: str = "v1.2"
    policy_version: str = "router-v01-static"
    status: Literal["running", "interrupted", "completed", "failed"] = "running"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[AgentStepEvent] = Field(default_factory=list)
    metrics: FinancialMetrics | None = None
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    verification: VerificationResult | None = None
    hitl_decisions: list[HITLDecision] = Field(default_factory=list)
    router_decisions: list[RouterDecision] = Field(default_factory=list)
    rl_transitions: list[RLTransition] = Field(default_factory=list)
    report: CreditReport | None = None
    interrupt_payload: dict[str, Any] | None = None


class WorkflowState(TypedDict, total=False):
    """État partagé dans le StateGraph LangGraph (sérialisable en JSON pour checkpoints)."""

    run_id: str
    workflow_version: str
    policy_version: str
    case: dict[str, Any]
    documents_text: str
    statement: dict[str, Any]
    metrics: dict[str, Any]
    risk_findings: list[dict[str, Any]]
    verification: dict[str, Any]
    hitl_decisions: list[dict[str, Any]]
    router_decisions: list[dict[str, Any]]
    rl_transitions: list[dict[str, Any]]
    remaining_budget_usd: float
    report: dict[str, Any]
    events: list[dict[str, Any]]
    synthesis: str
