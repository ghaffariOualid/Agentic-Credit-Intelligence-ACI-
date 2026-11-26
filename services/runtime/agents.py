"""Nœuds d'agents du workflow ACI connectés au Decision Engine (Routeur) et à l'Inference Gateway.

Garanties architecturales (Page 3 & 8) :
- Le routeur choisit le tier LLM (small/medium/large) pour les appels d'inférence.
- Le routeur ne contourne JAMAIS les calculs financiers déterministes ni les contrôles bloquants F05.
- Chaque décision de routage produit une trace `RouterDecision` et une `RLTransition`.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from agents.financial.calculator import (
    compute_financial_metrics,
    verify_statement_and_metrics,
)
from intelligence.decision_model.features import compute_reward
from services.contracts.domain import (
    CreditCase,
    CreditReport,
    FinancialMetrics,
    FinancialStatement,
    HITLAction,
    HITLDecision,
    RiskFinding,
    RiskSeverity,
    VerificationResult,
)
from services.contracts.inference import ChatCompletionRequest, ChatMessage
from services.contracts.intelligence import (
    RLTransition,
    RouterDecision,
    RouterState,
)
from services.inference.gateway import InferenceGateway
from services.router.engine import DecisionEngineService
from services.runtime.state import AgentStepEvent, WorkflowState


def _append_event(state: WorkflowState, event: AgentStepEvent) -> list[dict[str, Any]]:
    events = list(state.get("events", []))
    events.append(event.model_dump(mode="json"))
    return events


class ACIWorkflowNodes:
    """Implémentation des 5 nœuds du graphe d'exécution ACI."""

    def __init__(
        self,
        gateway: InferenceGateway,
        decision_engine: DecisionEngineService | None = None,
    ) -> None:
        self.gateway = gateway
        self.decision_engine = decision_engine or DecisionEngineService()

    def _route_task(
        self,
        state: WorkflowState,
        *,
        task_type: str,
        context_tokens: int,
        estimated_complexity: float,
        step_index: int,
        previous_confidence: float = 0.95,
    ) -> tuple[RouterState, RouterDecision]:
        queue_depth = self.gateway.metrics.in_flight
        gpu_util = min(0.95, 0.30 + 0.08 * queue_depth)
        vram_util = min(0.92, 0.45 + 0.05 * queue_depth)
        remaining_budget = float(state.get("remaining_budget_usd", 1.20))

        router_state = RouterState(
            task_type=task_type,
            context_tokens=context_tokens,
            estimated_complexity=round(min(max(estimated_complexity, 0.05), 0.98), 4),
            gpu_utilization=round(gpu_util, 4),
            vram_utilization=round(vram_util, 4),
            queue_depth=queue_depth,
            remaining_budget_usd=round(max(0.0, remaining_budget), 4),
            step_index=step_index,
            previous_confidence=round(min(max(previous_confidence, 0.0), 1.0), 4),
        )
        decision = self.decision_engine.predict(router_state)
        return router_state, decision

    def _record_transition(
        self,
        state: WorkflowState,
        router_state: RouterState,
        decision: RouterDecision,
        *,
        resolved_model: str,
        quality: float,
        latency_ms: float,
        cost_usd: float,
        done: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        router_decisions = list(state.get("router_decisions", []))
        router_decisions.append(decision.model_dump(mode="json"))

        cost_norm = min(cost_usd / 0.50, 1.0)
        latency_norm = min((latency_ms / 1000.0) / 8.0, 1.0)
        reward = compute_reward(quality, cost_norm, latency_norm, error_rate=0.0)

        transition = RLTransition(
            episode_id=state["run_id"],
            state=router_state,
            action=decision.action,
            reward=reward,
            next_state=None,
            done=done,
            model_version=resolved_model,
            policy_version=decision.policy_version,
            quality=round(quality, 4),
            latency_ms=round(latency_ms, 2),
            gpu_seconds=round(latency_ms / 1000.0, 4),
            masked_actions=decision.action.masked_actions,
            measurement_context={"decision_id": decision.decision_id, "cost_usd": cost_usd},
        )
        rl_transitions = list(state.get("rl_transitions", []))
        rl_transitions.append(transition.model_dump(mode="json"))
        next_budget = max(0.0, float(state.get("remaining_budget_usd", 1.20)) - cost_usd)
        return router_decisions, rl_transitions, round(next_budget, 4)

    async def document_agent(self, state: WorkflowState) -> dict[str, Any]:
        """F01/F02 — Extraction et validation du schéma FinancialStatement."""
        t0 = time.perf_counter()
        policy = self.decision_engine.active_policy_version
        case = CreditCase.model_validate(state["case"])

        if state.get("statement"):
            stmt = FinancialStatement.model_validate(state["statement"])
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            event = AgentStepEvent(
                node="Document Agent",
                status="completed",
                model="pydantic-schema-validator",
                tier="deterministic",
                backend="local_deterministic",
                tokens=0,
                latency_ms=elapsed_ms,
                policy=policy,
                message=f"Contrat FinancialStatement validé ({stmt.fiscal_year}, {stmt.currency.value}).",
            )
            return {
                "policy_version": policy,
                "statement": stmt.model_dump(mode="json"),
                "events": _append_event(state, event),
            }

        docs_text = state.get("documents_text", "")
        est_tokens = max(500, len(docs_text.split()) * 2)
        router_state, decision = self._route_task(
            state,
            task_type="financial_extraction",
            context_tokens=est_tokens,
            estimated_complexity=0.65,
            step_index=0,
        )
        selected_tier = decision.action.selected_tier.value
        selected_cand = next(
            c for c in decision.candidates if c.tier == decision.action.selected_tier
        )

        resp = await self.gateway.chat_completions(
            ChatCompletionRequest(
                model=selected_tier,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "Tu es un extracteur financier bancaire. Extrais les postes clés "
                            "en JSON valide sans inventer de chiffres."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Dossier {case.case_id} ({case.company_name}). Extraits:\n{docs_text[:3000]}",
                    ),
                ],
            )
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        router_decisions, rl_transitions, next_budget = self._record_transition(
            state,
            router_state,
            decision,
            resolved_model=resp.aci_meta.resolved_model,
            quality=selected_cand.predicted_quality,
            latency_ms=elapsed_ms,
            cost_usd=selected_cand.predicted_cost_usd,
        )
        event = AgentStepEvent(
            node="Document Agent",
            status="completed",
            model=resp.aci_meta.resolved_model,
            tier=selected_tier,
            backend=resp.aci_meta.backend,
            tokens=resp.usage.total_tokens,
            latency_ms=elapsed_ms,
            policy=decision.policy_version,
            message=f"Extraction documentaire routée vers '{selected_tier}' ({decision.decision_id}).",
            details={"decision_id": decision.decision_id, "rationale": decision.rationale},
        )
        return {
            "policy_version": decision.policy_version,
            "router_decisions": router_decisions,
            "rl_transitions": rl_transitions,
            "remaining_budget_usd": next_budget,
            "events": _append_event(state, event),
        }

    async def financial_agent(self, state: WorkflowState) -> dict[str, Any]:
        """F03 — Calculs financiers strictement déterministes (aucun calcul par LLM)."""
        t0 = time.perf_counter()
        policy = state.get("policy_version", self.decision_engine.active_policy_version)
        case = CreditCase.model_validate(state["case"])
        stmt = FinancialStatement.model_validate(state["statement"])

        metrics = compute_financial_metrics(stmt, requested_amount=case.requested_amount)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        event = AgentStepEvent(
            node="Financial Agent",
            status="completed",
            model="deterministic-calculator-v1",
            tier="deterministic",
            backend="local_deterministic",
            tokens=0,
            latency_ms=elapsed_ms,
            policy=policy,
            message=(
                f"Ratios calculés : Dette/CP={metrics.debt_to_equity}, "
                f"DSCR={metrics.dscr}, Liquidité={metrics.current_ratio}."
            ),
            details={"warnings_count": len(metrics.warnings)},
        )
        return {
            "metrics": metrics.model_dump(mode="json"),
            "events": _append_event(state, event),
        }

    async def risk_compliance_agent(self, state: WorkflowState) -> dict[str, Any]:
        """F04 — Identification des facteurs de risque sourcés + synthèse routée par Decision Engine."""
        t0 = time.perf_counter()
        case = CreditCase.model_validate(state["case"])
        stmt = FinancialStatement.model_validate(state["statement"])
        metrics = FinancialMetrics.model_validate(state["metrics"])

        findings: list[RiskFinding] = []
        if not metrics.is_debt_consistent:
            ev_list = [stmt.total_debt.evidence]
            if stmt.debt_schedule_total is not None:
                ev_list.append(stmt.debt_schedule_total.evidence)
            findings.append(
                RiskFinding(
                    code="COH-DEBT-01",
                    severity=RiskSeverity.BLOCKING,
                    category="Cohérence documentaire",
                    statement=(
                        "Écart significatif entre la dette financière au bilan et le total "
                        "du tableau détaillé des dettes ; rapprochement requis avant validation."
                    ),
                    evidence=ev_list,
                    uncertainty=0.05,
                )
            )

        if metrics.current_ratio is not None and metrics.current_ratio < 1.0:
            findings.append(
                RiskFinding(
                    code="LIQ-01",
                    severity=RiskSeverity.HIGH,
                    category="Liquidité",
                    statement=(
                        f"Ratio de liquidité générale inférieur à 1,0 ({metrics.current_ratio:.2f}) : "
                        "tension potentielle sur le besoin en fonds de roulement."
                    ),
                    evidence=[stmt.current_assets.evidence, stmt.current_liabilities.evidence],
                    uncertainty=0.12,
                )
            )

        if metrics.dscr is not None and metrics.dscr < 1.2:
            findings.append(
                RiskFinding(
                    code="DSCR-01",
                    severity=RiskSeverity.HIGH,
                    category="Capacité de remboursement",
                    statement=(
                        f"DSCR de {metrics.dscr:.2f} inférieur au seuil interne de vigilance (1,20)."
                    ),
                    evidence=[stmt.ebitda.evidence, stmt.annual_debt_service.evidence],
                    uncertainty=0.10,
                )
            )

        if not findings:
            findings.append(
                RiskFinding(
                    code="GEN-01",
                    severity=RiskSeverity.LOW,
                    category="Structure financière",
                    statement=(
                        f"Indicateurs principaux dans les seuils standards "
                        f"(Dette/CP={metrics.debt_to_equity}, DSCR={metrics.dscr})."
                    ),
                    evidence=[stmt.total_debt.evidence, stmt.equity.evidence],
                    uncertainty=0.08,
                )
            )

        # Estimation de complexité pour le Decision Engine
        est_complexity = min(0.95, 0.35 + 0.22 * len(metrics.warnings) + (0.20 if not metrics.is_debt_consistent else 0.0))
        router_state, decision = self._route_task(
            state,
            task_type="risk_analysis",
            context_tokens=2400 + 400 * len(findings),
            estimated_complexity=est_complexity,
            step_index=1,
            previous_confidence=case.extraction_quality or 0.95,
        )
        selected_tier = decision.action.selected_tier.value
        selected_cand = next(
            c for c in decision.candidates if c.tier == decision.action.selected_tier
        )

        prompt = (
            f"Entreprise : {case.company_name} ({case.sector}). "
            f"Montant demandé : {case.requested_amount:,.0f} {case.currency.value} sur {case.tenor_months} mois.\n"
            f"Ratios déterministes : CA={metrics.revenue:,.0f}, RN={metrics.net_income:,.0f}, "
            f"Dette/CP={metrics.debt_to_equity}, DSCR={metrics.dscr}, Liquidité={metrics.current_ratio}.\n"
            f"Alertes : {'; '.join(metrics.warnings) if metrics.warnings else 'Aucune'}.\n"
            "Rédige un bref constat factuel de risque (2 phrases max) en français, "
            "sans jamais recommander d'accorder ou de refuser le crédit."
        )
        resp = await self.gateway.chat_completions(
            ChatCompletionRequest(
                model=selected_tier,
                temperature=0.1,
                max_tokens=220,
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "Tu es un assistant d'analyse financière bancaire (ACI). "
                            "Tu produis uniquement des constats factuels sourcés, sans décision d'octroi."
                        ),
                    ),
                    ChatMessage(role="user", content=prompt),
                ],
            )
        )
        synthesis_text = resp.choices[0].message.content if resp.choices else ""
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        router_decisions, rl_transitions, next_budget = self._record_transition(
            state,
            router_state,
            decision,
            resolved_model=resp.aci_meta.resolved_model,
            quality=selected_cand.predicted_quality,
            latency_ms=elapsed_ms,
            cost_usd=selected_cand.predicted_cost_usd,
        )

        event = AgentStepEvent(
            node="Risk + Compliance",
            status="completed",
            model=resp.aci_meta.resolved_model,
            tier=selected_tier,
            backend=resp.aci_meta.backend,
            tokens=resp.usage.total_tokens,
            latency_ms=elapsed_ms,
            policy=decision.policy_version,
            message=(
                f"{len(findings)} constat(s) sourcé(s) ; modèle '{selected_tier}' "
                f"sélectionné par {decision.policy_version} ({decision.decision_id})."
            ),
            details={
                "decision_id": decision.decision_id,
                "rationale": decision.rationale,
                "synthesis_preview": synthesis_text[:200],
            },
        )
        return {
            "policy_version": decision.policy_version,
            "risk_findings": [f.model_dump(mode="json") for f in findings],
            "synthesis": synthesis_text,
            "router_decisions": router_decisions,
            "rl_transitions": rl_transitions,
            "remaining_budget_usd": next_budget,
            "events": _append_event(state, event),
        }

    async def verification_agent(self, state: WorkflowState) -> dict[str, Any]:
        """F05 & F07 — Contrôle bloquant déterministe (jamais contourné) et interruption HITL."""
        t0 = time.perf_counter()
        policy = state.get("policy_version", self.decision_engine.active_policy_version)
        case = CreditCase.model_validate(state["case"])
        stmt = FinancialStatement.model_validate(state["statement"])
        metrics = FinancialMetrics.model_validate(state["metrics"])
        verif = verify_statement_and_metrics(stmt, metrics)

        events = list(state.get("events", []))
        hitl_decisions = list(state.get("hitl_decisions", []))

        if verif.requires_hitl:
            resume_raw = interrupt(
                {
                    "node": "Verification Agent",
                    "suspension_reason": verif.suspension_reason,
                    "blocking_errors": verif.blocking_errors,
                    "missing_evidence_fields": verif.missing_evidence_fields,
                    "action_required": "Confirmer ou corriger les valeurs avant reprise.",
                }
            )
            decision = HITLDecision.model_validate(resume_raw)
            hitl_decisions.append(decision.model_dump(mode="json"))

            if decision.field_overrides:
                stmt_dict = stmt.model_dump(mode="python")
                for field_name, new_value in decision.field_overrides.items():
                    if field_name in stmt_dict and isinstance(stmt_dict[field_name], dict):
                        stmt_dict[field_name]["value"] = float(new_value)
                        stmt_dict[field_name]["evidence"]["confidence"] = 1.0
                        stmt_dict[field_name]["evidence"]["excerpt"] = (
                            f"Corrigé par {decision.reviewer_id}: {new_value}"
                        )
                stmt = FinancialStatement.model_validate(stmt_dict)
                metrics = compute_financial_metrics(stmt, requested_amount=case.requested_amount)
                verif = verify_statement_and_metrics(stmt, metrics)
            elif decision.action == HITLAction.APPROVE:
                verif = verif.model_copy(
                    update={"passed": True, "requires_hitl": False, "suspension_reason": None}
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            events.append(
                AgentStepEvent(
                    node="Verification Agent",
                    status="completed",
                    model="deterministic-verifier-v1",
                    tier="deterministic",
                    backend="local_deterministic",
                    tokens=0,
                    latency_ms=elapsed_ms,
                    policy=policy,
                    message=(
                        f"Reprise HITL validée par {decision.reviewer_id} "
                        f"({decision.action.value}). Contrôle final passé={verif.passed}."
                    ),
                    details={"reviewer_id": decision.reviewer_id, "overrides": decision.field_overrides},
                ).model_dump(mode="json")
            )
            return {
                "statement": stmt.model_dump(mode="json"),
                "metrics": metrics.model_dump(mode="json"),
                "verification": verif.model_dump(mode="json"),
                "hitl_decisions": hitl_decisions,
                "events": events,
            }

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        events.append(
            AgentStepEvent(
                node="Verification Agent",
                status="completed",
                model="deterministic-verifier-v1",
                tier="deterministic",
                backend="local_deterministic",
                tokens=0,
                latency_ms=elapsed_ms,
                policy=policy,
                message="Tous les contrôles obligatoires et preuves documentaires sont valides.",
            ).model_dump(mode="json")
        )
        return {
            "verification": verif.model_dump(mode="json"),
            "hitl_decisions": hitl_decisions,
            "events": events,
        }

    async def report_agent(self, state: WorkflowState) -> dict[str, Any]:
        """F06 — Génération du rapport versionné (sans décision automatique de crédit)."""
        t0 = time.perf_counter()
        policy = state.get("policy_version", self.decision_engine.active_policy_version)
        case = CreditCase.model_validate(state["case"])
        metrics = FinancialMetrics.model_validate(state["metrics"])
        findings = [RiskFinding.model_validate(f) for f in state.get("risk_findings", [])]
        verif = (
            VerificationResult.model_validate(state["verification"])
            if state.get("verification")
            else verify_statement_and_metrics(
                FinancialStatement.model_validate(state["statement"]), metrics
            )
        )
        hitl_decisions = [HITLDecision.model_validate(d) for d in state.get("hitl_decisions", [])]

        report = CreditReport(
            case_id=case.case_id,
            version=1,
            financial_metrics=metrics,
            risk_findings=findings,
            verification=verif,
            hitl_decisions=hitl_decisions,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        event = AgentStepEvent(
            node="Report Agent",
            status="completed",
            model="report-assembler-v1",
            tier="deterministic",
            backend="local_deterministic",
            tokens=0,
            latency_ms=elapsed_ms,
            policy=policy,
            message=f"Rapport versionné v{report.version} généré pour {case.case_id}.",
        )
        return {
            "report": report.model_dump(mode="json"),
            "events": _append_event(state, event),
        }
