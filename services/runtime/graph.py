"""Orchestrateur LangGraph avec MemorySaver, checkpoints et reprise HITL."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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
from services.inference.gateway import InferenceGateway
from services.router.engine import DecisionEngineService
from services.runtime.agents import ACIWorkflowNodes
from services.runtime.state import AgentStepEvent, RunRecord, WorkflowState


class ACIRuntime:
    """Pilote le StateGraph LangGraph et conserve l'historique des runs."""

    def __init__(
        self,
        gateway: InferenceGateway,
        decision_engine: DecisionEngineService | None = None,
    ) -> None:
        self.gateway = gateway
        self.decision_engine = decision_engine or DecisionEngineService()
        self.nodes = ACIWorkflowNodes(gateway, self.decision_engine)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self._runs: dict[str, RunRecord] = {}

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("document_agent", self.nodes.document_agent)
        builder.add_node("financial_agent", self.nodes.financial_agent)
        builder.add_node("risk_compliance_agent", self.nodes.risk_compliance_agent)
        builder.add_node("verification_agent", self.nodes.verification_agent)
        builder.add_node("report_agent", self.nodes.report_agent)

        builder.add_edge(START, "document_agent")
        builder.add_edge("document_agent", "financial_agent")
        builder.add_edge("financial_agent", "risk_compliance_agent")
        builder.add_edge("risk_compliance_agent", "verification_agent")
        builder.add_edge("verification_agent", "report_agent")
        builder.add_edge("report_agent", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _sync_record_from_snapshot(self, run_id: str, case_id: str, policy_version: str) -> RunRecord:
        config = {"configurable": {"thread_id": run_id}}
        snapshot = self.graph.get_state(config)
        values: dict[str, Any] = dict(snapshot.values) if snapshot and snapshot.values else {}

        interrupt_payload: dict[str, Any] | None = None
        if snapshot and snapshot.tasks:
            for task in snapshot.tasks:
                if getattr(task, "interrupts", None):
                    interrupt_payload = task.interrupts[0].value
                    break

        events = [AgentStepEvent.model_validate(e) for e in values.get("events", [])]
        if interrupt_payload is not None:
            status = "interrupted"
            # Ajout d'un événement visible d'interruption (Maquette UI 03)
            if not any(e.status == "interrupted" for e in events):
                events.append(
                    AgentStepEvent(
                        node=interrupt_payload.get("node", "Verification Agent"),
                        status="interrupted",
                        model="deterministic-verifier-v1",
                        tier="deterministic",
                        backend="local_deterministic",
                        policy=policy_version,
                        message=interrupt_payload.get("suspension_reason") or "Contrôle humain requis.",
                        details=interrupt_payload,
                    )
                )
        elif values.get("report"):
            status = "completed"
        else:
            status = "running"

        existing = self._runs.get(run_id)
        created_at = existing.created_at if existing else datetime.now(timezone.utc)

        active_policy = values.get("policy_version", policy_version)
        record = RunRecord(
            run_id=run_id,
            case_id=case_id,
            policy_version=active_policy,
            status=status,  # type: ignore[arg-type]
            created_at=created_at,
            updated_at=datetime.now(timezone.utc),
            events=events,
            metrics=(
                FinancialMetrics.model_validate(values["metrics"])
                if values.get("metrics")
                else None
            ),
            risk_findings=[
                RiskFinding.model_validate(f) for f in values.get("risk_findings", [])
            ],
            verification=(
                VerificationResult.model_validate(values["verification"])
                if values.get("verification")
                else None
            ),
            hitl_decisions=[
                HITLDecision.model_validate(d) for d in values.get("hitl_decisions", [])
            ],
            router_decisions=[
                RouterDecision.model_validate(d) for d in values.get("router_decisions", [])
            ],
            rl_transitions=[
                RLTransition.model_validate(t) for t in values.get("rl_transitions", [])
            ],
            report=(
                CreditReport.model_validate(values["report"]) if values.get("report") else None
            ),
            interrupt_payload=interrupt_payload,
        )
        self._runs[run_id] = record
        return record

    async def start_run(
        self,
        case: CreditCase,
        statement: FinancialStatement | None = None,
        documents_text: str = "",
        policy_version: str = "router-v01-static",
    ) -> RunRecord:
        run_id = f"R-{uuid.uuid4().hex[:8].upper()}"
        config = {"configurable": {"thread_id": run_id}}
        effective_policy = self.decision_engine.active_policy_version
        initial_state: WorkflowState = {
            "run_id": run_id,
            "workflow_version": "v1.2",
            "policy_version": effective_policy,
            "case": case.model_dump(mode="json"),
            "documents_text": documents_text,
            "events": [],
            "hitl_decisions": [],
            "router_decisions": [],
            "rl_transitions": [],
            "remaining_budget_usd": 1.20,
        }
        if statement is not None:
            initial_state["statement"] = statement.model_dump(mode="json")

        await self.graph.ainvoke(initial_state, config=config)
        return self._sync_record_from_snapshot(run_id, case.case_id, effective_policy)

    async def resume_run(self, run_id: str, decision: HITLDecision) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Run introuvable : {run_id}")
        if record.status != "interrupted":
            raise ValueError(f"Le run {run_id} n'est pas en attente HITL (statut={record.status}).")

        config = {"configurable": {"thread_id": run_id}}
        await self.graph.ainvoke(
            Command(resume=decision.model_dump(mode="json")),
            config=config,
        )
        return self._sync_record_from_snapshot(run_id, record.case_id, record.policy_version)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())

    def get_latest_report_for_case(self, case_id: str) -> CreditReport | None:
        for run in reversed(list(self._runs.values())):
            if run.case_id == case_id and run.report is not None:
                return run.report
        return None
