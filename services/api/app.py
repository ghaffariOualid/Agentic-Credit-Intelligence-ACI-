"""API FastAPI métier ACI : dossiers, exécutions LangGraph, SSE, reprise HITL et rapports."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from agents.document.parser import parse_uploaded_documents
from agents.report.pdf_exporter import build_credit_report_pdf
from datasets.synthetic.generator import build_flagship_cases
from services.contracts.domain import (
    CaseStatus,
    CreditCase,
    CreditCaseCreate,
    CreditReport,
    Currency,
    FinancialStatement,
    HITLDecision,
)
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway
from services.router.engine import DecisionEngineService
from services.router.routes import create_router_api
from services.runtime.graph import ACIRuntime
from services.runtime.state import RunRecord


class CaseCreateRequest(BaseModel):
    """Payload étendu pour POST /api/v1/cases (accepte un statement initial optionnel)."""

    case: CreditCaseCreate
    statement: FinancialStatement | None = None
    documents_text: str = Field(default="")


class AnalyzeRequest(BaseModel):
    policy_version: str = Field(default="router-v01-static")


def create_api_app(
    gateway: InferenceGateway | None = None,
    seed_flagship_cases: bool = False,
) -> FastAPI:
    app = FastAPI(
        title="ACI Business API",
        version="0.1.0",
        description="API métier d'analyse financière assistée par agents LangGraph et HITL.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    gw = gateway or InferenceGateway(InferenceSettings())
    decision_engine = DecisionEngineService()
    runtime = ACIRuntime(gw, decision_engine=decision_engine)
    app.include_router(create_router_api(decision_engine))

    cases_store: dict[str, CreditCase] = {}
    statements_store: dict[str, FinancialStatement] = {}
    docs_text_store: dict[str, str] = {}

    if seed_flagship_cases:
        for item in build_flagship_cases():
            cases_store[item.case.case_id] = item.case
            statements_store[item.case.case_id] = item.statement

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "cases_loaded": len(cases_store),
            "runs_count": len(runtime.list_runs()),
            "inference_mode": gw.settings.inference_mode,
            "vllm_configured": bool(gw.settings.vllm_base_url.strip()),
            "models": {
                "small": gw.settings.model_small,
                "medium": gw.settings.model_medium,
                "large": gw.settings.model_large,
            },
        }

    @app.get("/api/v1/inference/metrics")
    async def inference_metrics() -> dict[str, object]:
        return gw.metrics.snapshot()

    @app.get("/api/v1/cases", response_model=list[CreditCase])
    async def list_cases() -> list[CreditCase]:
        return list(cases_store.values())

    @app.get("/api/v1/cases/{case_id}", response_model=CreditCase)
    async def get_case(case_id: str) -> CreditCase:
        case = cases_store.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Dossier introuvable : {case_id}")
        return case

    @app.post("/api/v1/cases", response_model=CreditCase, status_code=201)
    async def create_case(payload: CaseCreateRequest) -> CreditCase:
        case_id = payload.case.case_id
        if case_id in cases_store:
            raise HTTPException(status_code=409, detail=f"Le dossier {case_id} existe déjà.")

        case = CreditCase(**payload.case.model_dump())
        cases_store[case_id] = case
        if payload.statement is not None:
            statements_store[case_id] = payload.statement
        if payload.documents_text:
            docs_text_store[case_id] = payload.documents_text
        return case

    @app.post("/api/v1/cases/upload", response_model=CreditCase, status_code=201)
    async def upload_case_with_documents(
        case_id: str = Form(...),
        company_name: str = Form(...),
        sector: str = Form("Industrie"),
        facility_type: str = Form("Financement équipement"),
        requested_amount: float = Form(...),
        currency: Currency = Form(Currency.MAD),
        tenor_months: int = Form(60),
        fiscal_year: str = Form("FY2025"),
        files: list[UploadFile] = File(...),
    ) -> CreditCase:
        if case_id in cases_store:
            raise HTTPException(status_code=409, detail=f"Le dossier {case_id} existe déjà.")

        raw_files: list[tuple[str, bytes]] = []
        for f in files:
            content = await f.read()
            raw_files.append((f.filename or "document.txt", content))

        try:
            doc_refs, statement, docs_text = parse_uploaded_documents(
                case_id=case_id,
                files=raw_files,
                currency=currency,
                fiscal_year=fiscal_year,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        case = CreditCase(
            case_id=case_id,
            company_name=company_name,
            sector=sector,
            facility_type=facility_type,
            requested_amount=requested_amount,
            currency=currency,
            tenor_months=tenor_months,
            documents=doc_refs,
            extraction_quality=0.92,
        )
        cases_store[case_id] = case
        statements_store[case_id] = statement
        docs_text_store[case_id] = docs_text
        return case

    @app.post("/api/v1/cases/{case_id}/analyze", response_model=RunRecord)
    async def analyze_case(
        case_id: str,
        payload: AnalyzeRequest | None = None,
    ) -> RunRecord:
        case = cases_store.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Dossier introuvable : {case_id}")

        stmt = statements_store.get(case_id)
        docs_text = docs_text_store.get(case_id, "")
        if stmt is None and not docs_text:
            raise HTTPException(
                status_code=400,
                detail="Aucune donnée financière ou texte documentaire associé à ce dossier.",
            )

        policy_version = payload.policy_version if payload else "router-v01-static"
        run = await runtime.start_run(
            case=case,
            statement=stmt,
            documents_text=docs_text,
            policy_version=policy_version,
        )

        # Synchronisation du statut du dossier
        if run.status == "interrupted":
            cases_store[case_id] = case.model_copy(update={"status": CaseStatus.NEEDS_REVIEW})
        elif run.status == "completed":
            cases_store[case_id] = case.model_copy(update={"status": CaseStatus.COMPLETED})
        return run

    @app.get("/api/v1/runs", response_model=list[RunRecord])
    async def list_runs() -> list[RunRecord]:
        return runtime.list_runs()

    @app.get("/api/v1/runs/{run_id}", response_model=RunRecord)
    async def get_run(run_id: str) -> RunRecord:
        run = runtime.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Exécution introuvable : {run_id}")
        return run

    @app.get("/api/v1/runs/{run_id}/events")
    async def stream_run_events(run_id: str) -> StreamingResponse:
        run = runtime.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Exécution introuvable : {run_id}")

        async def event_generator() -> AsyncIterator[str]:
            for step in run.events:
                data = json.dumps(step.model_dump(mode="json"), ensure_ascii=False)
                yield f"event: step\ndata: {data}\n\n"
            summary = json.dumps(
                {
                    "run_id": run.run_id,
                    "case_id": run.case_id,
                    "status": run.status,
                    "interrupt_payload": run.interrupt_payload,
                },
                ensure_ascii=False,
            )
            yield f"event: status\ndata: {summary}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/v1/runs/{run_id}/resume", response_model=RunRecord)
    async def resume_run(run_id: str, decision: HITLDecision) -> RunRecord:
        try:
            run = await runtime.resume_run(run_id, decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        case = cases_store.get(run.case_id)
        if case is not None and run.status == "completed":
            cases_store[run.case_id] = case.model_copy(update={"status": CaseStatus.COMPLETED})
        return run

    @app.get("/api/v1/cases/{case_id}/report", response_model=CreditReport)
    async def get_case_report(case_id: str) -> CreditReport:
        if case_id not in cases_store:
            raise HTTPException(status_code=404, detail=f"Dossier introuvable : {case_id}")
        report = runtime.get_latest_report_for_case(case_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun rapport final disponible pour {case_id} (analyse non terminée ou en attente HITL).",
            )
        return report

    @app.get("/api/v1/cases/{case_id}/report/pdf")
    async def download_case_report_pdf(case_id: str) -> Response:
        case = cases_store.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Dossier introuvable : {case_id}")
        report = runtime.get_latest_report_for_case(case_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun rapport final disponible pour {case_id}.",
            )
        pdf_bytes = build_credit_report_pdf(case, report)
        filename = f"ACI_Rapport_{case_id}_v{report.version}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


app = create_api_app()
