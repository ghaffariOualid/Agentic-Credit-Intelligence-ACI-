"""Tests du Lot 2 : Runtime LangGraph, HITL, SSE et API FastAPI métier."""

from fastapi.testclient import TestClient

from datasets.synthetic.generator import build_flagship_cases
from services.api.app import create_api_app
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway


def _make_client() -> TestClient:
    gw = InferenceGateway(InferenceSettings(inference_mode="mock"))
    return TestClient(create_api_app(gateway=gw, seed_flagship_cases=True))


def test_healthy_case_nova_completes_end_to_end_without_hitl() -> None:
    client = _make_client()

    r_run = client.post("/api/v1/cases/CR-0260/analyze")
    assert r_run.status_code == 200
    run = r_run.json()
    assert run["status"] == "completed"
    assert run["report"] is not None
    assert "aucune décision automatique de crédit" in run["report"]["disclaimer"]
    assert len(run["events"]) == 5

    r_rep = client.get("/api/v1/cases/CR-0260/report")
    assert r_rep.status_code == 200
    assert r_rep.json()["case_id"] == "CR-0260"
    assert r_rep.json()["verification"]["passed"] is True


def test_atlas_industrie_interrupts_for_hitl_and_resumes_after_correction() -> None:
    client = _make_client()

    # 1. Lancement de l'analyse sur CR-0261 (écart bilan 3,55M vs tableau dettes 3,92M)
    r_run = client.post("/api/v1/cases/CR-0261/analyze")
    assert r_run.status_code == 200
    run = r_run.json()
    run_id = run["run_id"]
    assert run["status"] == "interrupted"
    assert run["interrupt_payload"] is not None
    assert "Dette totale incohérente" in run["interrupt_payload"]["suspension_reason"]

    # Le rapport ne doit pas encore exister tant que le contrôle humain n'a pas validé
    assert client.get("/api/v1/cases/CR-0261/report").status_code == 404

    # 2. Vérification du streaming SSE des événements pendant l'interruption
    r_sse = client.get(f"/api/v1/runs/{run_id}/events")
    assert r_sse.status_code == 200
    assert "text/event-stream" in r_sse.headers["content-type"]
    assert "event: step" in r_sse.text
    assert "event: status" in r_sse.text
    assert "interrupted" in r_sse.text

    # 3. Reprise HITL avec correction du tableau des dettes par l'analyste
    r_resume = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={
            "reviewer_id": "analyste.oualid",
            "action": "correct",
            "comment": "Rapprochement effectué : ligne crédit-bail exclue de Dettes.xlsx.",
            "field_overrides": {"debt_schedule_total": 3_550_000.0},
        },
    )
    assert r_resume.status_code == 200
    resumed = r_resume.json()
    assert resumed["status"] == "completed"
    assert resumed["verification"]["passed"] is True
    assert resumed["metrics"]["is_debt_consistent"] is True
    assert len(resumed["hitl_decisions"]) == 1
    assert resumed["hitl_decisions"][0]["reviewer_id"] == "analyste.oualid"
    assert resumed["report"] is not None

    # 4. Le rapport final est désormais accessible
    r_rep = client.get("/api/v1/cases/CR-0261/report")
    assert r_rep.status_code == 200
    assert r_rep.json()["financial_metrics"]["debt_to_equity"] == 0.71


def test_create_custom_case_and_analyze() -> None:
    client = _make_client()
    nova = build_flagship_cases()[1]

    custom_case = nova.case.model_copy(update={"case_id": "CR-9001", "company_name": "SolarTech SA"})
    custom_stmt = nova.statement.model_copy(update={"case_id": "CR-9001"})

    r_create = client.post(
        "/api/v1/cases",
        json={
            "case": custom_case.model_dump(mode="json"),
            "statement": custom_stmt.model_dump(mode="json"),
        },
    )
    assert r_create.status_code == 201
    assert r_create.json()["case_id"] == "CR-9001"

    r_run = client.post("/api/v1/cases/CR-9001/analyze")
    assert r_run.status_code == 200
    assert r_run.json()["status"] == "completed"
