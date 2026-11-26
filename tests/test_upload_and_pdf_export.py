"""Tests d'upload réel de pièces (PDF/Excel/TXT) et d'export PDF versionné (F01/F06)."""

import io

from fastapi.testclient import TestClient
from openpyxl import Workbook

from services.api.app import create_api_app
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway


def _build_excel_bytes(total_debt: float, annual_service: float) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "T1"
    ws.append(["Ligne", "Devise", "Capital restant dû", "Échéance annuelle"])
    ws.append(["TOTAL", "MAD", total_debt, annual_service])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_upload_documents_analyze_and_download_pdf() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="mock"))
    client = TestClient(create_api_app(gateway=gw, seed_flagship_cases=False))

    bilan_txt = (
        "BILAN FY2025\n"
        "Chiffre d'affaires: 15 000 000 MAD\n"
        "EBITDA: 2 500 000 MAD\n"
        "Résultat net: 1 100 000 MAD\n"
        "Dettes financières totales: 3 000 000 MAD\n"
        "Capitaux propres: 6 000 000 MAD\n"
        "Trésorerie: 900 000 MAD\n"
        "Actif circulant: 4 500 000 MAD\n"
        "Passif circulant: 2 500 000 MAD\n"
        "Service annuel dette: 800 000 MAD\n"
    ).encode("utf-8")
    dettes_xlsx = _build_excel_bytes(3_000_000.0, 800_000.0)

    r_up = client.post(
        "/api/v1/cases/upload",
        data={
            "case_id": "CR-UP-01",
            "company_name": "Maroc Energie SA",
            "sector": "Énergie",
            "requested_amount": "1500000",
            "tenor_months": "48",
            "currency": "MAD",
        },
        files=[
            ("files", ("Bilan.txt", bilan_txt, "text/plain")),
            ("files", ("Dettes.xlsx", dettes_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ],
    )
    assert r_up.status_code == 201
    assert r_up.json()["case_id"] == "CR-UP-01"
    assert len(r_up.json()["documents"]) == 2

    # Analyse LangGraph complète
    r_run = client.post("/api/v1/cases/CR-UP-01/analyze")
    assert r_run.status_code == 200
    assert r_run.json()["status"] == "completed"
    assert r_run.json()["metrics"]["debt_to_equity"] == 0.5

    # Export PDF bancaire téléchargeable (F06)
    r_pdf = client.get("/api/v1/cases/CR-UP-01/report/pdf")
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert r_pdf.content.startswith(b"%PDF")
    assert len(r_pdf.content) > 1500


def test_upload_rejects_disallowed_extension() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="mock"))
    client = TestClient(create_api_app(gateway=gw, seed_flagship_cases=False))

    r_bad = client.post(
        "/api/v1/cases/upload",
        data={
            "case_id": "CR-BAD-01",
            "company_name": "Test Bad",
            "requested_amount": "500000",
        },
        files=[("files", ("malware.exe", b"MZ...", "application/octet-stream"))],
    )
    assert r_bad.status_code == 400
