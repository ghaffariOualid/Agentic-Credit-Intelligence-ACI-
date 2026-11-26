"""Parseur multi-formats (PDF, Excel, TXT/CSV) et extracteur de montants sourcés (F01/F02)."""

from __future__ import annotations

import io
import re
from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader

from services.contracts.domain import (
    Currency,
    DocumentRef,
    DocumentStatus,
    DocumentType,
    EvidenceRef,
    FinancialStatement,
    SourcedAmount,
    UnitScale,
)

ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".txt", ".csv"}


def infer_doc_type(filename: str) -> DocumentType:
    low = filename.lower()
    if "bilan" in low or "balance" in low:
        return DocumentType.BILAN
    if "resultat" in low or "résultat" in low or "income" in low or "pnl" in low:
        return DocumentType.RESULTAT
    if "dette" in low or "debt" in low:
        return DocumentType.DETTES
    if "releve" in low or "relevé" in low or "bank" in low:
        return DocumentType.RELEVES
    return DocumentType.AUTRE


def extract_segments_from_bytes(
    filename: str, content: bytes
) -> list[tuple[str, str]]:
    """Retourne une liste de (page_or_sheet, text) pour un fichier PDF, XLSX ou TXT/CSV."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        segments: list[tuple[str, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            segments.append((f"p. {idx}", page.extract_text() or ""))
        return segments or [("p. 1", "")]

    if ext == ".xlsx":
        wb = load_workbook(io.BytesIO(content), data_only=True)
        segments = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            lines = []
            for row in ws.iter_rows(values_only=True):
                vals = [str(v) for v in row if v is not None]
                if vals:
                    lines.append(" | ".join(vals))
            segments.append((sheet_name, "\n".join(lines)))
        return segments or [("Feuille 1", "")]

    # .txt / .csv
    text = content.decode("utf-8", errors="replace")
    return [("p. 1", text)]


def _parse_number(raw: str) -> tuple[float, UnitScale]:
    cleaned = raw.strip().replace("\u00a0", " ").replace(" ", "")
    scale = UnitScale.UNIT
    low = cleaned.lower()
    if low.endswith("m") or "mmad" in low or "meur" in low:
        scale = UnitScale.MILLION
        cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)
    elif low.endswith("k") or "kmad" in low or "keur" in low:
        scale = UnitScale.THOUSAND
        cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)
    else:
        cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)

    # Gestion séparateur décimal virgule vs milliers
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    return float(cleaned), scale


def _search_amount(
    all_docs: list[tuple[str, str, list[tuple[str, str]]]],
    patterns: list[str],
    *,
    currency: Currency,
    fiscal_year: str,
    fallback_value: float,
    field_label: str,
) -> SourcedAmount:
    for doc_id, filename, segments in all_docs:
        for page_ref, text in segments:
            for pat in patterns:
                match = re.search(pat, text, flags=re.IGNORECASE)
                if match:
                    raw_num = match.group(1)
                    try:
                        val, scale = _parse_number(raw_num)
                        excerpt = match.group(0).strip()[:140]
                        return SourcedAmount(
                            value=val,
                            currency=currency,
                            scale=scale,
                            period=fiscal_year,
                            evidence=EvidenceRef(
                                document_id=doc_id,
                                document_name=filename,
                                page_or_sheet=page_ref,
                                excerpt=excerpt,
                                confidence=0.94,
                            ),
                        )
                    except ValueError:
                        continue

    # Champ non trouvé : confiance volontairement basse (0.60) pour déclencher HITL F05
    first_doc_id = all_docs[0][0] if all_docs else "DOC-MISSING"
    first_name = all_docs[0][1] if all_docs else "Piece_manquante"
    return SourcedAmount(
        value=fallback_value,
        currency=currency,
        scale=UnitScale.UNIT,
        period=fiscal_year,
        evidence=EvidenceRef(
            document_id=first_doc_id,
            document_name=first_name,
            page_or_sheet="N/A",
            excerpt=f"{field_label} non détecté automatiquement — vérification analyste requise.",
            confidence=0.60,
        ),
    )


def parse_uploaded_documents(
    *,
    case_id: str,
    files: list[tuple[str, bytes]],
    currency: Currency = Currency.MAD,
    fiscal_year: str = "FY2025",
    uploads_root: Path | None = None,
) -> tuple[list[DocumentRef], FinancialStatement, str]:
    """Valide, isole sur disque par dossier et extrait un FinancialStatement sourcé (F01/F02)."""
    if not files:
        raise ValueError("Au moins un fichier PDF, Excel ou TXT est requis.")

    base_dir = (uploads_root or (Path(__file__).resolve().parents[2] / "datasets" / "uploads")) / case_id
    base_dir.mkdir(parents=True, exist_ok=True)

    doc_refs: list[DocumentRef] = []
    parsed_docs: list[tuple[str, str, list[tuple[str, str]]]] = []
    concatenated_text: list[str] = []

    for idx, (filename, content) in enumerate(files, start=1):
        safe_name = Path(filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Format non autorisé pour '{safe_name}'. Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}."
            )

        # Isolation physique par dossier (F01)
        (base_dir / safe_name).write_bytes(content)

        segments = extract_segments_from_bytes(safe_name, content)
        doc_id = f"{case_id}-UP{idx}"
        doc_refs.append(
            DocumentRef(
                document_id=doc_id,
                filename=safe_name,
                doc_type=infer_doc_type(safe_name),
                status=DocumentStatus.VALIDATED,
                page_or_sheet_count=max(1, len(segments)),
                storage_uri=str(base_dir / safe_name),
            )
        )
        parsed_docs.append((doc_id, safe_name, segments))
        for pg, txt in segments:
            concatenated_text.append(f"[{safe_name} | {pg}]\n{txt}")

    num_group = r"([+-]?\d(?:[\d\s\u00a0.,]*\d)?(?:\s*(?:kMAD|MMAD|kEUR|MEUR|[kKmM](?![a-zA-Z])))?)"
    revenue = _search_amount(
        parsed_docs,
        [rf"(?:chiffre\s*d['’]affaires|revenue|ca)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=10_000_000.0,
        field_label="Chiffre d'affaires",
    )
    ebitda = _search_amount(
        parsed_docs,
        [rf"(?:ebitda|ebe|exc[eé]dent\s+brut)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=1_500_000.0,
        field_label="EBITDA",
    )
    net_income = _search_amount(
        parsed_docs,
        [rf"(?:r[eé]sultat\s+net|net\s+income)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=600_000.0,
        field_label="Résultat net",
    )
    total_debt = _search_amount(
        parsed_docs,
        [rf"(?:dettes?\s+financi[eè]res?(?:\s+totales?)?|total\s+debt)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=2_500_000.0,
        field_label="Dettes financières",
    )
    equity = _search_amount(
        parsed_docs,
        [rf"(?:capitaux\s+propres|equity)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=4_000_000.0,
        field_label="Capitaux propres",
    )
    cash = _search_amount(
        parsed_docs,
        [rf"(?:tr[eé]sorerie(?:\s+cl[oô]ture)?|cash)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=750_000.0,
        field_label="Trésorerie",
    )
    current_assets = _search_amount(
        parsed_docs,
        [rf"(?:actif\s+circulant|current\s+assets)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=3_000_000.0,
        field_label="Actif circulant",
    )
    current_liabilities = _search_amount(
        parsed_docs,
        [rf"(?:passif\s+circulant|current\s+liabilities)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=2_000_000.0,
        field_label="Passif circulant",
    )
    annual_debt_service = _search_amount(
        parsed_docs,
        [rf"(?:service\s+annuel\s+dette|[eé]ch[eé]ances?\s+annuelles?|annual\s+debt\s+service)\s*[:|]?\s*{num_group}"],
        currency=currency,
        fiscal_year=fiscal_year,
        fallback_value=600_000.0,
        field_label="Service annuel de la dette",
    )

    # Rapprochement optionnel depuis Dettes.xlsx (ligne TOTAL)
    debt_schedule: SourcedAmount | None = None
    dettes_docs = [d for d in parsed_docs if infer_doc_type(d[1]) == DocumentType.DETTES]
    if dettes_docs:
        cand = _search_amount(
            dettes_docs,
            [rf"(?:total\s+tableau\s+dettes|total)\s*\|?\s*(?:[A-Z]{{3}}\s*\|?\s*)?{num_group}"],
            currency=currency,
            fiscal_year=fiscal_year,
            fallback_value=total_debt.normalized_value(),
            field_label="Total tableau des dettes",
        )
        if cand.evidence.confidence >= 0.9:
            debt_schedule = cand

    statement = FinancialStatement(
        case_id=case_id,
        fiscal_year=fiscal_year,
        currency=currency,
        revenue=revenue,
        ebitda=ebitda,
        net_income=net_income,
        total_debt=total_debt,
        equity=equity,
        cash_and_equivalents=cash,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        annual_debt_service=annual_debt_service,
        debt_schedule_total=debt_schedule,
    )
    return doc_refs, statement, "\n\n".join(concatenated_text)
