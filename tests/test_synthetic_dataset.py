"""Tests du générateur de dossiers synthétiques annotés (Lot 1)."""

import json
from pathlib import Path

from datasets.synthetic.generator import (
    generate_synthetic_corpus,
    write_corpus_artifacts,
)


def test_corpus_generation_is_deterministic() -> None:
    c1 = generate_synthetic_corpus(extra_count=6, seed=42)
    c2 = generate_synthetic_corpus(extra_count=6, seed=42)
    assert len(c1) == 9
    assert [x.statement.model_dump() for x in c1] == [x.statement.model_dump() for x in c2]
    assert [x.expected_metrics.model_dump() for x in c1] == [
        x.expected_metrics.model_dump() for x in c2
    ]


def test_corpus_covers_key_scenarios() -> None:
    corpus = generate_synthetic_corpus(extra_count=12, seed=42)
    scenarios = {item.scenario_type for item in corpus}
    assert "debt_reconciliation_mismatch" in scenarios
    assert "healthy_standard" in scenarios
    assert "negative_ebitda_edge_case" in scenarios
    assert "low_ocr_confidence" in scenarios


def test_write_corpus_artifacts_creates_json_and_documents(tmp_path: Path) -> None:
    corpus_path = write_corpus_artifacts(output_dir=tmp_path, extra_count=3, seed=7)
    assert corpus_path.exists()
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert len(data) == 6
    assert (tmp_path / "documents" / "CR-0261" / "Bilan.txt").exists()
    assert (tmp_path / "documents" / "CR-0261" / "Dettes.xlsx").exists()
