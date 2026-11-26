"""Tests d'intégration Decision Engine <-> Runtime LangGraph + Benchmark comparatif."""

from pathlib import Path

from fastapi.testclient import TestClient

from benchmarks.run_router_benchmark import run_benchmark
from services.api.app import create_api_app
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway


def test_runtime_uses_active_decision_engine_policy_and_logs_transitions() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="mock"))
    client = TestClient(create_api_app(gateway=gw, seed_flagship_cases=True))

    # 1. Premier run avec la politique heuristique par défaut
    r1 = client.post("/api/v1/cases/CR-0260/analyze")
    assert r1.status_code == 200
    run1 = r1.json()
    assert run1["policy_version"] == "router-v01-heuristic"
    assert len(run1["router_decisions"]) >= 1
    assert len(run1["rl_transitions"]) >= 1

    # 2. Entraînement + activation d'une politique PPO
    r_train = client.post(
        "/api/v1/training/jobs",
        json={"algorithm": "ppo", "episodes": 32, "seed": 42, "activate_policy": True},
    )
    assert r_train.status_code == 200
    ppo_policy = r_train.json()["policy_version"]

    # 3. Le run LangGraph suivant utilise automatiquement la nouvelle politique PPO active
    r2 = client.post("/api/v1/cases/CR-0260/analyze")
    assert r2.status_code == 200
    run2 = r2.json()
    assert run2["policy_version"] == ppo_policy
    assert run2["router_decisions"][0]["policy_version"] == ppo_policy
    assert run2["rl_transitions"][0]["policy_version"] == ppo_policy


def test_comparative_benchmark_generates_reports(tmp_path: Path) -> None:
    results = run_benchmark(
        train_episodes=32,
        eval_episodes=15,
        seeds=(42, 7),
        output_dir=tmp_path,
    )
    assert len(results) == 6
    names = {r.strategy for r in results}
    assert "Static-Large" in names
    assert "Bandit (LinUCB)" in names
    assert "Supervised (PyTorch MLP)" in names
    assert "RL (PPO + Masking)" in names
    assert (tmp_path / "router_benchmark.json").exists()
    assert (tmp_path / "router_benchmark.md").exists()
