"""Tests du Decision Engine : encodeur, action masking, Gymnasium T4, Supervisé, Bandit, PPO & API."""

from pathlib import Path

from fastapi.testclient import TestClient

from intelligence.decision_model.bandit import train_bandit_router
from intelligence.decision_model.features import (
    STATE_DIM,
    compute_action_mask,
    compute_reward,
    encode_state,
)
from intelligence.decision_model.supervised import train_supervised_router
from intelligence.reinforcement_learning.env import ACIRoutingEnv
from intelligence.reinforcement_learning.ppo import PPOConfig, train_ppo_router
from services.api.app import create_api_app
from services.contracts.intelligence import ModelTier, RouterState
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway


def test_feature_encoder_action_mask_and_reward() -> None:
    high_comp_state = RouterState(
        task_type="financial_extraction",
        context_tokens=4500,
        estimated_complexity=0.85,
        gpu_utilization=0.72,
        vram_utilization=0.92,
        queue_depth=12,
        remaining_budget_usd=0.80,
    )
    vec = encode_state(high_comp_state)
    assert vec.shape == (STATE_DIM,)
    assert float(vec.min()) >= 0.0 and float(vec.max()) <= 1.0

    mask, reasons = compute_action_mask(high_comp_state)
    # Complexité >= 0.80 bloque small ; VRAM >= 0.90 bloque large ; medium reste autorisé
    assert list(mask) == [False, True, False]
    assert len(reasons) >= 3

    # Reward exacte Page 8 : 0.6Q - 0.15C - 0.15L - 0.1E
    assert compute_reward(1.0, 0.2, 0.2, 0.0) == 0.54


def test_gymnasium_t4_env_episode_lifecycle() -> None:
    env = ACIRoutingEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (STATE_DIM,)
    assert info["action_mask"].shape == (3,)

    steps = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(1)
        assert info["transition"].model_version == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert -1.0 <= reward <= 1.0
        steps += 1
        done = terminated or truncated
    assert steps == 4


def test_supervised_bandit_and_ppo_trainers(tmp_path: Path) -> None:
    sup_path = tmp_path / "policy-supervised.pt"
    sup_net, sup_rep = train_supervised_router(
        num_episodes=40, epochs=8, seed=42, save_path=sup_path
    )
    assert sup_path.exists()
    assert 0.0 <= sup_rep.val_accuracy <= 1.0
    assert sup_rep.eval_mean_quality >= 0.80

    bandit, bandit_rep = train_bandit_router(num_episodes=40, seed=42)
    assert bandit_rep.eval_mean_quality >= 0.80

    ppo_path = tmp_path / "policy-ppo.pt"
    ppo_net, ppo_rep = train_ppo_router(
        config=PPOConfig(episodes=32, batch_episodes=8, update_epochs=2, seed=42),
        save_path=ppo_path,
    )
    assert ppo_path.exists()
    assert len(ppo_rep.reward_curve) == 4
    assert ppo_rep.mean_quality >= 0.80

    sample_state = RouterState(
        task_type="financial_extraction",
        context_tokens=4500,
        estimated_complexity=0.85,
        gpu_utilization=0.60,
        vram_utilization=0.92,
        queue_depth=8,
        remaining_budget_usd=0.50,
    )
    # Toutes les politiques doivent respecter le masquage (seul medium autorisé ici)
    assert sup_net.predict_tier(sample_state)[0] == ModelTier.MEDIUM
    assert bandit.predict_tier(sample_state)[0] == ModelTier.MEDIUM
    assert ppo_net.predict_tier(sample_state)[0] == ModelTier.MEDIUM


def test_router_and_training_api_endpoints() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="mock"))
    client = TestClient(create_api_app(gateway=gw, seed_flagship_cases=False))

    # 1. Prédiction explicable (Maquette UI 04, page 9)
    r_pred = client.post(
        "/api/v1/router/predict",
        json={
            "task_type": "financial_extraction",
            "context_tokens": 4500,
            "estimated_complexity": 0.65,
            "gpu_utilization": 0.72,
            "vram_utilization": 0.68,
            "queue_depth": 12,
            "remaining_budget_usd": 0.80,
        },
    )
    assert r_pred.status_code == 200
    dec = r_pred.json()
    assert len(dec["candidates"]) == 3
    assert dec["action"]["run_verification"] is True
    assert dec["action"]["selected_tier"] in {"small", "medium", "large"}

    # 2. Lancement d'un entraînement PPO contrôlé + activation
    r_job = client.post(
        "/api/v1/training/jobs",
        json={"algorithm": "ppo", "episodes": 32, "seed": 42, "activate_policy": True},
    )
    assert r_job.status_code == 200
    job = r_job.json()
    assert job["algorithm"] == "ppo"
    assert job["deployed"] is True
    assert len(job["reward_curve"]) >= 1

    r_pol = client.get("/api/v1/router/policy")
    assert r_pol.status_code == 200
    assert r_pol.json()["policy_type"] == "ppo"
