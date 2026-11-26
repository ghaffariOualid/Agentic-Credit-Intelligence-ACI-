"""Bandit contextuel LinUCB avec masquage d'actions (Baseline Étape 2 — Page 8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intelligence.decision_model.features import (
    STATE_DIM,
    TIER_LIST,
    compute_action_mask,
    encode_state,
)
from intelligence.reinforcement_learning.env import ACIRoutingEnv
from services.contracts.intelligence import ModelTier, RouterState


@dataclass
class BanditTrainReport:
    episodes: int
    alpha: float
    eval_mean_reward: float
    eval_mean_quality: float
    eval_mean_cost_usd: float


class LinUCBRouter:
    """Bandit contextuel disjoint LinUCB sur les 3 bras (small, medium, large)."""

    def __init__(self, state_dim: int = STATE_DIM, alpha: float = 0.25) -> None:
        self.state_dim = state_dim
        self.alpha = alpha
        self.n_arms = len(TIER_LIST)
        self.A = [np.eye(state_dim, dtype=np.float64) for _ in range(self.n_arms)]
        self.b = [np.zeros(state_dim, dtype=np.float64) for _ in range(self.n_arms)]

    def select_action(
        self, state: RouterState, explore: bool = True
    ) -> tuple[int, np.ndarray]:
        x = encode_state(state).astype(np.float64)
        mask, _ = compute_action_mask(state)
        scores = np.full(self.n_arms, -1e9, dtype=np.float64)

        for arm in range(self.n_arms):
            if not mask[arm]:
                continue
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean = float(theta @ x)
            bonus = float(self.alpha * np.sqrt(x @ A_inv @ x)) if explore else 0.0
            scores[arm] = mean + bonus

        return int(np.argmax(scores)), scores

    def update(self, state: RouterState, action: int, reward: float) -> None:
        x = encode_state(state).astype(np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def predict_tier(self, state: RouterState) -> tuple[ModelTier, np.ndarray]:
        idx, scores = self.select_action(state, explore=False)
        return TIER_LIST[idx], scores


def train_bandit_router(
    num_episodes: int = 150,
    alpha: float = 0.25,
    seed: int = 42,
) -> tuple[LinUCBRouter, BanditTrainReport]:
    """Entraîne le bandit contextuel LinUCB dans le simulateur T4 puis évalue sur 30 épisodes."""
    env = ACIRoutingEnv(seed=seed)
    bandit = LinUCBRouter(alpha=alpha)

    for _ in range(num_episodes):
        _, info = env.reset()
        done = False
        while not done:
            state = info["router_state"]
            action, _ = bandit.select_action(state, explore=True)
            _, r, terminated, truncated, info = env.step(action)
            bandit.update(state, action, r)
            done = terminated or truncated

    eval_env = ACIRoutingEnv(seed=seed + 1000)
    rewards, qualities, costs = [], [], []
    for _ in range(30):
        _, info = eval_env.reset()
        done = False
        while not done:
            action, _ = bandit.select_action(info["router_state"], explore=False)
            _, r, terminated, truncated, info = eval_env.step(action)
            rewards.append(r)
            qualities.append(info["quality"])
            costs.append(info["cost_usd"])
            done = terminated or truncated

    return bandit, BanditTrainReport(
        episodes=num_episodes,
        alpha=alpha,
        eval_mean_reward=round(float(np.mean(rewards)), 4),
        eval_mean_quality=round(float(np.mean(qualities)), 4),
        eval_mean_cost_usd=round(float(np.mean(costs)), 4),
    )
