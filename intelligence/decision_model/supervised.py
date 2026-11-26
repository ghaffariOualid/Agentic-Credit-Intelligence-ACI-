"""Classifieur supervisé PyTorch pour le routage de modèles (Baseline Étape 2 — Page 8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from intelligence.decision_model.features import (
    STATE_DIM,
    TIER_LIST,
    compute_action_mask,
    encode_state,
)
from intelligence.reinforcement_learning.env import (
    ACIRoutingEnv,
    estimate_candidate_outcome,
    generate_routing_dataset,
)
from services.contracts.intelligence import ModelTier, RLTransition, RouterState


class SupervisedRouterNet(nn.Module):
    """MLP PyTorch prédisant les logits sur les 3 tiers (small, medium, large)."""

    def __init__(self, input_dim: int = STATE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(TIER_LIST)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_tier(self, state: RouterState) -> tuple[ModelTier, np.ndarray]:
        self.eval()
        obs = torch.from_numpy(encode_state(state)).unsqueeze(0)
        logits = self.forward(obs).squeeze(0).numpy()
        mask, _ = compute_action_mask(state)
        masked_logits = np.where(mask, logits, -1e9)
        probs = np.exp(masked_logits - np.max(masked_logits))
        probs = probs / np.sum(probs)
        best_idx = int(np.argmax(masked_logits))
        return TIER_LIST[best_idx], probs


@dataclass
class SupervisedTrainReport:
    epochs: int
    train_samples: int
    val_samples: int
    val_accuracy: float
    eval_mean_reward: float
    eval_mean_quality: float
    eval_mean_cost_usd: float
    artifact_path: str | None = None


def _label_best_allowed_action(state: RouterState) -> int:
    mask, _ = compute_action_mask(state)
    valid = np.where(mask)[0]
    scores = [estimate_candidate_outcome(state, TIER_LIST[a])[4] for a in valid]
    return int(valid[int(np.argmax(scores))])


def train_supervised_router(
    num_episodes: int = 120,
    epochs: int = 20,
    lr: float = 3e-3,
    seed: int = 42,
    save_path: Path | None = None,
    transitions: list[RLTransition] | None = None,
) -> tuple[SupervisedRouterNet, SupervisedTrainReport]:
    """Entraîne le routeur supervisé PyTorch avec split strict par dossier (`episode_id`)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = transitions or generate_routing_dataset(num_episodes=num_episodes, seed=seed)
    episode_ids = sorted({t.episode_id for t in data})
    split_idx = max(1, int(0.8 * len(episode_ids)))
    train_eps = set(episode_ids[:split_idx])

    x_train, y_train, x_val, y_val = [], [], [], []
    for t in data:
        vec = encode_state(t.state)
        label = _label_best_allowed_action(t.state)
        if t.episode_id in train_eps:
            x_train.append(vec)
            y_train.append(label)
        else:
            x_val.append(vec)
            y_val.append(label)

    X_tr = torch.tensor(np.asarray(x_train), dtype=torch.float32)
    y_tr = torch.tensor(np.asarray(y_train), dtype=torch.long)
    X_va = torch.tensor(np.asarray(x_val), dtype=torch.float32)
    y_va = torch.tensor(np.asarray(y_val), dtype=torch.long)

    model = SupervisedRouterNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_va), dim=1)
        val_acc = float((val_preds == y_va).float().mean().item()) if len(y_va) > 0 else 0.0

    # Évaluation en simulation sur 30 épisodes indépendants
    eval_env = ACIRoutingEnv(seed=seed + 1000)
    rewards, qualities, costs = [], [], []
    for _ in range(30):
        _, info = eval_env.reset()
        done = False
        while not done:
            tier, _ = model.predict_tier(info["router_state"])
            action = TIER_LIST.index(tier)
            _, r, terminated, truncated, info = eval_env.step(action)
            rewards.append(r)
            qualities.append(info["quality"])
            costs.append(info["cost_usd"])
            done = terminated or truncated

    artifact_str: str | None = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        artifact_str = str(save_path)

    return model, SupervisedTrainReport(
        epochs=epochs,
        train_samples=len(x_train),
        val_samples=len(x_val),
        val_accuracy=round(val_acc, 4),
        eval_mean_reward=round(float(np.mean(rewards)), 4),
        eval_mean_quality=round(float(np.mean(qualities)), 4),
        eval_mean_cost_usd=round(float(np.mean(costs)), 4),
        artifact_path=artifact_str,
    )
