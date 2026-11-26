"""Simulateur Gymnasium calibré pour 1× NVIDIA T4 (16 Go VRAM) — Étape 3 RL (Page 8).

Modélise un épisode séquentiel de 4 étapes par dossier de crédit :
  0: financial_extraction
  1: risk_analysis
  2: verification
  3: report_synthesis

Pour chaque étape, l'environnement simule qualité, latence, coût GPU-seconde et erreurs
selon la complexité du dossier, la charge GPU/VRAM T4 et le tier choisi (small/medium/large).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from intelligence.decision_model.features import (
    STATE_DIM,
    TASK_TYPES,
    TIER_LIST,
    compute_action_mask,
    compute_reward,
    encode_state,
)
from services.contracts.intelligence import (
    ModelTier,
    RLTransition,
    RouterAction,
    RouterState,
)


@dataclass(frozen=True)
class T4TierProfile:
    tier: ModelTier
    model_name: str
    base_quality: float
    complexity_penalty: float
    base_latency_s: float
    cost_usd: float
    vram_footprint: float


T4_PROFILES: dict[ModelTier, T4TierProfile] = {
    ModelTier.SMALL: T4TierProfile(
        tier=ModelTier.SMALL,
        model_name="Qwen/Qwen2.5-3B-Instruct-AWQ",
        base_quality=0.93,
        complexity_penalty=0.28,
        base_latency_s=0.9,
        cost_usd=0.04,
        vram_footprint=0.22,
    ),
    ModelTier.MEDIUM: T4TierProfile(
        tier=ModelTier.MEDIUM,
        model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
        base_quality=0.97,
        complexity_penalty=0.07,
        base_latency_s=2.3,
        cost_usd=0.14,
        vram_footprint=0.45,
    ),
    ModelTier.LARGE: T4TierProfile(
        tier=ModelTier.LARGE,
        model_name="Qwen/Qwen2.5-14B-Instruct-AWQ",
        base_quality=0.99,
        complexity_penalty=0.02,
        base_latency_s=5.1,
        cost_usd=0.45,
        vram_footprint=0.72,
    ),
}


def estimate_candidate_outcome(
    state: RouterState, tier: ModelTier
) -> tuple[float, float, float, float, float]:
    """Estime (quality, latency_s, cost_usd, error_rate, score) déterministes pour un candidat."""
    prof = T4_PROFILES[tier]
    c = state.estimated_complexity
    ctx_factor = 1.0 + 0.35 * min(state.context_tokens / 8192.0, 1.5)
    queue_factor = 1.0 + 0.04 * state.queue_depth + 0.25 * state.gpu_utilization

    quality = float(np.clip(prof.base_quality - prof.complexity_penalty * (c**1.5), 0.45, 0.995))
    if state.previous_confidence < 0.75 and tier == ModelTier.SMALL:
        quality = max(0.45, quality - 0.08)

    latency_s = round(prof.base_latency_s * ctx_factor * queue_factor, 3)
    cost_usd = round(prof.cost_usd * (1.0 + 0.2 * (state.context_tokens / 8192.0)), 4)
    error_rate = float(np.clip(max(0.0, (0.90 - quality) * 1.4), 0.0, 1.0))

    # Normalisation pour la formule de reward
    cost_norm = min(cost_usd / 0.50, 1.0)
    latency_norm = min(latency_s / 8.0, 1.0)
    score = compute_reward(quality, cost_norm, latency_norm, error_rate)
    return round(quality, 4), latency_s, cost_usd, round(error_rate, 4), score


class ACIRoutingEnv(gym.Env):
    """Environnement Gymnasium de routage multi-étapes sur GPU T4."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(TIER_LIST))
        self._rng = np.random.default_rng(seed)
        self._episode_idx = 0
        self._step_idx = 0
        self._case_complexity = 0.5
        self._state: RouterState | None = None

    def _build_step_state(self, step_idx: int, prev_conf: float, budget: float) -> RouterState:
        task_type = TASK_TYPES[min(step_idx, len(TASK_TYPES) - 1)]
        # La complexité varie légèrement par étape autour de la complexité du dossier
        step_comp = float(
            np.clip(self._case_complexity + self._rng.normal(0.0, 0.08), 0.05, 0.98)
        )
        context_tokens = int(np.clip(self._rng.normal(4200 + 3500 * step_comp, 900), 600, 14000))
        gpu_util = float(np.clip(self._rng.beta(2.5, 2.0), 0.10, 0.96))
        vram_util = float(np.clip(0.35 + 0.55 * gpu_util + self._rng.normal(0.0, 0.05), 0.20, 0.95))
        queue_depth = int(np.clip(self._rng.poisson(lam=max(1.0, 14.0 * gpu_util)), 0, 30))
        return RouterState(
            task_type=task_type,
            context_tokens=context_tokens,
            estimated_complexity=round(step_comp, 4),
            gpu_utilization=round(gpu_util, 4),
            vram_utilization=round(vram_util, 4),
            queue_depth=queue_depth,
            remaining_budget_usd=round(max(0.0, budget), 4),
            step_index=step_idx,
            previous_confidence=round(prev_conf, 4),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode_idx += 1
        self._step_idx = 0
        self._case_complexity = float(self._rng.uniform(0.15, 0.95))
        self._state = self._build_step_state(step_idx=0, prev_conf=0.95, budget=1.20)
        mask, reasons = compute_action_mask(self._state)
        return encode_state(self._state), {
            "router_state": self._state,
            "action_mask": mask,
            "constraints": reasons,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Appeler reset() avant step()."
        prev_state = self._state
        mask, reasons = compute_action_mask(prev_state)
        action_idx = int(action)

        violation_penalty = 0.0
        if not mask[action_idx]:
            # En cas d'action interdite tentée, pénalité forte et fallback sécurisé sur medium
            violation_penalty = 0.35
            action_idx = 1

        tier = TIER_LIST[action_idx]
        q, lat_s, cost_usd, err, base_reward = estimate_candidate_outcome(prev_state, tier)
        # Bruit réaliste d'exécution GPU T4
        q_noisy = float(np.clip(q + self._rng.normal(0.0, 0.012), 0.40, 1.0))
        lat_noisy = max(0.2, round(lat_s * float(self._rng.uniform(0.92, 1.08)), 3))
        reward = round(base_reward - violation_penalty, 6)

        next_budget = max(0.0, prev_state.remaining_budget_usd - cost_usd)
        self._step_idx += 1
        terminated = self._step_idx >= len(TASK_TYPES)
        truncated = False

        if terminated:
            next_state = None
            next_obs = np.zeros(STATE_DIM, dtype=np.float32)
            next_mask = np.array([True, True, True], dtype=bool)
        else:
            next_state = self._build_step_state(
                step_idx=self._step_idx, prev_conf=q_noisy, budget=next_budget
            )
            self._state = next_state
            next_obs = encode_state(next_state)
            next_mask, _ = compute_action_mask(next_state)

        transition = RLTransition(
            episode_id=f"EP-{self._episode_idx:05d}",
            state=prev_state,
            action=RouterAction(
                selected_tier=tier,
                run_verification=True,
                masked_actions=[
                    TIER_LIST[i].value for i in range(len(TIER_LIST)) if not mask[i]
                ],
            ),
            reward=reward,
            next_state=next_state,
            done=terminated,
            model_version=T4_PROFILES[tier].model_name,
            policy_version="simulator-t4-v1",
            quality=round(q_noisy, 4),
            latency_ms=round(lat_noisy * 1000.0, 2),
            gpu_seconds=lat_noisy,
            masked_actions=[TIER_LIST[i].value for i in range(len(TIER_LIST)) if not mask[i]],
            measurement_context={"gpu": "nvidia-t4-16gb", "cost_usd": cost_usd, "error_rate": err},
        )

        return (
            next_obs,
            reward,
            terminated,
            truncated,
            {
                "router_state": next_state,
                "action_mask": next_mask,
                "quality": q_noisy,
                "cost_usd": cost_usd,
                "latency_s": lat_noisy,
                "transition": transition,
                "constraints": reasons,
            },
        )


def generate_routing_dataset(num_episodes: int = 150, seed: int = 42) -> list[RLTransition]:
    """Collecte un dataset reproductible de transitions RL sur le simulateur T4."""
    env = ACIRoutingEnv(seed=seed)
    rng = np.random.default_rng(seed)
    transitions: list[RLTransition] = []

    for _ in range(num_episodes):
        _, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            valid_actions = np.where(mask)[0]
            # 50 % exploration uniforme autorisée, 50 % oracle approché pour supervisé
            if rng.random() < 0.5:
                action = int(rng.choice(valid_actions))
            else:
                state = info["router_state"]
                scores = [
                    estimate_candidate_outcome(state, TIER_LIST[a])[4] for a in valid_actions
                ]
                action = int(valid_actions[int(np.argmax(scores))])
            _, _, terminated, truncated, info = env.step(action)
            transitions.append(info["transition"])
            done = terminated or truncated

    return transitions
