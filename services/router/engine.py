"""Moteur de décision inférence + entraînement contrôlé (Supervisé / Bandit / PPO)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from intelligence.decision_model.bandit import LinUCBRouter, train_bandit_router
from intelligence.decision_model.features import (
    TIER_LIST,
    compute_action_mask,
)
from intelligence.decision_model.supervised import (
    SupervisedRouterNet,
    train_supervised_router,
)
from intelligence.reinforcement_learning.env import estimate_candidate_outcome
from intelligence.reinforcement_learning.ppo import (
    ActorCriticNet,
    PPOConfig,
    train_ppo_router,
)
from services.contracts.intelligence import (
    ModelTier,
    RouterAction,
    RouterCandidateScore,
    RouterDecision,
    RouterState,
)


class TrainingJobRequest(BaseModel):
    algorithm: Literal["supervised", "bandit", "ppo"] = "ppo"
    episodes: int = Field(default=64, ge=16, le=20000)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.1)
    gamma: float = Field(default=0.99, ge=0.5, le=1.0)
    seed: int = Field(default=42)
    activate_policy: bool = Field(
        default=False,
        description="Si False, la politique reste en mode non déployée / évaluation requise.",
    )


class TrainingJobRecord(BaseModel):
    job_id: str
    algorithm: Literal["supervised", "bandit", "ppo"]
    policy_version: str
    episodes: int
    mean_reward: float
    mean_quality: float
    mean_cost_usd: float
    relative_cost: float
    reward_curve: list[float] = Field(default_factory=list)
    artifact_path: str | None = None
    deployed: bool = False
    status: Literal["completed", "needs_review"] = "needs_review"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionEngineService:
    """Gère la prédiction explicable (/router/predict) et les entraînements (/training/jobs)."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or (
            Path(__file__).resolve().parents[2] / "intelligence" / "artifacts"
        )
        self.active_policy_version: str = "router-v01-heuristic"
        self.active_policy_type: Literal["heuristic", "supervised", "bandit", "ppo"] = "heuristic"
        self._active_model: SupervisedRouterNet | LinUCBRouter | ActorCriticNet | None = None
        self._jobs: list[TrainingJobRecord] = []
        self._decision_counter: int = 0

    def predict(self, state: RouterState) -> RouterDecision:
        self._decision_counter += 1
        decision_id = f"D-{self._decision_counter:04d}"
        mask, constraints = compute_action_mask(state)

        candidates: list[RouterCandidateScore] = []
        best_heuristic_tier = ModelTier.MEDIUM
        best_heuristic_score = -1e9

        for idx, tier in enumerate(TIER_LIST):
            q, lat_s, cost_usd, _err, score = estimate_candidate_outcome(state, tier)
            allowed = bool(mask[idx])
            candidates.append(
                RouterCandidateScore(
                    tier=tier,
                    predicted_quality=q,
                    predicted_latency_s=lat_s,
                    predicted_cost_usd=cost_usd,
                    score=round(score, 4),
                    allowed=allowed,
                )
            )
            if allowed and score > best_heuristic_score:
                best_heuristic_score = score
                best_heuristic_tier = tier

        if self._active_model is not None:
            selected_tier, _ = self._active_model.predict_tier(state)
            # Garde-fou strict : jamais d'action masquée
            if not mask[TIER_LIST.index(selected_tier)]:
                selected_tier = best_heuristic_tier
        else:
            selected_tier = best_heuristic_tier

        masked_names = [TIER_LIST[i].value for i in range(len(TIER_LIST)) if not mask[i]]
        selected_cand = next(c for c in candidates if c.tier == selected_tier)
        rationale = (
            f"{selected_tier.value.capitalize()} sélectionné par {self.active_policy_version} "
            f"(qualité estimée {selected_cand.predicted_quality:.2f}, "
            f"latence {selected_cand.predicted_latency_s:.1f} s, "
            f"coût {selected_cand.predicted_cost_usd:.2f} $)."
        )

        return RouterDecision(
            decision_id=decision_id,
            policy_version=self.active_policy_version,
            action=RouterAction(
                selected_tier=selected_tier,
                run_verification=True,
                masked_actions=masked_names,
            ),
            candidates=candidates,
            rationale=rationale,
            constraints_applied=constraints,
        )

    def run_training_job(self, req: TrainingJobRequest) -> TrainingJobRecord:
        job_num = len(self._jobs) + 1
        job_id = f"EXP-{job_num:03d}"
        policy_version = f"policy-{req.algorithm}-v{job_num:02d}"

        if req.algorithm == "supervised":
            save_path = self.artifacts_dir / f"{policy_version}.pt"
            model, rep = train_supervised_router(
                num_episodes=req.episodes,
                lr=max(req.learning_rate, 1e-3),
                seed=req.seed,
                save_path=save_path,
            )
            record = TrainingJobRecord(
                job_id=job_id,
                algorithm="supervised",
                policy_version=policy_version,
                episodes=req.episodes,
                mean_reward=rep.eval_mean_reward,
                mean_quality=rep.eval_mean_quality,
                mean_cost_usd=rep.eval_mean_cost_usd,
                relative_cost=round(rep.eval_mean_cost_usd / 0.45, 4),
                reward_curve=[rep.eval_mean_reward],
                artifact_path=rep.artifact_path,
                deployed=req.activate_policy,
                status="completed" if req.activate_policy else "needs_review",
            )
            if req.activate_policy:
                self._active_model = model
                self.active_policy_type = "supervised"
                self.active_policy_version = policy_version

        elif req.algorithm == "bandit":
            model, rep = train_bandit_router(num_episodes=req.episodes, seed=req.seed)
            record = TrainingJobRecord(
                job_id=job_id,
                algorithm="bandit",
                policy_version=policy_version,
                episodes=req.episodes,
                mean_reward=rep.eval_mean_reward,
                mean_quality=rep.eval_mean_quality,
                mean_cost_usd=rep.eval_mean_cost_usd,
                relative_cost=round(rep.eval_mean_cost_usd / 0.45, 4),
                reward_curve=[rep.eval_mean_reward],
                artifact_path=None,
                deployed=req.activate_policy,
                status="completed" if req.activate_policy else "needs_review",
            )
            if req.activate_policy:
                self._active_model = model
                self.active_policy_type = "bandit"
                self.active_policy_version = policy_version

        else:  # ppo
            save_path = self.artifacts_dir / f"{policy_version}.pt"
            cfg = PPOConfig(
                episodes=req.episodes,
                learning_rate=req.learning_rate,
                gamma=req.gamma,
                seed=req.seed,
            )
            model, rep = train_ppo_router(config=cfg, save_path=save_path)
            record = TrainingJobRecord(
                job_id=job_id,
                algorithm="ppo",
                policy_version=policy_version,
                episodes=req.episodes,
                mean_reward=rep.mean_reward,
                mean_quality=rep.mean_quality,
                mean_cost_usd=rep.mean_cost_usd,
                relative_cost=rep.relative_cost_vs_static_large,
                reward_curve=rep.reward_curve,
                artifact_path=rep.artifact_path,
                deployed=req.activate_policy,
                status="completed" if req.activate_policy else "needs_review",
            )
            if req.activate_policy:
                self._active_model = model
                self.active_policy_type = "ppo"
                self.active_policy_version = policy_version

        self._jobs.append(record)
        return record

    def list_jobs(self) -> list[TrainingJobRecord]:
        return list(self._jobs)
