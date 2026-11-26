"""Contrats Pydantic pour le Decision Engine, le routeur et le RL (Pages 8, 9, 12)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RouterState(BaseModel):
    """Observation normalisée (État s) du moteur de décision (Page 8)."""

    task_type: str = Field(..., description="Ex: 'financial_extraction', 'risk_analysis', 'verification'")
    context_tokens: int = Field(..., ge=0)
    estimated_complexity: float = Field(..., ge=0.0, le=1.0)
    gpu_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    vram_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_depth: int = Field(default=0, ge=0)
    remaining_budget_usd: float = Field(default=1.0, ge=0.0)
    step_index: int = Field(default=0, ge=0)
    previous_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RouterAction(BaseModel):
    """Action (a) sélectionnée par la politique avec masquage des actions interdites."""

    selected_tier: ModelTier
    invoke_optional_agent: bool = False
    run_verification: bool = True
    retry: bool = False
    parallelize: bool = False
    masked_actions: list[str] = Field(default_factory=list)


class RouterCandidateScore(BaseModel):
    """Estimation par candidat pour explicabilité UI (Maquette 04, Page 9)."""

    tier: ModelTier
    predicted_quality: float = Field(..., ge=0.0, le=1.0)
    predicted_latency_s: float = Field(..., ge=0.0)
    predicted_cost_usd: float = Field(..., ge=0.0)
    score: float
    allowed: bool = True


class RouterDecision(BaseModel):
    """Réponse structurée de POST /api/v1/router/predict."""

    decision_id: str
    policy_version: str
    action: RouterAction
    candidates: list[RouterCandidateScore] = Field(default_factory=list)
    rationale: str
    constraints_applied: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RLTransition(BaseModel):
    """Contrat de transition RL reproductible (Page 12)."""

    episode_id: str = Field(..., min_length=1)
    state: RouterState
    action: RouterAction
    reward: float
    next_state: RouterState | None = None
    done: bool
    model_version: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)
    quality: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)
    gpu_seconds: float = Field(..., ge=0.0)
    masked_actions: list[str] = Field(default_factory=list)
    measurement_context: dict[str, Any] = Field(default_factory=dict)
