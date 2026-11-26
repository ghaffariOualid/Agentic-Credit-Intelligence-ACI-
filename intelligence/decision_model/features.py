"""Encodage de l'état s, masquage des actions interdites et fonction de reward (Page 8)."""

from __future__ import annotations

import numpy as np

from services.contracts.intelligence import ModelTier, RouterState

TASK_TYPES: tuple[str, ...] = (
    "financial_extraction",
    "risk_analysis",
    "verification",
    "report_synthesis",
)

TIER_LIST: tuple[ModelTier, ...] = (
    ModelTier.SMALL,
    ModelTier.MEDIUM,
    ModelTier.LARGE,
)

STATE_DIM: int = len(TASK_TYPES) + 8  # 4 one-hot + 8 métriques normalisées = 12


def encode_state(state: RouterState) -> np.ndarray:
    """Convertit un RouterState en vecteur float32 normalisé dans [0, 1]."""
    task_one_hot = [1.0 if state.task_type == t else 0.0 for t in TASK_TYPES]
    continuous = [
        min(state.context_tokens / 16384.0, 1.0),
        float(np.clip(state.estimated_complexity, 0.0, 1.0)),
        float(np.clip(state.gpu_utilization, 0.0, 1.0)),
        float(np.clip(state.vram_utilization, 0.0, 1.0)),
        min(state.queue_depth / 32.0, 1.0),
        min(state.remaining_budget_usd / 2.0, 1.0),
        min(state.step_index / 4.0, 1.0),
        float(np.clip(state.previous_confidence, 0.0, 1.0)),
    ]
    return np.asarray(task_one_hot + continuous, dtype=np.float32)


def compute_action_mask(state: RouterState) -> tuple[np.ndarray, list[str]]:
    """Calcule le masque booléen [small, medium, large] et les contraintes appliquées.

    Règles de sécurité métier (Page 8 & 9) :
    - Les contrôles obligatoires ne peuvent jamais être contournés.
    - `small` est interdit sur les tâches critiques à forte complexité (>= 0.80).
    - `large` est interdit si la VRAM T4 est saturée (>= 0.90) ou le budget résiduel < 0.05 $.
    """
    mask = np.array([True, True, True], dtype=bool)
    reasons: list[str] = ["Contrôle documentaire obligatoire (non contournable)."]

    if state.estimated_complexity >= 0.80 and state.task_type in {
        "financial_extraction",
        "risk_analysis",
    }:
        mask[0] = False
        reasons.append("Action 'small' masquée : complexité >= 0,80 sur tâche critique.")

    if state.vram_utilization >= 0.90:
        mask[2] = False
        reasons.append("Action 'large' masquée : VRAM GPU T4 >= 90 %.")

    if state.remaining_budget_usd < 0.05:
        mask[2] = False
        reasons.append("Action 'large' masquée : budget résiduel insuffisant (< 0,05 $).")

    # Sécurité : toujours au moins `medium` autorisé
    if not mask.any():
        mask[1] = True

    return mask, reasons


def compute_reward(
    quality: float,
    cost_norm: float,
    latency_norm: float,
    error_rate: float,
) -> float:
    """Fonction de reward pondérée initiale (Page 8) : 0,6Q - 0,15C - 0,15L - 0,1E."""
    return round(
        0.60 * float(quality)
        - 0.15 * float(cost_norm)
        - 0.15 * float(latency_norm)
        - 0.10 * float(error_rate),
        6,
    )
