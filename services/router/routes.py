"""Routes FastAPI pour le routeur intelligent et le Studio RL (Page 13)."""

from __future__ import annotations

from fastapi import APIRouter

from services.contracts.intelligence import RouterDecision, RouterState
from services.router.engine import (
    DecisionEngineService,
    TrainingJobRecord,
    TrainingJobRequest,
)


def create_router_api(engine: DecisionEngineService | None = None) -> APIRouter:
    router = APIRouter(tags=["Decision Engine & Training"])
    svc = engine or DecisionEngineService()

    @router.post("/api/v1/router/predict", response_model=RouterDecision)
    async def predict_route(state: RouterState) -> RouterDecision:
        return svc.predict(state)

    @router.get("/api/v1/router/policy")
    async def get_active_policy() -> dict[str, object]:
        return {
            "policy_version": svc.active_policy_version,
            "policy_type": svc.active_policy_type,
            "jobs_count": len(svc.list_jobs()),
        }

    @router.post("/api/v1/training/jobs", response_model=TrainingJobRecord)
    async def launch_training_job(request: TrainingJobRequest) -> TrainingJobRecord:
        return svc.run_training_job(request)

    @router.get("/api/v1/training/jobs", response_model=list[TrainingJobRecord])
    async def list_training_jobs() -> list[TrainingJobRecord]:
        return svc.list_jobs()

    return router
