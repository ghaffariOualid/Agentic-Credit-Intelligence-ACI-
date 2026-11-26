"""Application FastAPI pour l'Inference Gateway ACI (POST /v1/chat/completions)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from services.contracts.inference import ChatCompletionRequest, ChatCompletionResponse
from services.inference.config import InferenceSettings
from services.inference.gateway import (
    InferenceGateway,
    QuotaExceededError,
    UpstreamInferenceError,
)


def create_app(gateway: InferenceGateway | None = None) -> FastAPI:
    app = FastAPI(
        title="ACI Inference Gateway",
        version="0.1.0",
        description="Passerelle OpenAI-compatible pour vLLM (Colab T4) avec quotas et fallback.",
    )
    gw = gateway or InferenceGateway(InferenceSettings())

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "mode": gw.settings.inference_mode,
            "vllm_configured": bool(gw.settings.vllm_base_url.strip()),
            "gpu_profile": "nvidia-t4-16gb-colab",
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": "small", "resolved_model": gw.settings.model_small, "object": "model"},
                {"id": "medium", "resolved_model": gw.settings.model_medium, "object": "model"},
                {"id": "large", "resolved_model": gw.settings.model_large, "object": "model"},
            ],
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return gw.metrics.snapshot()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            return await gw.chat_completions(request)
        except QuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except UpstreamInferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
