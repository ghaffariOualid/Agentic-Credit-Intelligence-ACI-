"""Contrats Pydantic OpenAI-compatible pour l'Inference Gateway (Page 13)."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.contracts.intelligence import ModelTier


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Requête compatible POST /v1/chat/completions."""

    model: str = Field(
        default="medium",
        description="Tier ACI ('small', 'medium', 'large') ou nom exact de modèle vLLM",
    )
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class InferenceGatewayMeta(BaseModel):
    """Métadonnées techniques ajoutées par la Gateway pour observabilité et traces RL."""

    requested_tier: ModelTier
    resolved_model: str
    backend: Literal["vllm_colab_t4", "external_fallback", "local_mock"]
    latency_ms: float = Field(..., ge=0.0)
    fallback_used: bool = False
    fallback_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Réponse compatible OpenAI enrichie des métadonnées ACI."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    aci_meta: InferenceGatewayMeta
