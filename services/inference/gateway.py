"""Inference Gateway OpenAI-compatible avec routage de tiers, quotas et fallback T4 Colab."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from services.contracts.inference import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    InferenceGatewayMeta,
    UsageInfo,
)
from services.contracts.intelligence import ModelTier
from services.inference.config import InferenceSettings


class QuotaExceededError(RuntimeError):
    """Levée lorsque la limite de requêtes par minute (RPM) est atteinte."""


class UpstreamInferenceError(RuntimeError):
    """Levée lorsqu'aucun backend n'est disponible en mode strict (`vllm`)."""


@dataclass
class GatewayMetrics:
    total_requests: int = 0
    requests_by_tier: dict[str, int] = field(
        default_factory=lambda: {"small": 0, "medium": 0, "large": 0}
    )
    fallback_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    in_flight: int = 0
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        avg_latency = (
            round(self.total_latency_ms / self.total_requests, 2)
            if self.total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "requests_by_tier": dict(self.requests_by_tier),
            "fallback_count": self.fallback_count,
            "error_count": self.error_count,
            "avg_latency_ms": avg_latency,
            "in_flight": self.in_flight,
            "last_error": self.last_error,
        }


class InferenceGateway:
    """Routeur d'inférence vers vLLM (Colab T4), fallback externe ou mock local déterministe."""

    def __init__(
        self,
        settings: InferenceSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or InferenceSettings()
        self._client = http_client
        self._semaphore = asyncio.Semaphore(self.settings.inference_max_concurrent)
        self._request_timestamps: deque[float] = deque()
        self.metrics = GatewayMetrics()

    def resolve_tier_and_model(self, requested_model: str) -> tuple[ModelTier, str]:
        normalized = requested_model.strip().lower()
        if normalized == ModelTier.SMALL.value or normalized == "vllm-small":
            return ModelTier.SMALL, self.settings.model_small
        if normalized == ModelTier.MEDIUM.value or normalized == "vllm-medium":
            return ModelTier.MEDIUM, self.settings.model_medium
        if normalized == ModelTier.LARGE.value or normalized == "vllm-large":
            return ModelTier.LARGE, self.settings.model_large

        # Nom de modèle explicite passé directement
        if "3b" in normalized or "small" in normalized:
            return ModelTier.SMALL, requested_model
        if "14b" in normalized or "32b" in normalized or "large" in normalized:
            return ModelTier.LARGE, requested_model
        return ModelTier.MEDIUM, requested_model

    def _check_rate_limit(self) -> None:
        now = time.monotonic()
        window_start = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] < window_start:
            self._request_timestamps.popleft()
        if len(self._request_timestamps) >= self.settings.inference_rate_limit_rpm:
            raise QuotaExceededError(
                f"Quota dépassé : {self.settings.inference_rate_limit_rpm} requêtes/min maximum."
            )
        self._request_timestamps.append(now)

    async def _call_openai_backend(
        self,
        base_url: str,
        api_key: str,
        resolved_model: str,
        request: ChatCompletionRequest,
    ) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.extra_body:
            payload.update(request.extra_body)

        last_exc: Exception | None = None
        for attempt in range(self.settings.inference_max_retries + 1):
            try:
                if self._client is not None:
                    resp = await self._client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.settings.inference_timeout_seconds,
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self.settings.inference_timeout_seconds
                    ) as client:
                        resp = await client.post(url, json=payload, headers=headers)

                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.settings.inference_max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))

        raise UpstreamInferenceError(f"Échec d'appel backend {url}: {last_exc}")

    def _build_mock_response(
        self,
        request: ChatCompletionRequest,
        tier: ModelTier,
        resolved_model: str,
        latency_ms: float,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> ChatCompletionResponse:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        wants_json = (
            request.response_format is not None
            and request.response_format.get("type") in {"json_object", "json_schema"}
        )

        if wants_json:
            content = json.dumps(
                {
                    "mode": "local_mock",
                    "tier": tier.value,
                    "status": "ok",
                    "summary": "Réponse structurée simulée (Colab T4 hors ligne).",
                    "input_preview": last_user[:120],
                },
                ensure_ascii=False,
            )
        else:
            content = (
                f"[ACI Mock/{tier.value}] Analyse préparatoire générée localement "
                f"(aucune décision automatique de crédit). Entrée reçue : {last_user[:120]}"
            )

        prompt_tokens = max(1, sum(len(m.content.split()) for m in request.messages))
        completion_tokens = max(1, len(content.split()))

        return ChatCompletionResponse(
            id=f"chatcmpl-aci-{uuid.uuid4().hex[:12]}",
            model=resolved_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            aci_meta=InferenceGatewayMeta(
                requested_tier=tier,
                resolved_model=resolved_model,
                backend="local_mock",
                latency_ms=round(latency_ms, 2),
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            ),
        )

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._check_rate_limit()
        tier, resolved_model = self.resolve_tier_and_model(request.model)

        t0 = time.perf_counter()
        self.metrics.in_flight += 1
        try:
            async with self._semaphore:
                if self.settings.inference_mode == "mock":
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    resp = self._build_mock_response(
                        request, tier, resolved_model, elapsed_ms, fallback_used=False, fallback_reason=None
                    )
                    self._record_success(tier, elapsed_ms, fallback=False)
                    return resp

                # Sélection de la cible upstream (Colab T4 par défaut, ou fallback externe pour Large)
                backend_label = "vllm_colab_t4"
                base_url = self.settings.vllm_base_url.strip()
                api_key = self.settings.vllm_api_key

                if (
                    tier == ModelTier.LARGE
                    and self.settings.large_fallback_base_url.strip()
                    and not base_url
                ):
                    base_url = self.settings.large_fallback_base_url.strip()
                    api_key = self.settings.large_fallback_api_key
                    backend_label = "external_fallback"

                if base_url:
                    try:
                        raw = await self._call_openai_backend(
                            base_url, api_key, resolved_model, request
                        )
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        choices = [
                            ChatCompletionChoice(
                                index=c.get("index", idx),
                                message=ChatMessage(**c["message"]),
                                finish_reason=c.get("finish_reason", "stop"),
                            )
                            for idx, c in enumerate(raw.get("choices", []))
                        ]
                        usage = UsageInfo(**raw["usage"]) if "usage" in raw else UsageInfo()
                        self._record_success(tier, elapsed_ms, fallback=False)
                        return ChatCompletionResponse(
                            id=raw.get("id", f"chatcmpl-aci-{uuid.uuid4().hex[:12]}"),
                            model=raw.get("model", resolved_model),
                            choices=choices,
                            usage=usage,
                            aci_meta=InferenceGatewayMeta(
                                requested_tier=tier,
                                resolved_model=resolved_model,
                                backend=backend_label,  # type: ignore[arg-type]
                                latency_ms=round(elapsed_ms, 2),
                                fallback_used=False,
                            ),
                        )
                    except UpstreamInferenceError as exc:
                        self.metrics.last_error = str(exc)
                        if self.settings.inference_mode == "vllm":
                            self.metrics.error_count += 1
                            raise
                        # Mode auto -> fallback propre sur mock local
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        resp = self._build_mock_response(
                            request,
                            tier,
                            resolved_model,
                            elapsed_ms,
                            fallback_used=True,
                            fallback_reason=str(exc),
                        )
                        self._record_success(tier, elapsed_ms, fallback=True)
                        return resp

                # Aucune URL vLLM configurée
                if self.settings.inference_mode == "vllm":
                    self.metrics.error_count += 1
                    raise UpstreamInferenceError(
                        "ACI_VLLM_BASE_URL non configurée alors que ACI_INFERENCE_MODE=vllm."
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                resp = self._build_mock_response(
                    request,
                    tier,
                    resolved_model,
                    elapsed_ms,
                    fallback_used=True,
                    fallback_reason="ACI_VLLM_BASE_URL non configurée (Colab T4 hors ligne).",
                )
                self._record_success(tier, elapsed_ms, fallback=True)
                return resp
        finally:
            self.metrics.in_flight -= 1

    def _record_success(self, tier: ModelTier, elapsed_ms: float, fallback: bool) -> None:
        self.metrics.total_requests += 1
        self.metrics.requests_by_tier[tier.value] = (
            self.metrics.requests_by_tier.get(tier.value, 0) + 1
        )
        self.metrics.total_latency_ms += elapsed_ms
        if fallback:
            self.metrics.fallback_count += 1
