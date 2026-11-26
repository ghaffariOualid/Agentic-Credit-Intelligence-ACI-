"""Tests de l'Inference Gateway OpenAI-compatible (vLLM Colab T4 + fallback + quotas)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from services.contracts.inference import ChatCompletionRequest, ChatMessage
from services.inference.app import create_app
from services.inference.config import InferenceSettings
from services.inference.gateway import InferenceGateway


def test_gateway_health_and_models_endpoints() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="auto", vllm_base_url=""))
    client = TestClient(create_app(gw))

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["gpu_profile"] == "nvidia-t4-16gb-colab"

    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    ids = [m["id"] for m in r_models.json()["data"]]
    assert ids == ["small", "medium", "large"]


def test_gateway_auto_fallback_to_local_mock_with_json_format() -> None:
    gw = InferenceGateway(InferenceSettings(inference_mode="auto", vllm_base_url=""))
    client = TestClient(create_app(gw))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "medium",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "Extraire le chiffre d'affaires de Atlas Industrie."}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["aci_meta"]["backend"] == "local_mock"
    assert body["aci_meta"]["fallback_used"] is True
    assert body["aci_meta"]["requested_tier"] == "medium"
    parsed = json.loads(body["choices"][0]["message"]["content"])
    assert parsed["mode"] == "local_mock"


@pytest.mark.asyncio
async def test_gateway_routes_to_vllm_colab_t4_when_configured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers.get("Authorization") == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-vllm-123",
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"revenue": 12000000}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        gw = InferenceGateway(
            settings=InferenceSettings(
                inference_mode="vllm",
                vllm_base_url="https://demo-t4.trycloudflare.com/v1",
                vllm_api_key="test-key",
            ),
            http_client=mock_http,
        )
        res = await gw.chat_completions(
            ChatCompletionRequest(
                model="medium",
                messages=[ChatMessage(role="user", content="Test T4")],
            )
        )
        assert res.aci_meta.backend == "vllm_colab_t4"
        assert res.aci_meta.fallback_used is False
        assert res.usage.total_tokens == 52


def test_gateway_enforces_rpm_quota() -> None:
    gw = InferenceGateway(
        InferenceSettings(inference_mode="mock", inference_rate_limit_rpm=2)
    )
    client = TestClient(create_app(gw))
    payload = {"model": "small", "messages": [{"role": "user", "content": "ping"}]}

    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 429
