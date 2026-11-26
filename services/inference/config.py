"""Configuration de l'Inference Gateway (adaptée à 1× NVIDIA T4 sur Google Colab)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_mode: Literal["auto", "vllm", "mock"] = Field(default="auto")
    vllm_base_url: str = Field(default="")
    vllm_api_key: str = Field(default="aci-colab-dev-key")

    model_small: str = Field(default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    model_medium: str = Field(default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    model_large: str = Field(default="Qwen/Qwen2.5-14B-Instruct-AWQ")

    large_fallback_base_url: str = Field(default="")
    large_fallback_api_key: str = Field(default="")

    inference_timeout_seconds: float = Field(default=45.0, gt=0)
    inference_max_retries: int = Field(default=2, ge=0, le=5)
    inference_rate_limit_rpm: int = Field(default=60, ge=1)
    inference_max_concurrent: int = Field(default=8, ge=1)
