"""LLM client for Thet's pipeline — routes through Model Broker.

Provides a synchronous interface that Thet's pipeline files call instead
of OpenAI directly. When MODEL_BROKER_URL is set, routes through the
Model Broker for token tracking, tier routing, cost control, and Langfuse
tracing. Falls back to direct OpenAI when MODEL_BROKER_URL is unset.

Architecture Invariant #6: ALL LLM calls go through the Model Broker.
Exception: OpenAI Moderation API (free, not LLM) stays direct.
Exception: Visual understanding (multimodal image input) stays direct
until Model Broker supports image payloads.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx

from af_shared.utils.schema_compat import clean_for_gemini

MODEL_BROKER_URL = os.getenv("MODEL_BROKER_URL", "")
_TIMEOUT = float(os.getenv("MODEL_BROKER_TIMEOUT", "60"))


@dataclass
class LlmResponse:
    """Unified response from either Model Broker or direct OpenAI."""

    content: str
    model_used: str = "unknown"
    model_tier: str = "HIGH"
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class LlmClientStats:
    """Accumulated token/cost stats for telemetry."""

    request_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


# Module-level stats accumulator
_stats = LlmClientStats()


def get_stats() -> LlmClientStats:
    """Return accumulated LLM call stats for this process."""
    return _stats


def generate(
    *,
    task_key: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    response_format: str | None = None,
    response_schema: dict | None = None,
    prompt_version: str = "validator/pipeline@v1",
    session_id: str | None = None,
) -> LlmResponse:
    """Send a generation request through Model Broker.

    If MODEL_BROKER_URL is set, routes through Model Broker.
    Otherwise falls back to direct OpenAI (for local dev without Broker).
    """
    session = session_id or os.getenv("CURRENT_WORKFLOW_ID", "unknown")

    if MODEL_BROKER_URL:
        return _call_model_broker(
            task_key=task_key,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            response_schema=response_schema,
            prompt_version=prompt_version,
            session_id=session,
        )

    return _call_openai_direct(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        response_schema=response_schema,
        task_key=task_key,
    )


def generate_structured(
    *,
    task_key: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    prompt_version: str = "validator/pipeline@v1",
) -> tuple[object, LlmResponse]:
    """Generate structured output conforming to a Pydantic model.

    Returns (parsed_model_instance, raw_llm_response).
    The Pydantic model is used to generate the JSON schema for the
    Model Broker and to validate/parse the response.
    """
    schema = clean_for_gemini(response_model.model_json_schema())

    resp = generate(
        task_key=task_key,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format="json",
        response_schema=schema,
        prompt_version=prompt_version,
    )

    parsed = response_model.model_validate_json(resp.content)
    return parsed, resp


# ---------------------------------------------------------------------------
# Model Broker path
# ---------------------------------------------------------------------------


def _call_model_broker(
    *,
    task_key: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    response_format: str | None,
    response_schema: dict | None,
    prompt_version: str,
    session_id: str,
) -> LlmResponse:
    """Route through Model Broker HTTP API."""
    body: dict = {
        "task_key": task_key,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "session_id": session_id,
        "agent_id": "validator-agent",
        "prompt_version": prompt_version,
    }
    if system_prompt:
        body["system_prompt"] = system_prompt
    if response_format:
        body["response_format"] = response_format
    if response_schema:
        body["response_schema"] = response_schema

    with httpx.Client(base_url=MODEL_BROKER_URL, timeout=_TIMEOUT) as client:
        response = client.post("/api/v1/generate", json=body)
        response.raise_for_status()
        data = response.json()

    tokens = data.get("token_usage", {})
    tokens_total = tokens.get("total_tokens", 0)
    cost = data.get("cost_usd", 0.0)

    _stats.request_count += 1
    _stats.total_tokens += tokens_total
    _stats.total_cost_usd += cost

    return LlmResponse(
        content=data["content"],
        model_used=data.get("model_used", "unknown"),
        model_tier=data.get("model_tier", "HIGH"),
        tokens_input=tokens.get("prompt_tokens", 0),
        tokens_output=tokens.get("completion_tokens", 0),
        cost_usd=cost,
        latency_ms=data.get("latency_ms", 0),
    )


# ---------------------------------------------------------------------------
# Direct OpenAI fallback (local dev without Model Broker)
# ---------------------------------------------------------------------------


def _call_openai_direct(
    *,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    response_format: str | None,
    response_schema: dict | None,
    task_key: str,
) -> LlmResponse:
    """Fall back to direct OpenAI when Model Broker is not available."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return LlmResponse(content="{}", model_used="no-key")

    # Pick model based on task_key tier hint
    model = "gpt-4o-mini" if "classifier" in task_key or "classification" in task_key else "gpt-4o"

    client = OpenAI(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    usage = response.usage

    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    _stats.request_count += 1
    _stats.total_tokens += (tokens_in + tokens_out)

    return LlmResponse(
        content=choice.message.content or "{}",
        model_used=response.model,
        model_tier="CHEAP" if "mini" in model else "HIGH",
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=0.0,
        latency_ms=0,
    )
