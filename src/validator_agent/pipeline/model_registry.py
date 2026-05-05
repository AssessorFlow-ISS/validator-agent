"""Per-agent model registry with tier-based selection.

Replaces the centralized Model Broker proxy pattern. Each agent manages
its own LLM connections directly — no proxy bottleneck, no single point
of failure.

Tier mapping:
  CHEAP    → fast, low-cost tasks (page classification)
  EXPENSIVE → complex reasoning (content analyzers, synthesizer, visual understanding)

The base_url supports OpenAI-compatible APIs (OpenAI, Gemini, Azure OpenAI,
local vLLM), so switching providers is an env var change.

Architectural decision: Previously routed through a centralized Model Broker
service (POST /api/v1/generate). Removed because:
  1. Bottleneck — all agents funneling through one service
  2. Single point of failure — Model Broker down = all agents down
  3. Over-engineered — model selection is deterministic (cheap/expensive),
     not dynamic. A local registry is sufficient.
  4. Scalability — stateless agents should scale independently
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import cache

from openai import OpenAI


class ModelTier(StrEnum):
    """Model cost/capability tiers."""

    CHEAP = "cheap"
    EXPENSIVE = "expensive"


# ── Per-tier configuration from environment ──


def _get_tier_config(tier: ModelTier) -> dict[str, str]:
    """Return base_url, api_key, model_id for a given tier."""
    if tier is ModelTier.CHEAP:
        return {
            "base_url": os.getenv("LLM_CHEAP_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("LLM_CHEAP_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            "model_id": os.getenv("LLM_CHEAP_MODEL", "gpt-4.1-mini"),
        }
    return {
        "base_url": os.getenv("LLM_EXPENSIVE_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("LLM_EXPENSIVE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "model_id": os.getenv("LLM_EXPENSIVE_MODEL", "gpt-4.1"),
    }


# ── Tier assignment per pipeline component ──

_COMPONENT_TIER_MAP: dict[str, ModelTier] = {
    # Component 2b: Page classification (simple TEXT/VISUAL decision)
    "page_classifier": ModelTier.CHEAP,
    # Component 3b: Content safety analyzers (complex reasoning)
    "analyzer_child_safety": ModelTier.EXPENSIVE,
    "analyzer_content_quality": ModelTier.EXPENSIVE,
    "analyzer_legal_compliance": ModelTier.EXPENSIVE,
    "analyzer_singapore_compliance": ModelTier.EXPENSIVE,
    # Component 3c: Synthesizer (merges analyzer findings)
    "content_synthesizer": ModelTier.EXPENSIVE,
    # Component 2c: Visual understanding (multimodal)
    "visual_generator": ModelTier.EXPENSIVE,
    "visual_evaluator": ModelTier.EXPENSIVE,
}


def get_tier(component: str) -> ModelTier:
    """Return the model tier for a pipeline component."""
    return _COMPONENT_TIER_MAP.get(component, ModelTier.EXPENSIVE)


def get_model_id(component: str) -> str:
    """Return the model ID for a pipeline component."""
    tier = get_tier(component)
    return _get_tier_config(tier)["model_id"]


@cache
def _get_cached_client(tier: ModelTier) -> OpenAI:
    """Create and cache an OpenAI client per tier."""
    config = _get_tier_config(tier)
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
    )


def get_client(component: str) -> OpenAI:
    """Return the OpenAI client for a pipeline component."""
    tier = get_tier(component)
    return _get_cached_client(tier)


# ── Cost estimation (per 1M tokens) ──

_COST_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Gemini (via OpenAI-compatible API)
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


def estimate_cost(model_id: str, tokens_input: int, tokens_output: int) -> float:
    """Estimate USD cost from token counts.

    Returns 0.0 for unknown models (safe fallback).
    """
    # Normalize model_id:
    #   - Strip provider prefix from OpenRouter (e.g. "openai/gpt-4.1" -> "gpt-4.1")
    #   - Strip date suffixes (e.g. "gpt-4.1-2025-04-14" -> "gpt-4.1")
    base_model = model_id
    if "/" in base_model:
        base_model = base_model.split("/", 1)[1]
    for known in _COST_PER_1M_TOKENS:
        if base_model.startswith(known):
            base_model = known
            break

    rates = _COST_PER_1M_TOKENS.get(base_model, {"input": 0.0, "output": 0.0})
    return round(
        (tokens_input * rates["input"] + tokens_output * rates["output"]) / 1_000_000,
        6,
    )


# ── Moderation client (always OpenAI, always free) ──


@cache
def _get_moderation_client() -> OpenAI:
    """Moderation API is always OpenAI direct (free, not available on OpenRouter).

    Uses MODERATION_API_KEY (OpenAI direct key) with hardcoded api.openai.com base URL.
    Falls back to OPENAI_API_KEY if MODERATION_API_KEY is not set.
    """
    return OpenAI(
        api_key=os.getenv("MODERATION_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        base_url="https://api.openai.com/v1",
        timeout=30.0,
    )


def get_moderation_client() -> OpenAI:
    """Return the OpenAI client for moderation checks."""
    return _get_moderation_client()
