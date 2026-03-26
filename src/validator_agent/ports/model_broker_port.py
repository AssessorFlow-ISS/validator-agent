"""ModelBrokerPort — abstract interface for LLM inference.

All LLM calls go through the Model Broker (Invariant #6).  The port
abstracts away the provider (Vertex AI, Google AI Studio, stub) and
routes by model tier (HIGH, CHEAP) per ADR-38.

Returns ModelBrokerResponse with full telemetry (tokens, cost, latency,
model_used) so agents can forward to both sinks (ADR-40).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBrokerResponse:
    """Response from the Model Broker including telemetry.

    Agents MUST use this to extract telemetry for both sinks:
      Sink 1: decision_audit.log_token_usage()  → Pub/Sub → PostgreSQL
      Sink 2: tracing.trace_llm_call()          → Langfuse SDK → Langfuse
    """

    content: str
    model_used: str = "stub"
    model_tier: str = "HIGH"
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ModelBrokerPort(ABC):
    """Port for LLM inference via the Model Broker (L-09)."""

    @abstractmethod
    async def generate(self, *, prompt: str, model_tier: str) -> ModelBrokerResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: The fully rendered prompt string.
            model_tier: Tier hint for model routing (e.g. "HIGH", "CHEAP").

        Returns:
            ModelBrokerResponse with content + full telemetry.
        """
