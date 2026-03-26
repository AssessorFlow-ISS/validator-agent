"""TracingPort — abstract interface for Langfuse observability tracing.

Dual-sink architecture (ADR-40):
  Sink 1: DecisionAuditPort → Pub/Sub → PostgreSQL (compliance, app-level reads)
  Sink 2: TracingPort → Langfuse SDK → Langfuse (operational visibility)

Owner: Walfa implements the real LangfuseTracingAdapter.
Dale provides the port + StubTracingAdapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from validator_agent.ports.model_broker_port import ModelBrokerResponse


class TracingPort(ABC):
    """Port for Langfuse observability tracing. Fire-and-forget."""

    @abstractmethod
    async def trace_llm_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        task_key: str,
        prompt_version: str,
        model_response: ModelBrokerResponse,
    ) -> None:
        """Trace an LLM call (Langfuse as_type='generation')."""

    @abstractmethod
    async def trace_tool_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        tool_name: str,
        input_params: dict[str, Any],
        output_summary: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Trace a tool call (Langfuse as_type='tool')."""

    @abstractmethod
    async def trace_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Trace an agent decision (Langfuse as_type='agent')."""
