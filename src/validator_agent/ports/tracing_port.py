"""Port interface for LLM observability tracing (ADR-08, ADR-40).

Dual-sink architecture:
    Sink 1: DecisionAuditPort -> PostgreSQL (compliance, app-level reads)
    Sink 2: TracingPort       -> Langfuse   (operational visibility)

Adapters:
    - StubTracingAdapter: logs to structlog (local dev, tests)
    - LangfuseTracingAdapter: Langfuse SDK (production)
Config: TRACING_ADAPTER=stub|langfuse (default: stub)

Inlined from af_shared.ports.tracing — no external dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from validator_agent.domain.audit_models import DecisionLogEntry


class TracingPort(ABC):
    """Abstract port for LLM observability tracing.

    All methods are fire-and-forget. Tracing failures must never
    break agent workflows.
    """

    @abstractmethod
    async def trace_decision(self, entry: DecisionLogEntry) -> None:
        """Send decision data to Langfuse as an agent observation."""

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
        """Send a tool call to Langfuse as a tool observation."""
