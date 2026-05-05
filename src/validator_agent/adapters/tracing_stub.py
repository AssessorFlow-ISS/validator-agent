"""Stub adapter for LLM observability tracing.

Logs to structlog for local development and testing.
Stores entries in-memory for test assertions.

Inlined from af_shared.adapters.stubs.tracing_stub.
"""

from __future__ import annotations

from typing import Any

import structlog

from validator_agent.domain.audit_models import DecisionLogEntry
from validator_agent.ports.tracing_port import TracingPort

logger = structlog.get_logger(__name__)


class StubTracingAdapter(TracingPort):
    """In-memory stub for Langfuse tracing."""

    def __init__(self) -> None:
        self.decisions: list[DecisionLogEntry] = []
        self.tool_calls: list[dict] = []

    async def trace_decision(self, entry: DecisionLogEntry) -> None:
        self.decisions.append(entry)
        logger.info("stub_trace_decision", workflow_id=entry.workflow_id, agent_name=entry.agent_name)

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
        record = {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "input_params": input_params,
            "output_summary": output_summary,
            "latency_ms": latency_ms,
        }
        self.tool_calls.append(record)
        logger.info("stub_trace_tool_call", workflow_id=workflow_id, tool_name=tool_name)

    def clear(self) -> None:
        self.decisions.clear()
        self.tool_calls.clear()
