"""StubTracingAdapter — in-memory tracing for local dev and testing."""
from __future__ import annotations

from typing import Any

import structlog

from validator_agent.ports.model_broker_port import ModelBrokerResponse
from validator_agent.ports.tracing_port import TracingPort

logger = structlog.get_logger(__name__)


class StubTracingAdapter(TracingPort):
    """Logs trace events to structlog. Stores entries for test assertions."""

    def __init__(self) -> None:
        self.llm_calls: list[dict] = []
        self.tool_calls: list[dict] = []
        self.decisions: list[dict] = []

    async def trace_llm_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        task_key: str,
        prompt_version: str,
        model_response: ModelBrokerResponse,
    ) -> None:
        record = {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "task_key": task_key,
            "prompt_version": prompt_version,
            "model_used": model_response.model_used,
            "model_tier": model_response.model_tier,
            "tokens_input": model_response.tokens_input,
            "tokens_output": model_response.tokens_output,
            "cost_usd": model_response.cost_usd,
            "latency_ms": model_response.latency_ms,
        }
        self.llm_calls.append(record)
        logger.info("stub_trace_llm_call", **record)

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
        logger.info("stub_trace_tool_call", workflow_id=workflow_id, tool_name=tool_name, latency_ms=latency_ms)

    async def trace_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "decision_type": decision_type,
            "payload": payload,
        }
        self.decisions.append(record)
        logger.info("stub_trace_decision", workflow_id=workflow_id, agent_name=agent_name, decision_type=decision_type)
