"""Langfuse tracing adapter (v4 SDK).

All methods are fire-and-forget: exceptions are caught and logged.
Tracing failures must never block agent workflows.

Trace-per-workflow: each workflow_id maps to a single Langfuse trace
via MD5 hash. All observations within a workflow are grouped under
that trace for a unified timeline view.

Environment variables:
    LANGFUSE_PUBLIC_KEY  — Langfuse project public key
    LANGFUSE_SECRET_KEY  — Langfuse project secret key
    LANGFUSE_HOST        — Langfuse server URL (default: https://cloud.langfuse.com)

Inlined from af_shared.adapters.real.tracing_langfuse.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from validator_agent.domain.audit_models import DecisionLogEntry
from validator_agent.ports.tracing_port import TracingPort

logger = structlog.get_logger(__name__)


class LangfuseTracingAdapter(TracingPort):
    """Langfuse-backed tracing adapter (v4 SDK)."""

    def __init__(self) -> None:
        self._client: Any = None
        self._traces: dict[str, dict[str, str]] = {}
        try:
            from langfuse import Langfuse
            self._client = Langfuse()
            logger.info("langfuse_client_initialized")
        except ImportError:
            logger.warning("langfuse_sdk_not_installed", hint="pip install langfuse>=4.0")
        except Exception:
            logger.warning("langfuse_client_init_failed", exc_info=True)

    def _get_or_create_trace(self, workflow_id: str) -> dict[str, str]:
        if workflow_id in self._traces:
            return self._traces[workflow_id]
        trace_id = hashlib.md5(workflow_id.encode()).hexdigest()
        trace_ctx: dict[str, str] = {"trace_id": trace_id}
        self._traces[workflow_id] = trace_ctx
        return trace_ctx

    async def trace_decision(self, entry: DecisionLogEntry) -> None:
        try:
            if self._client is None:
                return
            trace_ctx = self._get_or_create_trace(entry.workflow_id)
            self._client.create_event(
                trace_context=trace_ctx,
                name=f"decision/{entry.decision_type}",
                input=entry.input,
                output=entry.output,
                metadata={
                    "agent_name": entry.agent_name, "decision_type": entry.decision_type,
                    "confidence_score": entry.confidence_score, "prompt_version": entry.prompt_version,
                    "model_id": entry.model_id, "reasoning_steps": entry.reasoning_steps,
                    "grounding_sources": entry.grounding_sources, "assessor_id": entry.assessor_id,
                },
            )
        except Exception:
            logger.warning("langfuse_trace_decision_failed", workflow_id=entry.workflow_id, exc_info=True)

    async def trace_tool_call(
        self, *, workflow_id: str, agent_name: str, tool_name: str,
        input_params: dict[str, Any], output_summary: dict[str, Any],
        latency_ms: float,
    ) -> None:
        try:
            if self._client is None:
                return
            trace_ctx = self._get_or_create_trace(workflow_id)
            tool = self._client.start_observation(
                trace_context=trace_ctx, name=f"tool/{tool_name}", as_type="tool",
                input=input_params, output=output_summary,
                metadata={"agent_name": agent_name, "latency_ms": latency_ms},
            )
            tool.end()
        except Exception:
            logger.warning("langfuse_trace_tool_call_failed", workflow_id=workflow_id, exc_info=True)

    def flush(self) -> None:
        try:
            if self._client is not None:
                self._client.flush()
        except Exception:
            logger.warning("langfuse_flush_failed", exc_info=True)
