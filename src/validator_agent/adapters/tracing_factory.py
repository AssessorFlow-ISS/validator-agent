"""Config-driven tracing adapter factory.

Returns stub or Langfuse adapter based on TRACING_ADAPTER env var.
"""

from __future__ import annotations

import os

from validator_agent.ports.tracing_port import TracingPort


def get_tracing() -> TracingPort:
    """Return a TracingPort based on TRACING_ADAPTER env var.

    Options:
        - "stub" (default): StubTracingAdapter with structlog output
        - "langfuse": LangfuseTracingAdapter using Langfuse SDK v4
    """
    adapter = os.getenv("TRACING_ADAPTER", "stub")
    if adapter == "stub":
        from validator_agent.adapters.tracing_stub import StubTracingAdapter

        return StubTracingAdapter()
    if adapter == "langfuse":
        from validator_agent.adapters.tracing_langfuse import LangfuseTracingAdapter

        return LangfuseTracingAdapter()
    raise ValueError(f"Unknown TRACING_ADAPTER: {adapter}")
