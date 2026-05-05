"""Stage 1: OpenAI Moderation API pre-filter.

Free, fast (~100ms), deterministic. Catches obviously harmful content
before spending on expensive LLM analyzers.

Calls OpenAI Moderation API directly (not through Model Broker).
Moderation API is free and not an LLM call — no reason to proxy it.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from validator_agent.pipeline.model_registry import get_moderation_client

logger = logging.getLogger(__name__)


class ModerationCheckResult(BaseModel):
    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    error: str | None = None


def check_moderation(text: str) -> ModerationCheckResult:
    """Run OpenAI Moderation API on text directly."""
    try:
        truncated = text[:32000]
        client = get_moderation_client()
        response = client.moderations.create(
            input=truncated,
            model="omni-moderation-latest",
        )

        result = response.results[0]
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items() if flagged
        ]

        check_result = ModerationCheckResult(
            flagged=result.flagged,
            categories=flagged_categories,
        )

        # Langfuse trace (fire-and-forget)
        from validator_agent.pipeline.llm_client import trace_tool
        trace_tool(
            tool_name="openai-moderation-prefilter",
            input_params={"text_length": len(truncated)},
            output_summary={"flagged": result.flagged, "categories": flagged_categories},
            latency_ms=0,
        )

        return check_result
    except Exception as e:
        logger.warning("moderation_prefilter_failed: %s", e)
        return ModerationCheckResult(error=f"Moderation API error: {e}")
