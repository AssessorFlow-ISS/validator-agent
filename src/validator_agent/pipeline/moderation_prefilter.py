"""Stage 1: OpenAI Moderation API pre-filter (via Model Broker).

Free, fast (~100ms), deterministic. Catches obviously harmful content
before spending on 3 GPT-4o analyzers. Saves ~$0.20 per rejected document.

Routes through Model Broker POST /api/v1/moderate so that all external
API calls are centralized (ADR-42 invariant #6).
"""

from __future__ import annotations

import logging
import os

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_BROKER_URL = os.getenv("MODEL_BROKER_URL", "http://localhost:8010")


class ModerationCheckResult(BaseModel):
    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    error: str | None = None


def check_moderation(text: str) -> ModerationCheckResult:
    """Run OpenAI Moderation API on text via Model Broker."""
    try:
        truncated = text[:32000]

        response = httpx.post(
            f"{MODEL_BROKER_URL}/api/v1/moderate",
            json={
                "input": truncated,
                "model": "omni-moderation-latest",
                "session_id": os.getenv("CURRENT_WORKFLOW_ID", "no-session"),
                "agent_id": "validator-agent",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        return ModerationCheckResult(
            flagged=data["flagged"],
            categories=data.get("categories", []),
        )
    except Exception as e:
        logger.warning("moderation_prefilter_failed: %s", e)
        return ModerationCheckResult(error=f"Moderation API error: {e}")
