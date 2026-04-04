"""Stage 1: OpenAI Moderation API pre-filter.

Free, fast (~100ms), deterministic. Catches obviously harmful content
before spending on 3 GPT-4o analyzers. Saves ~$0.20 per rejected document.
"""

from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel, Field

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class ModerationCheckResult(BaseModel):
    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    error: str | None = None


def check_moderation(text: str) -> ModerationCheckResult:
    """Run OpenAI Moderation API on text."""
    if not OPENAI_API_KEY:
        return ModerationCheckResult(error="No OPENAI_API_KEY set — skipping pre-filter")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        truncated = text[:32000]

        response = client.moderations.create(
            model="omni-moderation-latest",
            input=truncated,
        )

        result = response.results[0]
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items() if flagged
        ]

        return ModerationCheckResult(
            flagged=result.flagged,
            categories=flagged_categories,
        )
    except Exception as e:
        return ModerationCheckResult(error=f"Moderation API error: {e}")
