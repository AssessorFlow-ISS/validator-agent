"""Domain models for decision audit logging.

Inlined from af_shared.models.domain — only the models the Validator Agent uses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DecisionLogEntry(BaseModel):
    """An entry for the agent_decision_log (Decision Audit Service L-11)."""

    workflow_id: str
    agent_name: str
    decision_type: str
    assessor_id: str | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    reasoning_steps: list[dict[str, Any]] | None = None
    confidence_score: float | None = None
    prompt_version: str | None = None
    model_id: str | None = None
    grounding_sources: list[str] | None = None
