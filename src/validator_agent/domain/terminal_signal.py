"""Terminal Signal model for the Validator Agent.

The Terminal Signal is the structured contract returned by the Validator Agent
for every validation decision. It uses a binary PROCEED/TERMINATE status with
a reason code and human-readable message for audit trail and explainability.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TerminalSignalStatus(StrEnum):
    """Outcome of validation: proceed, proceed with warnings, or terminate."""

    PROCEED = "PROCEED"
    PROCEED_WITH_WARNINGS = "PROCEED_WITH_WARNINGS"
    TERMINATE = "TERMINATE"


class ReasonCode(StrEnum):
    """Structured reason codes for validation outcomes.

    Each code maps to a specific validation check failure (or overall pass).
    Hard gates produce TERMINATE; soft gates produce PROCEED_WITH_WARNINGS.
    """

    # Success
    VALIDATION_PASSED = "VALIDATION_PASSED"

    # Component 1: MRC (hard gate)
    BLURRY_UNREADABLE = "BLURRY_UNREADABLE"

    # Component 2: OCR (hard gate)
    OCR_FAILED = "OCR_FAILED"
    HARMFUL_IMAGE = "HARMFUL_IMAGE"

    # Component 3: Content Safety — hard gates (TERMINATE)
    HARMFUL_CONTENT = "HARMFUL_CONTENT"
    RELIGIOUS_POLITICAL_VIOLATION = "RELIGIOUS_POLITICAL_VIOLATION"
    CONTENT_SAFETY_UNAVAILABLE = "CONTENT_SAFETY_UNAVAILABLE"

    # Component 3: Content Safety — soft gates (PROCEED_WITH_WARNINGS)
    CONTENT_POLICY_VIOLATION = "CONTENT_POLICY_VIOLATION"
    PII_DETECTED = "PII_DETECTED"
    COPYRIGHT_VIOLATION = "COPYRIGHT_VIOLATION"


class TerminalSignal(BaseModel):
    """Structured Terminal Signal returned per validated content item.

    On PROCEED the workflow continues; on TERMINATE the Orchestrator halts
    the workflow and the assessor must create a new assessment.
    """

    status: TerminalSignalStatus
    reason_code: ReasonCode
    message: str = Field(
        ...,
        max_length=500,
        description="Human-readable explanation for the decision",
    )
