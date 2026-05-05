"""PII detection via compiled regex patterns.

Inlined from af_shared.guardrails.pii.
5 patterns covering Singapore NRIC, credit card, phone, email, and SSN.
Regex-only implementation (no Presidio dependency) for lightweight usage.
"""

from __future__ import annotations

import re

from validator_agent.guardrails.models import GuardrailViolation

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "NRIC",
        re.compile(r"\b[STFGM]\d{7}[A-Z]\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    ),
    (
        "PHONE",
        re.compile(r"\b(\+65[\s-]?)?[689]\d{3}[\s-]?\d{4}\b"),
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
]


def detect_pii(
    text: str,
    entities: list[str] | None = None,
) -> list[GuardrailViolation]:
    """Scan text for PII patterns.

    Args:
        text: The text to scan.
        entities: Optional list of entity types to check (e.g. ["NRIC", "EMAIL"]).
            If None, all entity types are checked.

    Returns a list of GuardrailViolation for each matched PII pattern.
    """
    violations: list[GuardrailViolation] = []
    for pattern_name, pattern in PII_PATTERNS:
        if entities is not None and pattern_name not in entities:
            continue
        match = pattern.search(text)
        if match:
            violations.append(
                GuardrailViolation(
                    category="pii",
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    severity="high",
                )
            )
    return violations
