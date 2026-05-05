"""Prompt injection and jailbreak detection via compiled regex patterns.

Inlined from af_shared.guardrails.prompt_injection.
11 patterns: 9 ported from the TypeScript security.ts reference implementation
plus CANARY_PROBE and DATA_ENUMERATION.
All patterns are compiled at module load time for performance.
Zero LLM token cost — pure regex matching.
"""

from __future__ import annotations

import re

from validator_agent.guardrails.models import GuardrailViolation

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "IGNORE_INSTRUCTIONS",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions|prompts|rules|context)",
            re.IGNORECASE,
        ),
    ),
    (
        "ROLE_OVERRIDE",
        re.compile(
            r"(you\s+are\s+now\s+|act\s+as\s+(a|an|if)\s+|"
            r"pretend\s+(to\s+be|you\s+are))",
            re.IGNORECASE,
        ),
    ),
    (
        "SYSTEM_PROMPT_LEAK",
        re.compile(
            r"(repeat|show|reveal|display|print|output)\s+(me\s+)?(your\s+)?"
            r"(system\s+prompt|instructions|rules|configuration)",
            re.IGNORECASE,
        ),
    ),
    (
        "TEST_MODE",
        re.compile(
            r"(test|debug|developer|admin|maintenance)\s+mode",
            re.IGNORECASE,
        ),
    ),
    (
        "DISREGARD",
        re.compile(
            r"disregard\s+(your|all|the|any)\s+"
            r"(previous|prior|above|rules|instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "JAILBREAK",
        re.compile(
            r"\b(DAN|do\s+anything\s+now|jailbreak|"
            r"bypass\s+(safety|filter|guard)|escape\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ENCODED_INSTRUCTIONS",
        re.compile(
            r"(base64\s*:|atob\s*\(|\\x[0-9a-f]{2}\\x|&#x?[0-9a-f]+;&#)",
            re.IGNORECASE,
        ),
    ),
    (
        "OUTPUT_FORMAT_OVERRIDE",
        re.compile(
            r"(always\s+)?(output|return|respond\s+with|classify\s+as|"
            r"set\s+score\s+to)\s+(only|just|exactly|always)\s+",
            re.IGNORECASE,
        ),
    ),
    (
        "INSTRUCTION_DELIMITER",
        re.compile(
            r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|system\|>|"
            r"###\s*(System|Human|Assistant)\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "CANARY_PROBE",
        re.compile(
            r"(INTERNAL_MARKER|CANARY_[a-f0-9]|"
            r"(output|show|reveal|print|return)\s+.{0,50}CANARY)",
            re.IGNORECASE,
        ),
    ),
    (
        "DATA_ENUMERATION",
        re.compile(
            r"(SELECT\s+\*?\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM|"
            r"document_chunks|enriched_chunks|policy_chunks|assessment_configs|"
            r"agent_decision_log|token_usage_ledger|workflow_events|"
            r"assessor_id\s*=|WHERE\s+.+_id\s*=)",
            re.IGNORECASE,
        ),
    ),
]


def detect_injection(text: str) -> list[GuardrailViolation]:
    """Scan text for prompt injection and jailbreak patterns.

    Returns a list of GuardrailViolation for each matched pattern.
    Empty list if no patterns match.
    """
    violations: list[GuardrailViolation] = []
    for pattern_name, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                GuardrailViolation(
                    category="prompt_injection",
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    severity="high",
                )
            )
    return violations
