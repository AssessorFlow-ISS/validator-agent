"""Data models for the GuardrailScanner module.

Inlined from af_shared.guardrails.models.
All guardrail scanning results are expressed as dataclasses for lightweight,
zero-dependency usage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardrailViolation:
    """A single guardrail violation detected during scanning.

    Attributes:
        category: The violation category. One of:
            "prompt_injection", "jailbreak", "pii", "toxicity", "canary_leak"
        pattern_name: The specific pattern that triggered, e.g. "IGNORE_INSTRUCTIONS", "NRIC".
        matched_text: The text that triggered the violation (redacted if PII).
        severity: Severity level: "high", "medium", or "low".
    """
    category: str
    pattern_name: str
    matched_text: str
    severity: str


@dataclass
class GuardrailResult:
    """Result of a guardrail scan (input or output).

    Attributes:
        passed: True if no blocking violations were found.
        violations: List of all detected violations (may be non-empty even if passed=True
            when the violation category is configured as non-blocking).
        scan_type: Either "input" (pre-LLM) or "output" (post-LLM).
        latency_ms: Time taken for the scan in milliseconds.
    """
    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    scan_type: str = "input"
    latency_ms: float = 0.0
