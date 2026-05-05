"""Canary token injection and detection for system prompt leak prevention.

Inlined from af_shared.guardrails.canary.
A canary token is a random marker injected into the system prompt.
If the LLM leaks the system prompt in its output, the canary token
will be present and detectable — indicating a system prompt leak (OWASP LLM07).
"""

from __future__ import annotations

import secrets

from validator_agent.guardrails.models import GuardrailViolation


def generate_canary() -> str:
    """Generate a random canary token. Format: CANARY_{16 hex characters}."""
    return f"CANARY_{secrets.token_hex(8)}"


def inject_canary(system_prompt: str, canary: str) -> str:
    """Inject canary into system prompt as invisible marker."""
    return f"{system_prompt}\n\n[INTERNAL_MARKER:{canary}]"


def detect_canary_leak(output: str, canary: str) -> GuardrailViolation | None:
    """Check if canary token leaked into LLM output.

    Returns a GuardrailViolation if the canary is found, None otherwise.
    """
    if canary in output:
        return GuardrailViolation(
            category="canary_leak",
            pattern_name="CANARY_TOKEN",
            matched_text=canary,
            severity="high",
        )
    return None
