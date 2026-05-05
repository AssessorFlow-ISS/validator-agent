"""Main GuardrailScanner orchestrator.

Inlined from af_shared.guardrails.scanner.
Centralized guardrail scanner wrapping every LLM call in the Validator Agent.
Rule-based with zero additional LLM token cost.
Config-driven: GUARDRAILS_ENABLED, GUARDRAILS_BLOCK_ON_INJECTION, etc.

This is the Validator Agent's local implementation of Model Broker's L-10
Guardrails Service — same patterns, same security posture, no proxy dependency.
"""

from __future__ import annotations

import time

from validator_agent.guardrails.canary import detect_canary_leak, generate_canary, inject_canary
from validator_agent.guardrails.models import GuardrailResult, GuardrailViolation
from validator_agent.guardrails.normalization import normalize_for_evasion
from validator_agent.guardrails.pii import detect_pii
from validator_agent.guardrails.prompt_injection import detect_injection


class GuardrailScanner:
    """Rule-based guardrail scanner for LLM call input/output protection.

    Wraps every LLM call in the Validator Agent pipeline:
      PRE-LLM:  scan_input() checks for injection, PII
      SYSTEM:   prepare_system_prompt() injects canary token
      POST-LLM: scan_output() checks for canary leak

    Config-driven via constructor parameters (mapped from env vars by caller).

    Args:
        enabled: Master switch. When False, all scans return passed=True.
        block_on_pii: When True, PII violations cause scan to fail.
            Default False — educational content legitimately contains names/emails.
        block_on_injection: When True, injection violations cause scan to fail.
            Default False for document text (may contain "ignore previous value"
            in coding content). System prompt injection is always blocked.
        canary_enabled: When True, prepare_system_prompt injects a canary token.
        pii_entities: Optional list of PII entity types to scan for.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        block_on_pii: bool = False,
        block_on_injection: bool = False,
        canary_enabled: bool = True,
        pii_entities: list[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._block_on_pii = block_on_pii
        self._block_on_injection = block_on_injection
        self._canary_enabled = canary_enabled
        self._pii_entities = pii_entities

    def scan_input(self, prompt: str) -> GuardrailResult:
        """Pre-LLM: scan prompt for injection, jailbreak, PII.

        Runs two passes: raw text + evasion-normalized text.
        Returns GuardrailResult with pass/fail and detected violations.
        """
        start = time.monotonic()

        if not self._enabled:
            return GuardrailResult(
                passed=True,
                violations=[],
                scan_type="input",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        all_violations: list[GuardrailViolation] = []
        blocking = False

        # Pass 1: scan raw text for injection
        injection_violations = detect_injection(prompt)
        if injection_violations:
            all_violations.extend(injection_violations)
            if self._block_on_injection:
                blocking = True

        # Pass 2: scan evasion-normalized text
        normalized = normalize_for_evasion(prompt)
        if normalized != prompt:
            norm_violations = detect_injection(normalized)
            existing_names = {v.pattern_name for v in all_violations}
            for v in norm_violations:
                if v.pattern_name not in existing_names:
                    all_violations.append(
                        GuardrailViolation(
                            category=v.category,
                            pattern_name=v.pattern_name,
                            matched_text=v.matched_text,
                            severity=v.severity,
                        )
                    )
                    if self._block_on_injection:
                        blocking = True

        # Scan for PII (raw + normalized)
        pii_violations = detect_pii(prompt, entities=self._pii_entities)
        if pii_violations:
            all_violations.extend(pii_violations)
            if self._block_on_pii:
                blocking = True

        if normalized != prompt:
            norm_pii = detect_pii(normalized, entities=self._pii_entities)
            existing_pii = {v.pattern_name for v in all_violations if v.category == "pii"}
            for v in norm_pii:
                if v.pattern_name not in existing_pii:
                    all_violations.append(v)
                    if self._block_on_pii:
                        blocking = True

        return GuardrailResult(
            passed=not blocking,
            violations=all_violations,
            scan_type="input",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def scan_output(self, output: str, *, canary: str | None = None) -> GuardrailResult:
        """Post-LLM: scan output for canary leak.

        PII detection in output is handled by Guardrails AI (NER-based,
        more comprehensive than regex). This method focuses on canary only.
        """
        start = time.monotonic()

        if not self._enabled:
            return GuardrailResult(
                passed=True,
                violations=[],
                scan_type="output",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        all_violations: list[GuardrailViolation] = []
        blocking = False

        # Canary leak check — always blocking
        if canary is not None:
            canary_violation = detect_canary_leak(output, canary)
            if canary_violation is not None:
                all_violations.append(canary_violation)
                blocking = True

        return GuardrailResult(
            passed=not blocking,
            violations=all_violations,
            scan_type="output",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def prepare_system_prompt(self, system_prompt: str) -> tuple[str, str | None]:
        """Inject canary token into system prompt if enabled.

        Returns (augmented_prompt, canary_token).
        If canary is disabled, returns (original_prompt, None).
        """
        if not self._canary_enabled or not self._enabled:
            return system_prompt, None

        canary = generate_canary()
        augmented = inject_canary(system_prompt, canary)
        return augmented, canary
