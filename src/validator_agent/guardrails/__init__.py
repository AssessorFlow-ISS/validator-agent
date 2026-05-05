"""Guardrails module — rule-based input/output scanning for LLM calls.

Inlined from af_shared.guardrails. Zero external dependencies.
Zero additional LLM cost — pure regex + string matching.

Usage:
    from validator_agent.guardrails import GuardrailScanner
    scanner = GuardrailScanner(enabled=True, block_on_injection=True)
    result = scanner.scan_input(prompt_text)
"""

from validator_agent.guardrails.scanner import GuardrailScanner

__all__ = ["GuardrailScanner"]
