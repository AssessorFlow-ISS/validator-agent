"""Content Safety Reasoner — LLM-based safety check for document content.

Uses the Model Broker port to send extracted text through a content safety
prompt and parse the structured JSON response. The prompt template is loaded
from prompts/content_safety.yaml (ADR-39).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from validator_agent.domain.terminal_signal import ReasonCode
from validator_agent.ports.model_broker_port import ModelBrokerPort

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


@dataclass(frozen=True)
class LlmCallMetrics:
    """Telemetry captured from a Model Broker LLM call."""

    model_used: str
    model_tier: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class SafetyResult:
    """Result of the content safety check."""

    is_safe: bool
    reason_code: ReasonCode
    message: str
    llm_metrics: LlmCallMetrics | None = None


def _load_prompt_template(name: str) -> tuple[str, dict]:
    """Load a YAML prompt template, returning (template_body, frontmatter)."""
    path = _PROMPT_DIR / f"{name}.yaml"
    raw = path.read_text(encoding="utf-8")

    # Split YAML frontmatter from body
    parts = raw.split("---", 2)
    if len(parts) >= 3:
        frontmatter = yaml.safe_load(parts[1])
        body = parts[2].strip()
    else:
        frontmatter = {}
        body = raw.strip()

    return body, frontmatter


class ContentSafetyReasoner:
    """Checks document text for harmful content, PII, and copyright violations.

    In production this sends the text to the Model Broker for LLM-based
    reasoning.  In tests the StubModelBrokerAdapter returns canned responses.
    """

    def __init__(self, model_broker: ModelBrokerPort) -> None:
        self._model_broker = model_broker
        self._template, self._frontmatter = _load_prompt_template("content_safety")

    @property
    def prompt_version(self) -> str:
        version = self._frontmatter.get("version", "1")
        return f"validator/content_safety@v{version}"

    async def check(self, content: str, file_name: str) -> SafetyResult:
        """Run content safety reasoning on extracted text.

        Returns a SafetyResult indicating whether the content is safe and,
        if not, the reason code and human-readable message.
        """
        if not content or not content.strip():
            return SafetyResult(
                is_safe=True,
                reason_code=ReasonCode.VALIDATION_PASSED,
                message="No content to evaluate",
            )

        # Use regex substitution instead of str.format() because the prompt
        # template contains literal JSON curly braces that conflict with
        # Python's format string syntax.
        prompt = self._template
        prompt = re.sub(r"\{content\}", content, prompt)
        prompt = re.sub(r"\{file_name\}", file_name, prompt)
        model_tier = self._frontmatter.get("model_tier", "HIGH")

        model_response = await self._model_broker.generate(
            prompt=prompt,
            model_tier=model_tier,
        )

        result = self._parse_response(model_response.content, file_name)

        # Attach LLM telemetry so the caller can forward to both sinks
        metrics = LlmCallMetrics(
            model_used=model_response.model_used,
            model_tier=model_response.model_tier,
            tokens_input=model_response.tokens_input,
            tokens_output=model_response.tokens_output,
            cost_usd=model_response.cost_usd,
            latency_ms=model_response.latency_ms,
        )
        return SafetyResult(
            is_safe=result.is_safe,
            reason_code=result.reason_code,
            message=result.message,
            llm_metrics=metrics,
        )

    @staticmethod
    def _parse_response(raw: str, file_name: str) -> SafetyResult:
        """Parse LLM JSON response into a SafetyResult."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM safety response as JSON, treating as safe",
                extra={"raw_response": raw, "file_name": file_name},
            )
            return SafetyResult(
                is_safe=True,
                reason_code=ReasonCode.VALIDATION_PASSED,
                message=f"Content safety check returned unparseable response for {file_name}, defaulting to safe",
            )

        is_safe = data.get("is_safe", True)
        reason_code_str = data.get("reason_code", "VALIDATION_PASSED")
        message = data.get("message", "")

        try:
            reason_code = ReasonCode(reason_code_str)
        except ValueError:
            reason_code = (
                ReasonCode.VALIDATION_PASSED
                if is_safe
                else ReasonCode.CONTENT_POLICY_VIOLATION
            )

        return SafetyResult(
            is_safe=is_safe,
            reason_code=reason_code,
            message=message or f"Content safety check for {file_name}",
        )
