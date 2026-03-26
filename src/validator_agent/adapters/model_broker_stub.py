"""StubModelBrokerAdapter — keyword-based content safety responses.

Returns canned JSON safety responses based on keyword matching in the
document content portion of the prompt (not the template instructions).
This avoids needing a real LLM for unit and integration tests.

Keyword detection priority (first match wins):
  1. Harmful: "extremist", "manifesto", "bomb", "weapon"
  2. PII: "SSN:", "social security", email address patterns
  3. Copyright: "ISBN:", "All rights reserved", "Copyright (c)"
  4. Default: safe (is_safe=True)
"""
from __future__ import annotations

import json
import re

from validator_agent.ports.model_broker_port import ModelBrokerPort, ModelBrokerResponse

_HARMFUL_KEYWORDS = ("extremist", "manifesto", "bomb", "weapon")
_PII_PATTERNS = (r"SSN:\s*\d", r"social security", r"[\w.+-]+@[\w-]+\.[\w.]+")
_COPYRIGHT_KEYWORDS = ("ISBN:", "All rights reserved", "Copyright ©", "Copyright (c)")

# Marker in the content_safety.yaml prompt that separates instructions from document text
_DOCUMENT_TEXT_MARKER = "Document text:"
_FILE_NAME_MARKER = "File name:"


def _extract_document_content(prompt: str) -> str:
    """Extract just the document content from the rendered prompt.

    The content_safety.yaml prompt has "Document text:" followed by the actual
    document, then "File name:" and the rest.  We only want to scan the
    document text section for safety keywords, not the template instructions
    (which themselves contain example keywords like email patterns).
    """
    start = prompt.find(_DOCUMENT_TEXT_MARKER)
    if start == -1:
        # Fallback: if no marker found, scan entire prompt
        return prompt

    content_start = start + len(_DOCUMENT_TEXT_MARKER)

    end = prompt.find(_FILE_NAME_MARKER, content_start)
    if end == -1:
        # No file name marker — take everything after document text marker
        return prompt[content_start:]

    return prompt[content_start:end]


class StubModelBrokerAdapter(ModelBrokerPort):
    """Stub adapter that returns content safety responses via keyword matching.

    No LLM call is made.  The stub inspects only the document content
    section of the prompt and returns a JSON response mimicking the
    content_safety.yaml prompt format.
    """

    async def generate(self, *, prompt: str, model_tier: str) -> ModelBrokerResponse:
        """Return a canned JSON safety response based on document content keywords."""
        document_content = _extract_document_content(prompt)
        content_lower = document_content.lower()

        content: str

        # Priority 1: Harmful content
        for keyword in _HARMFUL_KEYWORDS:
            if keyword.lower() in content_lower:
                content = json.dumps({
                    "is_safe": False,
                    "reason_code": "HARMFUL_CONTENT",
                    "message": f"Harmful content detected: contains '{keyword}' reference",
                })
                return self._wrap(content, model_tier)

        # Priority 2: PII
        for pattern in _PII_PATTERNS:
            if re.search(pattern, document_content, re.IGNORECASE):
                content = json.dumps({
                    "is_safe": False,
                    "reason_code": "PII_DETECTED",
                    "message": "Personally identifiable information detected in document",
                })
                return self._wrap(content, model_tier)

        # Priority 3: Copyright
        for keyword in _COPYRIGHT_KEYWORDS:
            if keyword.lower() in content_lower:
                content = json.dumps({
                    "is_safe": False,
                    "reason_code": "COPYRIGHT_VIOLATION",
                    "message": f"Copyright violation detected: contains '{keyword}'",
                })
                return self._wrap(content, model_tier)

        # Default: safe
        content = json.dumps({
            "is_safe": True,
            "reason_code": "VALIDATION_PASSED",
            "message": "No safety concerns detected in document content",
        })
        return self._wrap(content, model_tier)

    @staticmethod
    def _wrap(content: str, model_tier: str) -> ModelBrokerResponse:
        """Wrap content in a ModelBrokerResponse with stub telemetry."""
        return ModelBrokerResponse(
            content=content,
            model_used="gemini-2.0-flash-stub",
            model_tier=model_tier,
            tokens_input=len(content.split()) * 2,
            tokens_output=len(content.split()),
            cost_usd=0.001,
            latency_ms=50.0,
        )
