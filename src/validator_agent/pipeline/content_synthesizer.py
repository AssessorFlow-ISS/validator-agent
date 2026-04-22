"""Synthesizer: Consolidates findings from 4 analyzers.

Resolves disagreements using voting logic:
- 3/4 or 4/4 agree → CONFIRMED
- 2/4 agree → LIKELY REAL
- 1/4 flags → Synthesizer verifies against original text

Prompt template loaded from ``prompts/content_synthesizer.yaml`` (ADR-39).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from validator_agent.pipeline.content_analyzers import AllAnalyzerResults
from validator_agent.pipeline.llm_client import generate_structured
from validator_agent.pipeline.models import Finding, PageOcrResult
from validator_agent.pipeline.prompt_loader import load_prompt

# ── Load prompt template (ADR-39) ────

_SYNTHESIZER_PROMPT, _SYNTHESIZER_META = load_prompt("content_synthesizer")
_MAX_INPUT_CHARS = int(_SYNTHESIZER_META.get("max_input_chars", 15000))

# Public alias for backward compatibility
SYNTHESIZER_SYSTEM = _SYNTHESIZER_PROMPT


class SynthesizedFinding(BaseModel):
    page: int
    type: str
    detail: str
    original: str | None = None
    redacted_to: str | None = None
    confidence: str | None = None


class DiscardedFinding(BaseModel):
    page: int
    type: str
    detail: str
    reason: str


class SynthesizerResponse(BaseModel):
    findings: list[SynthesizedFinding] = Field(default_factory=list)
    discarded: list[DiscardedFinding] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    findings: list[dict] = Field(default_factory=list)
    discarded: list[dict] = Field(default_factory=list)
    error: str | None = None


def synthesize_findings(
    analyzer_results: AllAnalyzerResults,
    pages: list[PageOcrResult],
) -> SynthesisResult:
    """Consolidate findings from 4 analyzers via Model Broker."""
    all_findings = {
        "analyzer_a": analyzer_results.analyzer_a.findings,
        "analyzer_b": analyzer_results.analyzer_b.findings,
        "analyzer_c": analyzer_results.analyzer_c.findings,
        "analyzer_d": analyzer_results.analyzer_d.findings,
    }

    original_text_parts = []
    for page in pages:
        original_text_parts.append(f"=== PAGE {page.page_number} ===\n{page.extracted_text}")
    original_text = "\n\n".join(original_text_parts)

    user_prompt = f"""Analyzer reports:
{json.dumps(all_findings, indent=2)}

Original document text:
{original_text[:_MAX_INPUT_CHARS]}

Consolidate the findings following the instructions."""

    try:
        parsed, _resp = generate_structured(
            task_key="validator.content_synthesizer",
            system_prompt=SYNTHESIZER_SYSTEM,
            user_prompt=user_prompt,
            response_model=SynthesizerResponse,
            temperature=0,
            prompt_version="validator/content_synthesizer@v1",
        )

        return SynthesisResult(
            findings=[f.model_dump() for f in parsed.findings],
            discarded=[d.model_dump() for d in parsed.discarded],
        )
    except Exception as e:
        return _merge_without_synthesis(analyzer_results, error=f"Synthesizer failed: {e}")


def _merge_without_synthesis(
    analyzer_results: AllAnalyzerResults,
    error: str = "Synthesizer unavailable — findings merged without verification",
) -> SynthesisResult:
    """Fallback: merge all findings without LLM synthesis."""
    all_findings = []
    for result in [analyzer_results.analyzer_a, analyzer_results.analyzer_b, analyzer_results.analyzer_c, analyzer_results.analyzer_d]:
        for f in result.findings:
            f_copy = dict(f)
            f_copy["confidence"] = "unverified"
            all_findings.append(f_copy)
    return SynthesisResult(
        findings=all_findings,
        error=error,
    )


def findings_to_models(raw_findings: list[dict]) -> list[Finding]:
    """Convert raw finding dicts to Finding model instances."""
    return [Finding(**f) for f in raw_findings]
