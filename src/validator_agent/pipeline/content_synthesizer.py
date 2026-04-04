"""Synthesizer: Consolidates findings from 4 analyzers.

Resolves disagreements using voting logic:
- 3/4 or 4/4 agree → CONFIRMED
- 2/4 agree → LIKELY REAL
- 1/4 flags → Synthesizer verifies against original text
"""

from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field

from validator_agent.pipeline.content_analyzers import AllAnalyzerResults
from validator_agent.pipeline.llm_client import generate_structured
from validator_agent.pipeline.models import Finding, PageOcrResult

SYNTHESIZER_SYSTEM = """You are a content safety synthesizer. You receive findings from 4 independent analyzers who reviewed the same document, plus the original document text.

The 4 analyzers are:
- Analyzer A: Child Safety Expert (hard gate — harmful content)
- Analyzer B: Content Quality Expert (soft gate — misinformation)
- Analyzer C: Legal Compliance Expert (soft gate — PII, copyright)
- Analyzer D: Singapore Compliance Expert (hard gate — religious/political sensitivity)

Your job:
1. Consolidate all findings — merge duplicates (same page + same issue across analyzers)
2. For each unique finding, count how many analyzers flagged it:
   - 3/4 or 4/4 agree → confidence: "confirmed"
   - 2/4 agree → confidence: "likely"
   - 1/4 flagged → CHECK the original text. If the finding is verifiable in the text, set confidence: "verified". If not verifiable, DISCARD it (likely hallucination)
3. For NAME findings specifically — verify the context:
   - Author names, public figures, academically cited experts → DISCARD (not PII)
   - Names linked to grades, NRIC, medical/financial records → KEEP (private context PII)
   - If an analyzer flagged an author name as PII, discard it with reason "Author attribution — public context"
4. RECLASSIFY mistyped findings — analyzers may categorize findings under the wrong type:
   - Religious bias ("X is the best religion") flagged as "misinformation" by Analyzer B → reclassify to "religious_political" (hard gate)
   - Political advocacy flagged as "misinformation" by Analyzer B → reclassify to "religious_political" (hard gate)
   - A finding's TYPE determines which gate it triggers (hard gate vs soft gate), so correct classification is critical
5. For PII findings, include the "original" text exactly as it appears (needed for redaction)
6. For PII findings, suggest "redacted_to" replacement:
   - Names → [REDACTED_NAME]
   - NRIC → [REDACTED_NRIC]
   - Phone → [REDACTED_PHONE]
   - Email → [REDACTED_EMAIL]
   - Address → [REDACTED_ADDRESS]
   - Financial → [REDACTED_FINANCIAL]"""


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
{original_text[:15000]}

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
