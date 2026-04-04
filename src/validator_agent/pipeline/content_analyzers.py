"""Stage 2: Three parallel LLM analyzers with different expert perspectives.

Each analyzer receives ALL pages and returns findings with exact page numbers.
They run in parallel (concurrent) for speed.

Prompt templates are loaded from ``prompts/`` YAML files (ADR-39).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from validator_agent.pipeline.llm_client import generate_structured
from validator_agent.pipeline.models import PageOcrResult
from validator_agent.pipeline.prompt_loader import load_prompt

# ── Load prompt templates (ADR-39) ────

_ANALYZER_A_PROMPT, _ANALYZER_A_META = load_prompt("analyzer_child_safety")
_ANALYZER_B_PROMPT, _ANALYZER_B_META = load_prompt("analyzer_content_quality")
_ANALYZER_C_PROMPT, _ANALYZER_C_META = load_prompt("analyzer_legal_compliance")
_ANALYZER_D_PROMPT, _ANALYZER_D_META = load_prompt("analyzer_singapore_compliance")

# Public aliases for backward compatibility (tests may reference these)
ANALYZER_A_SYSTEM = _ANALYZER_A_PROMPT
ANALYZER_B_SYSTEM = _ANALYZER_B_PROMPT
ANALYZER_C_SYSTEM = _ANALYZER_C_PROMPT
ANALYZER_D_SYSTEM = _ANALYZER_D_PROMPT


class AnalyzerFinding(BaseModel):
    page: int
    type: str
    detail: str
    original: str | None = None


class AnalyzerResponse(BaseModel):
    findings: list[AnalyzerFinding] = Field(default_factory=list)


class AnalyzerResult(BaseModel):
    findings: list[dict] = Field(default_factory=list)
    error: str | None = None


def _format_pages_for_prompt(pages: list[PageOcrResult]) -> str:
    """Format pages with page numbers for the analyzer prompt."""
    parts = []
    for page in pages:
        parts.append(f"=== PAGE {page.page_number} ===\n{page.extracted_text}")
    return "\n\n".join(parts)


def _run_analyzer(system_prompt: str, formatted_pages: str, analyzer_name: str) -> AnalyzerResult:
    """Run a single analyzer via Model Broker and return parsed findings."""
    try:
        parsed, _resp = generate_structured(
            task_key=f"validator.content_analysis.{analyzer_name}",
            system_prompt=system_prompt,
            user_prompt=formatted_pages,
            response_model=AnalyzerResponse,
            temperature=0,
            prompt_version=f"validator/{analyzer_name}@v1",
        )

        findings = [f.model_dump() for f in parsed.findings]
        for f in findings:
            f["source"] = analyzer_name

        return AnalyzerResult(findings=findings)
    except Exception as e:
        return AnalyzerResult(error=f"{analyzer_name}: {e}")


class AllAnalyzerResults(BaseModel):
    analyzer_a: AnalyzerResult = Field(default_factory=AnalyzerResult)  # Child Safety (hard gate)
    analyzer_b: AnalyzerResult = Field(default_factory=AnalyzerResult)  # Content Quality (soft gate)
    analyzer_c: AnalyzerResult = Field(default_factory=AnalyzerResult)  # Legal/PII/Copyright (soft gate)
    analyzer_d: AnalyzerResult = Field(default_factory=AnalyzerResult)  # Religious/Political (hard gate)
    errors: list[str] = Field(default_factory=list)


def run_all_analyzers(pages: list[PageOcrResult]) -> AllAnalyzerResults:
    """Run all 4 analyzers in parallel."""
    formatted_pages = _format_pages_for_prompt(pages)

    analyzers = [
        (_ANALYZER_A_PROMPT, formatted_pages, "analyzer_a"),
        (_ANALYZER_B_PROMPT, formatted_pages, "analyzer_b"),
        (_ANALYZER_C_PROMPT, formatted_pages, "analyzer_c"),
        (_ANALYZER_D_PROMPT, formatted_pages, "analyzer_d"),
    ]

    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_analyzer, sys, pages_text, name): name
            for sys, pages_text, name in analyzers
        }
        for future in futures:
            name = futures[future]
            result = future.result()
            results[name] = result
            if result.error:
                errors.append(result.error)

    return AllAnalyzerResults(
        analyzer_a=results.get("analyzer_a", AnalyzerResult()),
        analyzer_b=results.get("analyzer_b", AnalyzerResult()),
        analyzer_c=results.get("analyzer_c", AnalyzerResult()),
        analyzer_d=results.get("analyzer_d", AnalyzerResult()),
        errors=errors,
    )
