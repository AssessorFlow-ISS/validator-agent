"""Stage 2: Three parallel LLM analyzers with different expert perspectives.

Each analyzer receives ALL pages and returns findings with exact page numbers.
They run in parallel (concurrent) for speed.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel, Field

from validator_agent.pipeline.models import PageOcrResult


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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gpt-4o")


def _format_pages_for_prompt(pages: list[PageOcrResult]) -> str:
    """Format pages with page numbers for the analyzer prompt."""
    parts = []
    for page in pages:
        parts.append(f"=== PAGE {page.page_number} ===\n{page.extracted_text}")
    return "\n\n".join(parts)


ANALYZER_A_SYSTEM = """You are a child safety expert reviewing educational material for primary school students in Singapore.

Analyze the following document pages for:
- Explicit violence or gore descriptions (not academic historical references)
- Hate speech targeting any group
- Sexual or explicit content
- Instructions for weapons, drugs, or dangerous activities
- Self-harm encouragement
- Age-inappropriate content for primary school students

Context matters: "World War II caused millions of deaths" is educational and acceptable. "Here's how to build a weapon" is not.

For each finding, return a JSON array. If no issues found, return an empty array.
Each finding must include: page (int), type ("harmful"), detail (string explaining the issue), original (the exact problematic text).

"""

ANALYZER_B_SYSTEM = """You are an educational content quality expert reviewing material for primary school students in Singapore.

Analyze the following document pages for:
- Factual errors or inaccuracies
- Overstated or misleading claims presented as fact
- Outdated information presented as current
- Opinions presented as established facts without attribution
- Misleading framing or cherry-picked data

Be careful: only flag clear misinformation, not legitimate academic debate or simplified explanations appropriate for primary school level.

For each finding, return a JSON array. If no issues found, return an empty array.
Each finding must include: page (int), type ("misinformation"), detail (string explaining the concern and why it may be inaccurate).

"""

ANALYZER_C_SYSTEM = """You are a legal and data protection compliance officer specializing in Singapore's education regulations, PDPA (Personal Data Protection Act), and Maintenance of Religious Harmony Act.

Analyze the following document pages for:

1. PII (Personal Identifiable Information):

   ALWAYS redact (these have zero educational value):
   - NRIC/FIN numbers (format: S/T/F/G + 7 digits + letter, e.g. S1234567A)
   - Phone numbers (Singapore: +65 XXXX XXXX or 8/9XXX XXXX)
   - Email addresses
   - Physical addresses (Singapore format: Block/street/postal)
   - Financial data (bank accounts, credit card numbers)

   Names — CONTEXT MATTERS:
   - DO NOT flag: author names ("written by John Tan"), public figures ("Winston Churchill"),
     academically cited experts ("according to John Kindervag of Forrester")
   - DO flag: names in private/personal context — linked to grades ("Student John Tan scored 95%"),
     linked to NRIC/medical/financial records, or appearing alongside other PII

   The rule: names in public/professional/academic context are fine.
   Names in private/personal context (grades, NRIC, health, financial) are PII.

2. Copyright markers (IMPORTANT — scan EVERY page carefully, even if it is just 1-2 lines on a long page):
   - "All rights reserved" (any case)
   - "Copyright ©" or "©" or "Ⓒ" (any copyright symbol variant)
   - "Do not reproduce" or "reproduction prohibited"
   - "may not be reproduced without permission"
   - Restrictive license statements
   These are often at the top or bottom of a page and easy to miss among long text. You MUST catch them.

For each finding, include:
- page (int)
- type: "name", "nric", "phone", "email", "address", "financial", or "copyright"
- detail (string)
- original (the exact text found — required for PII so we can redact it)

"""

ANALYZER_D_SYSTEM = """You are a Singapore government compliance officer specializing in the Maintenance of Religious Harmony Act and political sensitivity in education.

Analyze the following document pages for religious and political sensitivity:

TERMINATE (hard gate — these must be caught):
- Content promoting or disparaging any specific religion (e.g. "Buddhism is the best religion")
- Religiously biased content that favours one faith over others
- Politically biased or partisan content
- Political propaganda or persuasion
- Content undermining racial or religious harmony in Singapore's multiracial context

ALLOW (not a violation):
- Neutral academic discussion of religion ("Buddhism originated in India in the 5th century BCE")
- Factual political science ("Singapore uses a parliamentary system")
- Historical references to religion or politics in educational context
- Comparative religion studies that treat all faiths equally

The key principle is NEUTRALITY. Factual, academic discussion is fine. Advocacy, promotion, disparagement, or persuasion is not.

This is a HARD GATE in the Validator Agent — findings here will TERMINATE the workflow. Be accurate. Do not flag neutral academic content.

For each finding, include:
- page (int)
- type: "religious_political"
- detail (string explaining exactly what is problematic and why)

"""


def _run_analyzer(system_prompt: str, formatted_pages: str, analyzer_name: str) -> AnalyzerResult:
    """Run a single analyzer and return parsed findings."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.beta.chat.completions.parse(
            model=ANALYZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_pages},
            ],
            response_format=AnalyzerResponse,
            temperature=0,
        )

        parsed = response.choices[0].message.parsed
        findings = [f.model_dump() for f in parsed.findings]

        # Tag each finding with the analyzer source
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
        (ANALYZER_A_SYSTEM, formatted_pages, "analyzer_a"),
        (ANALYZER_B_SYSTEM, formatted_pages, "analyzer_b"),
        (ANALYZER_C_SYSTEM, formatted_pages, "analyzer_c"),
        (ANALYZER_D_SYSTEM, formatted_pages, "analyzer_d"),
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
