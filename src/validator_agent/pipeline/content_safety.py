"""Component 3: Content Safety — main entry point.

Two-stage pipeline:
  Stage 1: OpenAI Moderation API pre-filter (free, fast)
  Stage 2: 4 parallel LLM analyzers + synthesizer

Five checks in two tiers:
  Hard gates: harmful content (TERMINATE), religious/political (TERMINATE)
  Soft gates: PII (REDACT), copyright (WARNING), misinformation (WARNING)
"""

from __future__ import annotations

from validator_agent.pipeline.content_analyzers import run_all_analyzers
from validator_agent.pipeline.content_synthesizer import findings_to_models, synthesize_findings
from validator_agent.pipeline.models import ContentSafetyResult, Finding, PageOcrResult
from validator_agent.pipeline.moderation_prefilter import check_moderation

# Types that map to hard gates (TERMINATE)
HARMFUL_TYPES = {"harmful"}
RELIGIOUS_POLITICAL_TYPES = {"religious_political"}
PII_TYPES = {"name", "nric", "phone", "email", "address", "financial"}

# Types that map to soft gates (WARNING)
COPYRIGHT_TYPES = {"copyright"}
MISINFORMATION_TYPES = {"misinformation"}


def _apply_redactions(pages: list[PageOcrResult], pii_findings: list[Finding]) -> str:
    """Redact PII from pages and rebuild cleaned text.

    Args:
        pages: Original pages from Component 2.
        pii_findings: Confirmed PII findings with 'original' and 'redacted_to'.

    Returns:
        Cleaned text with PII replaced by [REDACTED_*] tokens.
    """
    # Build a map: page_number → list of (original, redacted_to)
    redactions_by_page: dict[int, list[tuple[str, str]]] = {}
    for f in pii_findings:
        if f.original and f.redacted_to:
            redactions_by_page.setdefault(f.page, []).append((f.original, f.redacted_to))

    cleaned_parts = []
    for page in pages:
        text = page.extracted_text
        if page.page_number in redactions_by_page:
            for original, replacement in redactions_by_page[page.page_number]:
                text = text.replace(original, replacement)
        cleaned_parts.append(text)

    return "\n\n".join(cleaned_parts)


def check_content_safety(pages: list[PageOcrResult]) -> ContentSafetyResult:
    """Run content safety checks on document pages.

    Args:
        pages: Per-page text from Component 2 (OcrResult.pages).

    Returns:
        ContentSafetyResult with findings, cleaned_text, and overall status.
    """
    if not pages:
        return ContentSafetyResult(
            overall_status="PROCEED",
            cleaned_text="",
        )

    # ── Stage 1: Moderation API pre-filter ────
    combined_text = "\n\n".join(p.extracted_text for p in pages)
    moderation_result = check_moderation(combined_text)

    if moderation_result.flagged:
        categories = ", ".join(moderation_result.categories)
        return ContentSafetyResult(
            overall_status="TERMINATE",
            termination_reason="HARMFUL_CONTENT",
            termination_detail=f"OpenAI Moderation API pre-filter flagged: {categories}",
            harmful_detected=True,
            harmful_findings=[Finding(
                page=0,  # moderation API doesn't give page-level detail
                type="harmful",
                detail=f"OpenAI Moderation API flagged content: {categories}",
                source="moderation_api",
                confidence="confirmed",
            )],
            cleaned_text="",
        )

    # ── Stage 2: 4 Parallel LLM Analyzers ────
    analyzer_results = run_all_analyzers(pages)

    # Tally analyzer error state. Surfaced on the result so the validator
    # summary can distinguish "0 findings (all clean)" from "0 findings (all
    # errored)" — see WF-3257BB (2026-04-24): Google AI Studio key was dead
    # so all 4 analyzers errored, but the user-facing summary read like a
    # clean pass.
    analyzer_list = [
        analyzer_results.analyzer_a, analyzer_results.analyzer_b,
        analyzer_results.analyzer_c, analyzer_results.analyzer_d,
    ]
    error_count = sum(1 for r in analyzer_list if r.error is not None)
    error_messages = [r.error for r in analyzer_list if r.error is not None]

    all_failed = all(
        result.error is not None and not result.findings
        for result in analyzer_list
    )
    if all_failed:
        # Include the upstream error in the termination detail so the
        # operator sees WHY (e.g. "API key not valid") instead of just
        # CONTENT_SAFETY_UNAVAILABLE.
        first_error = error_messages[0] if error_messages else "unknown"
        return ContentSafetyResult(
            overall_status="TERMINATE",
            termination_reason="CONTENT_SAFETY_UNAVAILABLE",
            termination_detail=f"All 4 content safety analyzers errored — first error: {first_error[:200]}",
            error_message=f"All content safety analyzers failed — first: {first_error[:200]}",
            cleaned_text="",
            analyzer_error_count=error_count,
            analyzer_error_messages=error_messages[:4],
        )

    # ── Synthesizer ────
    synth_result = synthesize_findings(analyzer_results, pages)
    all_findings = findings_to_models(synth_result.findings)

    # ── Categorize findings ────
    harmful_findings = [f for f in all_findings if f.type in HARMFUL_TYPES]
    pii_findings = [f for f in all_findings if f.type in PII_TYPES]
    religious_political_findings = [f for f in all_findings if f.type in RELIGIOUS_POLITICAL_TYPES]
    copyright_findings = [f for f in all_findings if f.type in COPYRIGHT_TYPES]
    misinformation_findings = [f for f in all_findings if f.type in MISINFORMATION_TYPES]

    # ── Apply decision logic ────

    # Hard gate 1: Harmful content → TERMINATE
    if harmful_findings:
        first = harmful_findings[0]
        return ContentSafetyResult(
            overall_status="TERMINATE",
            termination_reason="HARMFUL_CONTENT",
            termination_detail=f"Page {first.page}: {first.detail}",
            harmful_detected=True,
            harmful_findings=harmful_findings,
            pii_detected=bool(pii_findings),
            pii_findings=pii_findings,
            religious_political_detected=bool(religious_political_findings),
            religious_political_findings=religious_political_findings,
            copyright_detected=bool(copyright_findings),
            copyright_findings=copyright_findings,
            misinformation_detected=bool(misinformation_findings),
            misinformation_findings=misinformation_findings,
            cleaned_text="",
            analyzer_error_count=error_count,
            analyzer_error_messages=error_messages[:4],
        )

    # Hard gate 2: Religious/Political → TERMINATE
    if religious_political_findings:
        first = religious_political_findings[0]
        return ContentSafetyResult(
            overall_status="TERMINATE",
            termination_reason="RELIGIOUS_POLITICAL_VIOLATION",
            termination_detail=f"Page {first.page}: {first.detail}",
            religious_political_detected=True,
            religious_political_findings=religious_political_findings,
            pii_detected=bool(pii_findings),
            pii_findings=pii_findings,
            copyright_detected=bool(copyright_findings),
            copyright_findings=copyright_findings,
            misinformation_detected=bool(misinformation_findings),
            misinformation_findings=misinformation_findings,
            cleaned_text="",
            analyzer_error_count=error_count,
            analyzer_error_messages=error_messages[:4],
        )

    # Apply PII redactions
    cleaned_text = _apply_redactions(pages, pii_findings)

    # Determine overall status
    has_soft_warnings = bool(copyright_findings or misinformation_findings)

    if has_soft_warnings:
        overall_status = "PROCEED_WITH_WARNINGS"
    else:
        overall_status = "PROCEED"

    # Build assessor warnings — ready to display in UI, no querying needed
    assessor_warnings = []
    for f in pii_findings:
        assessor_warnings.append({
            "page": f.page,
            "type": "pii_redacted",
            "detail": f"Found {f.type} and auto-redacted",
        })
    for f in copyright_findings:
        assessor_warnings.append({
            "page": f.page,
            "type": "copyright",
            "detail": f.detail,
        })
    for f in misinformation_findings:
        assessor_warnings.append({
            "page": f.page,
            "type": "misinformation",
            "detail": f.detail,
        })

    return ContentSafetyResult(
        overall_status=overall_status,
        pii_detected=bool(pii_findings),
        pii_findings=pii_findings,
        copyright_detected=bool(copyright_findings),
        copyright_findings=copyright_findings,
        misinformation_detected=bool(misinformation_findings),
        misinformation_findings=misinformation_findings,
        cleaned_text=cleaned_text,
        assessor_action_required=has_soft_warnings,
        assessor_warnings=assessor_warnings,
        analyzer_error_count=error_count,
        analyzer_error_messages=error_messages[:4],
    )
