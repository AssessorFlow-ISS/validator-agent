"""Validator Agent — End-to-End Pipeline.

Wires all 3 components sequentially:
  Component 1: MRC (readability check) → PROCEED / PROCEED_WITH_EXCLUSIONS / TERMINATE
  Component 2: OCR (text extraction + visual understanding) → PROCEED / TERMINATE
  Component 3: Content Safety (5 checks) → PROCEED / PROCEED_WITH_WARNINGS / TERMINATE

Each component can terminate the pipeline. If terminated, subsequent components are skipped.
"""

from __future__ import annotations

import time

from validator_agent.pipeline.content_safety import check_content_safety
from validator_agent.pipeline.models import ValidatorResult
from validator_agent.pipeline.mrc_client import check_readability
from validator_agent.pipeline.ocr_pipeline import extract_text


def validate_file(file_bytes: bytes, file_name: str) -> ValidatorResult:
    """Run the full Validator Agent pipeline on a file.

    Args:
        file_bytes: Raw file content (PDF, PNG, JPG).
        file_name: Original file name.

    Returns:
        ValidatorResult with all 3 component results and final status.
    """
    start_ms = time.time() * 1000

    # ── Component 1: MRC ────
    mrc_result = check_readability(file_bytes, file_name)

    if mrc_result.overall_status == "TERMINATE":
        return ValidatorResult(
            file_name=file_name,
            overall_status="TERMINATE",
            termination_reason="BLURRY_UNREADABLE",
            termination_detail=f"{mrc_result.unreadable_pages}/{mrc_result.total_pages} pages unreadable ({mrc_result.blurry_ratio:.0%} > 30% threshold)",
            terminated_at_component="mrc",
            mrc=mrc_result,
            total_time_ms=round(time.time() * 1000 - start_ms, 2),
        )

    # Determine which pages to process
    readable_pages = mrc_result.readable_page_numbers if mrc_result.readable_page_numbers else None

    # ── Component 2: OCR ────
    ocr_result = extract_text(file_bytes, file_name, readable_pages=readable_pages)

    if ocr_result.overall_status == "TERMINATE":
        reason = "HARMFUL_IMAGE" if ocr_result.harmful_image_detected else "OCR_FAILED"
        detail = ocr_result.harmful_image_detail or ocr_result.error_message or "OCR failed"
        return ValidatorResult(
            file_name=file_name,
            overall_status="TERMINATE",
            termination_reason=reason,
            termination_detail=detail,
            terminated_at_component="ocr",
            mrc=mrc_result,
            ocr=ocr_result,
            total_time_ms=round(time.time() * 1000 - start_ms, 2),
        )

    # ── Component 3: Content Safety ────
    safety_result = check_content_safety(ocr_result.pages)

    if safety_result.overall_status == "TERMINATE":
        return ValidatorResult(
            file_name=file_name,
            overall_status="TERMINATE",
            termination_reason=safety_result.termination_reason,
            termination_detail=safety_result.termination_detail,
            terminated_at_component="content_safety",
            mrc=mrc_result,
            ocr=ocr_result,
            content_safety=safety_result,
            total_time_ms=round(time.time() * 1000 - start_ms, 2),
        )

    # ── All passed ────
    # Build assessor warnings from all components
    assessor_warnings = []

    # MRC exclusions
    if mrc_result.excluded_pages:
        for page_num in mrc_result.excluded_pages:
            assessor_warnings.append({
                "page": page_num,
                "type": "page_excluded",
                "detail": "Page excluded — blurry/unreadable (detected by MRC)",
            })

    # Content Safety warnings
    assessor_warnings.extend(safety_result.assessor_warnings)

    overall_status = safety_result.overall_status
    # If MRC excluded pages, upgrade to PROCEED_WITH_WARNINGS
    if mrc_result.excluded_pages and overall_status == "PROCEED":
        overall_status = "PROCEED_WITH_WARNINGS"

    return ValidatorResult(
        file_name=file_name,
        overall_status=overall_status,
        mrc=mrc_result,
        ocr=ocr_result,
        content_safety=safety_result,
        cleaned_text=safety_result.cleaned_text,
        assessor_warnings=assessor_warnings,
        total_time_ms=round(time.time() * 1000 - start_ms, 2),
    )
