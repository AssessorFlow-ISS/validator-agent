"""Stub replacement for Thet's 3-component validate_file() pipeline.

Uses keyword detection on file content to simulate MRC, OCR, and Content Safety
results. Returns ValidatorResult matching Thet's schema for integration with
Dale's ValidatorService bridge layer.

No external calls — safe for unit tests and CI.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from validator_agent.pipeline.models import (
    ContentSafetyResult,
    MrcPageResult,
    MrcResult,
    OcrResult,
    PageOcrResult,
    ValidatorResult,
)

_HARMFUL_KEYWORDS = ("extremist", "manifesto", "bomb", "weapon")
_PII_PATTERNS = (r"SSN:\s*\d", r"social security", r"[\w.+-]+@[\w-]+\.[\w.]+")
_COPYRIGHT_KEYWORDS = ("ISBN:", "All rights reserved", "Copyright ©", "Copyright (c)")

_DEFAULT_TEXT = (
    "This is a sample educational document about software engineering principles. "
    "It covers topics such as design patterns, testing strategies, and agile methodologies. "
    "Students should review chapters 1 through 5 before the assessment."
)


def make_stub_pipeline(
    *,
    mrc_readiness: bool = True,
    mrc_confidence: float = 0.95,
    ocr_text: str | None = None,
    min_ocr_length: int = 50,
) -> Callable[[bytes, str], ValidatorResult]:
    """Factory for configurable stub pipeline functions.

    Args:
        mrc_readiness: Simulate MRC readability result.
        mrc_confidence: Simulate MRC confidence score.
        ocr_text: Override OCR output text. None = use file content or default.
        min_ocr_length: Minimum text length for OCR pass (matches Thet's pipeline).
    """
    def _fn(file_bytes: bytes, file_name: str) -> ValidatorResult:
        return _stub_pipeline_impl(
            file_bytes, file_name,
            mrc_readiness=mrc_readiness,
            mrc_confidence=mrc_confidence,
            ocr_text_override=ocr_text,
            min_ocr_length=min_ocr_length,
        )
    return _fn


def stub_pipeline_fn(file_bytes: bytes, file_name: str) -> ValidatorResult:
    """Default stub replacement for Thet's validate_file()."""
    return _stub_pipeline_impl(file_bytes, file_name)


def _stub_pipeline_impl(
    file_bytes: bytes,
    file_name: str,
    *,
    mrc_readiness: bool = True,
    mrc_confidence: float = 0.95,
    ocr_text_override: str | None = None,
    min_ocr_length: int = 50,
) -> ValidatorResult:
    """Configurable stub implementation."""
    try:
        content = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        content = ""

    mrc = MrcResult(
        overall_readiness=mrc_readiness,
        overall_confidence=mrc_confidence,
        total_pages=1,
        readable_pages=1 if mrc_readiness else 0,
        unreadable_pages=0 if mrc_readiness else 1,
        page_results=[MrcPageResult(page=1, readiness=mrc_readiness, confidence=mrc_confidence)],
        file_name=file_name,
        inference_time_ms=50.0,
        overall_status="PROCEED" if mrc_readiness else "TERMINATE",
        blurry_ratio=0.0 if mrc_readiness else 1.0,
        readable_page_numbers=[1] if mrc_readiness else [],
    )

    # MRC failure → early termination
    if not mrc_readiness:
        return ValidatorResult(
            file_name=file_name,
            overall_status="TERMINATE",
            termination_reason="BLURRY_UNREADABLE",
            termination_detail=f"Document unreadable (confidence={mrc_confidence})",
            terminated_at_component="mrc",
            mrc=mrc,
            total_time_ms=100.0,
        )

    # Determine OCR text
    if ocr_text_override is not None:
        ocr_text = ocr_text_override
    else:
        ocr_text = content if len(content) > 50 else _DEFAULT_TEXT

    # OCR length check
    if len(ocr_text.strip()) < min_ocr_length:
        return ValidatorResult(
            file_name=file_name,
            overall_status="TERMINATE",
            termination_reason="OCR_FAILED",
            termination_detail=f"OCR extracted insufficient text ({len(ocr_text.strip())} chars)",
            terminated_at_component="ocr",
            mrc=mrc,
            ocr=OcrResult(
                file_name=file_name, total_pages=1,
                pages=[PageOcrResult(page_number=1, extracted_text=ocr_text, word_count=len(ocr_text.split()))],
                total_word_count=len(ocr_text.split()), ocr_time_ms=100.0,
                overall_status="TERMINATE", success=False,
            ),
            total_time_ms=150.0,
        )
    ocr = OcrResult(
        file_name=file_name,
        total_pages=1,
        pages=[PageOcrResult(
            page_number=1,
            extracted_text=ocr_text,
            word_count=len(ocr_text.split()),
            confidence=0.98,
            classification="TEXT",
            source="ocr",
        )],
        total_word_count=len(ocr_text.split()),
        ocr_time_ms=100.0,
        overall_status="PROCEED",
        success=True,
    )

    content_lower = content.lower()

    for keyword in _HARMFUL_KEYWORDS:
        if keyword.lower() in content_lower:
            return ValidatorResult(
                file_name=file_name,
                overall_status="TERMINATE",
                termination_reason="HARMFUL_CONTENT",
                termination_detail=f"Harmful content detected: contains '{keyword}' reference",
                terminated_at_component="content_safety",
                mrc=mrc, ocr=ocr,
                content_safety=ContentSafetyResult(
                    overall_status="TERMINATE",
                    termination_reason="HARMFUL_CONTENT",
                    harmful_detected=True,
                ),
                total_time_ms=200.0,
            )

    for pattern in _PII_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return ValidatorResult(
                file_name=file_name,
                overall_status="TERMINATE",
                termination_reason="PII_DETECTED",
                termination_detail="Personally identifiable information detected",
                terminated_at_component="content_safety",
                mrc=mrc, ocr=ocr,
                content_safety=ContentSafetyResult(
                    overall_status="TERMINATE",
                    termination_reason="PII_DETECTED",
                    pii_detected=True,
                ),
                total_time_ms=200.0,
            )

    for keyword in _COPYRIGHT_KEYWORDS:
        if keyword.lower() in content_lower:
            return ValidatorResult(
                file_name=file_name,
                overall_status="TERMINATE",
                termination_reason="COPYRIGHT_VIOLATION",
                termination_detail=f"Copyright violation detected: contains '{keyword}'",
                terminated_at_component="content_safety",
                mrc=mrc, ocr=ocr,
                content_safety=ContentSafetyResult(
                    overall_status="TERMINATE",
                    termination_reason="COPYRIGHT_VIOLATION",
                    copyright_detected=True,
                ),
                total_time_ms=200.0,
            )

    return ValidatorResult(
        file_name=file_name,
        overall_status="PROCEED",
        mrc=mrc, ocr=ocr,
        content_safety=ContentSafetyResult(
            overall_status="PROCEED",
            cleaned_text=ocr_text,
        ),
        cleaned_text=ocr_text,
        total_time_ms=200.0,
    )
