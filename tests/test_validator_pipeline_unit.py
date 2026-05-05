"""Unit tests for validator_agent.pipeline.validator_pipeline.validate_file.

Mocks the three component entry points (check_readability, extract_text,
check_content_safety) to cover the TERMINATE branches and PROCEED path.
"""

from __future__ import annotations

from unittest.mock import patch

from validator_agent.pipeline.models import (
    ContentSafetyResult,
    MrcPageResult,
    MrcResult,
    OcrResult,
    PageOcrResult,
)
from validator_agent.pipeline.validator_pipeline import validate_file


def _mrc(
    *,
    overall_status: str = "PROCEED",
    readable: int = 1,
    total: int = 1,
    unreadable: int = 0,
    excluded: list[int] | None = None,
    blurry_ratio: float = 0.0,
) -> MrcResult:
    return MrcResult(
        overall_readiness=(unreadable == 0),
        overall_confidence=0.95,
        total_pages=total,
        readable_pages=readable,
        unreadable_pages=unreadable,
        page_results=[MrcPageResult(page=i + 1, readiness=True, confidence=0.95) for i in range(total)],
        file_name="doc.pdf",
        inference_time_ms=10.0,
        overall_status=overall_status,
        blurry_ratio=blurry_ratio,
        readable_page_numbers=[i + 1 for i in range(readable)],
        excluded_pages=excluded or [],
    )


def _ocr(
    *,
    overall_status: str = "PROCEED",
    harmful: bool = False,
    harmful_detail: str | None = None,
    error_message: str | None = None,
    pages: list[PageOcrResult] | None = None,
) -> OcrResult:
    return OcrResult(
        file_name="doc.pdf",
        total_pages=len(pages) if pages else 1,
        pages=pages or [PageOcrResult(page_number=1, extracted_text="Hello", word_count=1)],
        total_word_count=1,
        ocr_time_ms=20.0,
        overall_status=overall_status,
        success=(overall_status != "TERMINATE"),
        harmful_image_detected=harmful,
        harmful_image_detail=harmful_detail,
        error_message=error_message,
    )


def _safety(
    *,
    overall_status: str = "PROCEED",
    termination_reason: str | None = None,
    termination_detail: str | None = None,
    cleaned_text: str = "",
    assessor_warnings: list[dict] | None = None,
) -> ContentSafetyResult:
    return ContentSafetyResult(
        overall_status=overall_status,
        termination_reason=termination_reason,
        termination_detail=termination_detail,
        cleaned_text=cleaned_text,
        assessor_warnings=assessor_warnings or [],
    )


class TestValidateFile:
    """End-to-end pipeline orchestration."""

    def test_mrc_terminate_skips_ocr_and_safety(self) -> None:
        mrc_bad = _mrc(overall_status="TERMINATE", unreadable=5, total=5, readable=0, blurry_ratio=1.0)
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=mrc_bad) as mrc_mock,
            patch("validator_agent.pipeline.validator_pipeline.extract_text") as ocr_mock,
            patch("validator_agent.pipeline.validator_pipeline.check_content_safety") as safety_mock,
        ):
            result = validate_file(b"PDF", "doc.pdf")

        mrc_mock.assert_called_once()
        ocr_mock.assert_not_called()
        safety_mock.assert_not_called()
        assert result.overall_status == "TERMINATE"
        assert result.termination_reason == "BLURRY_UNREADABLE"
        assert result.terminated_at_component == "mrc"

    def test_ocr_harmful_image_terminates_with_reason(self) -> None:
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=_mrc()),
            patch(
                "validator_agent.pipeline.validator_pipeline.extract_text",
                return_value=_ocr(overall_status="TERMINATE", harmful=True, harmful_detail="nude imagery"),
            ),
            patch("validator_agent.pipeline.validator_pipeline.check_content_safety") as safety_mock,
        ):
            result = validate_file(b"PDF", "doc.pdf")

        safety_mock.assert_not_called()
        assert result.overall_status == "TERMINATE"
        assert result.termination_reason == "HARMFUL_IMAGE"
        assert "nude imagery" in (result.termination_detail or "")
        assert result.terminated_at_component == "ocr"

    def test_ocr_plain_failure_terminates_with_ocr_failed(self) -> None:
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=_mrc()),
            patch(
                "validator_agent.pipeline.validator_pipeline.extract_text",
                return_value=_ocr(overall_status="TERMINATE", error_message="no text"),
            ),
            patch("validator_agent.pipeline.validator_pipeline.check_content_safety"),
        ):
            result = validate_file(b"PDF", "doc.pdf")

        assert result.overall_status == "TERMINATE"
        assert result.termination_reason == "OCR_FAILED"

    def test_safety_terminate_propagates_reason(self) -> None:
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=_mrc()),
            patch("validator_agent.pipeline.validator_pipeline.extract_text", return_value=_ocr()),
            patch(
                "validator_agent.pipeline.validator_pipeline.check_content_safety",
                return_value=_safety(
                    overall_status="TERMINATE",
                    termination_reason="HARMFUL_CONTENT",
                    termination_detail="bad words",
                ),
            ),
        ):
            result = validate_file(b"PDF", "doc.pdf")

        assert result.overall_status == "TERMINATE"
        assert result.termination_reason == "HARMFUL_CONTENT"
        assert result.terminated_at_component == "content_safety"

    def test_all_proceed_returns_proceed_with_cleaned_text(self) -> None:
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=_mrc()),
            patch("validator_agent.pipeline.validator_pipeline.extract_text", return_value=_ocr()),
            patch(
                "validator_agent.pipeline.validator_pipeline.check_content_safety",
                return_value=_safety(cleaned_text="cleaned output"),
            ),
        ):
            result = validate_file(b"PDF", "doc.pdf")

        assert result.overall_status == "PROCEED"
        assert result.cleaned_text == "cleaned output"
        assert result.terminated_at_component is None

    def test_mrc_exclusions_upgrade_to_warnings(self) -> None:
        """MRC excludes blurry pages → overall becomes PROCEED_WITH_WARNINGS."""
        mrc_with_excl = _mrc(excluded=[3, 7])
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=mrc_with_excl),
            patch("validator_agent.pipeline.validator_pipeline.extract_text", return_value=_ocr()),
            patch(
                "validator_agent.pipeline.validator_pipeline.check_content_safety",
                return_value=_safety(cleaned_text="x"),
            ),
        ):
            result = validate_file(b"PDF", "doc.pdf")

        assert result.overall_status == "PROCEED_WITH_WARNINGS"
        excl_warnings = [w for w in result.assessor_warnings if w.get("type") == "page_excluded"]
        assert len(excl_warnings) == 2

    def test_safety_warnings_merged_into_result(self) -> None:
        extras = [{"page": 1, "type": "pii_redacted", "detail": "SSN redacted"}]
        with (
            patch("validator_agent.pipeline.validator_pipeline.check_readability", return_value=_mrc()),
            patch("validator_agent.pipeline.validator_pipeline.extract_text", return_value=_ocr()),
            patch(
                "validator_agent.pipeline.validator_pipeline.check_content_safety",
                return_value=_safety(
                    overall_status="PROCEED_WITH_WARNINGS",
                    cleaned_text="x",
                    assessor_warnings=extras,
                ),
            ),
        ):
            result = validate_file(b"PDF", "doc.pdf")

        assert result.overall_status == "PROCEED_WITH_WARNINGS"
        assert extras[0] in result.assessor_warnings
