"""Unit tests for ValidatorService static helpers (reasoning steps, overall signal).

These are pure functions over pipeline results, so no adapters are required.
"""

from __future__ import annotations

from datetime import UTC, datetime

from validator_agent.api.schemas import FileInfo, FileResult
from validator_agent.domain.services import ValidatorService
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)
from validator_agent.pipeline.models import (
    ContentSafetyResult,
    Finding,
    MrcPageResult,
    MrcResult,
    OcrResult,
    PageOcrResult,
    ValidatorResult,
)

FILE_INFO = FileInfo(file_name="doc.pdf", storage_path="doc.pdf", file_type="pdf")


def _mrc(
    *,
    overall_status: str = "PROCEED",
    total: int = 3,
    readable: int = 3,
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
        inference_time_ms=12.0,
        overall_status=overall_status,
        blurry_ratio=blurry_ratio,
        readable_page_numbers=[i + 1 for i in range(readable)],
        excluded_pages=excluded or [],
    )


def _ocr(
    *,
    overall_status: str = "PROCEED",
    pages: list[PageOcrResult] | None = None,
    harmful: bool = False,
    harmful_detail: str | None = None,
    harmful_page: int | None = None,
    visual_processed: int = 0,
    processing_mode: str | None = "normal",
) -> OcrResult:
    return OcrResult(
        file_name="doc.pdf",
        total_pages=len(pages) if pages is not None else 1,
        pages=pages
        or [PageOcrResult(page_number=1, extracted_text="hi", word_count=1, classification="TEXT", source="ocr")],
        total_word_count=1,
        ocr_time_ms=30.0,
        overall_status=overall_status,
        success=overall_status != "TERMINATE",
        harmful_image_detected=harmful,
        harmful_image_detail=harmful_detail,
        harmful_image_page=harmful_page,
        visual_pages_processed=visual_processed,
        processing_mode=processing_mode,
    )


def _safety(
    *,
    overall_status: str = "PROCEED",
    termination_reason: str | None = None,
    termination_detail: str | None = None,
    harmful_detected: bool = False,
    harmful_findings: list[Finding] | None = None,
    pii_findings: list[Finding] | None = None,
    copyright_findings: list[Finding] | None = None,
    religious_political_findings: list[Finding] | None = None,
    misinformation_findings: list[Finding] | None = None,
    religious_political_detected: bool = False,
    pii_detected: bool = False,
    copyright_detected: bool = False,
    misinformation_detected: bool = False,
) -> ContentSafetyResult:
    return ContentSafetyResult(
        overall_status=overall_status,
        termination_reason=termination_reason,
        termination_detail=termination_detail,
        harmful_detected=harmful_detected,
        harmful_findings=harmful_findings or [],
        pii_findings=pii_findings or [],
        copyright_findings=copyright_findings or [],
        religious_political_findings=religious_political_findings or [],
        misinformation_findings=misinformation_findings or [],
        religious_political_detected=religious_political_detected,
        pii_detected=pii_detected,
        copyright_detected=copyright_detected,
        misinformation_detected=misinformation_detected,
    )


def _result(
    *,
    mrc: MrcResult | None = None,
    ocr: OcrResult | None = None,
    content_safety: ContentSafetyResult | None = None,
    overall_status: str = "PROCEED",
    terminated_at_component: str | None = None,
    cleaned_text: str = "",
) -> ValidatorResult:
    return ValidatorResult(
        file_name="doc.pdf",
        overall_status=overall_status,
        mrc=mrc,
        ocr=ocr,
        content_safety=content_safety,
        cleaned_text=cleaned_text,
        terminated_at_component=terminated_at_component,
    )


class TestBuildReasoningStepsMrc:
    def test_mrc_only_proceed_with_timestamp(self) -> None:
        start = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        res = _result(mrc=_mrc())
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res, pipeline_start=start)
        assert steps[0]["component"] == "mrc"
        assert "timestamp" in steps[0]

    def test_mrc_terminate_returns_only_mrc_step(self) -> None:
        res = _result(
            mrc=_mrc(overall_status="TERMINATE", unreadable=3, readable=0),
            overall_status="TERMINATE",
            terminated_at_component="mrc",
        )
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        assert len(steps) == 1
        assert steps[0]["status"] == "TERMINATE"

    def test_mrc_excluded_pages_in_action(self) -> None:
        res = _result(mrc=_mrc(excluded=[2]))
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        assert "excluded pages: [2]" in steps[0]["action"]


class TestBuildReasoningStepsOcr:
    def test_ocr_terminate_harmful_image(self) -> None:
        mrc = _mrc()
        ocr = _ocr(
            overall_status="TERMINATE",
            harmful=True,
            harmful_detail="nude",
            harmful_page=1,
        )
        res = _result(mrc=mrc, ocr=ocr, terminated_at_component="ocr")
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        # Expect ocr step + page_classification + image_moderation termination
        comps = [s["component"] for s in steps]
        assert "image_moderation" in comps

    def test_ocr_with_visual_pages_processed(self) -> None:
        pages = [
            PageOcrResult(page_number=1, extracted_text="a", word_count=1, classification="TEXT", source="ocr"),
            PageOcrResult(
                page_number=2,
                extracted_text="b",
                word_count=1,
                classification="VISUAL",
                source="llm",
                visual_attempts=1,
            ),
        ]
        ocr = _ocr(pages=pages, visual_processed=1)
        safety = _safety()
        res = _result(mrc=_mrc(total=2, readable=2), ocr=ocr, content_safety=safety, cleaned_text="clean")
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        comps = [s["component"] for s in steps]
        assert "visual_understanding" in comps
        assert "knowledge_service" in comps


class TestBuildReasoningStepsSafety:
    def test_safety_harmful_text_moderation_terminate(self) -> None:
        safety = _safety(
            overall_status="TERMINATE",
            termination_reason="HARMFUL_CONTENT",
            termination_detail="banned",
            harmful_detected=False,  # No findings → moderation pre-filter path
        )
        res = _result(mrc=_mrc(), ocr=_ocr(), content_safety=safety, overall_status="TERMINATE")
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        # Should terminate at text_moderation
        assert any(s["component"] == "text_moderation" and s["status"] == "TERMINATE" for s in steps)

    def test_safety_hard_gate_religious_political(self) -> None:
        f = Finding(page=1, type="religious_political", detail="x")
        safety = _safety(
            overall_status="TERMINATE",
            termination_reason="RELIGIOUS_POLITICAL_VIOLATION",
            religious_political_detected=True,
            religious_political_findings=[f],
        )
        res = _result(mrc=_mrc(), ocr=_ocr(), content_safety=safety, overall_status="TERMINATE")
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        comps = [s["component"] for s in steps]
        assert "content_fit_decision" in comps
        terminate_step = [s for s in steps if s["component"] == "content_fit_decision"][-1]
        assert terminate_step["status"] == "TERMINATE"

    def test_safety_terminate_analyzers_unavailable(self) -> None:
        safety = _safety(
            overall_status="TERMINATE",
            termination_reason="CONTENT_SAFETY_UNAVAILABLE",
            harmful_findings=[Finding(page=1, type="x", detail="y")],  # non-empty → skips moderation branch
        )
        res = _result(mrc=_mrc(), ocr=_ocr(), content_safety=safety, overall_status="TERMINATE")
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        # Should end with content_fit_decision terminate
        assert steps[-1]["status"] == "TERMINATE"

    def test_safety_soft_warnings_included(self) -> None:
        safety = _safety(
            overall_status="PROCEED_WITH_WARNINGS",
            pii_detected=True,
            pii_findings=[Finding(page=1, type="pii", detail="ssn")],
            copyright_detected=True,
            copyright_findings=[Finding(page=1, type="copy", detail="c")],
            misinformation_detected=True,
            misinformation_findings=[Finding(page=1, type="m", detail="m")],
        )
        res = _result(
            mrc=_mrc(), ocr=_ocr(), content_safety=safety, cleaned_text="ok", overall_status="PROCEED_WITH_WARNINGS"
        )
        steps = ValidatorService._build_reasoning_steps(FILE_INFO, res)
        fit = [s for s in steps if s["component"] == "content_fit_decision"][-1]
        assert "PII redacted" in fit["action"]
        assert "copyright warned" in fit["action"]
        assert "misinformation warned" in fit["action"]


class TestComputeOverallSignal:
    def _fr(
        self, name: str, status: TerminalSignalStatus, code: ReasonCode = ReasonCode.VALIDATION_PASSED
    ) -> FileResult:
        return FileResult(
            file_name=name,
            terminal_signal=TerminalSignal(status=status, reason_code=code, message="m"),
        )

    def test_single_file_returns_its_signal(self) -> None:
        fr = self._fr("a", TerminalSignalStatus.PROCEED_WITH_WARNINGS, ReasonCode.PII_DETECTED)
        signal = ValidatorService._compute_overall_signal([fr])
        assert signal.reason_code == ReasonCode.PII_DETECTED

    def test_any_terminate_wins(self) -> None:
        results = [
            self._fr("a", TerminalSignalStatus.PROCEED),
            self._fr("b", TerminalSignalStatus.TERMINATE, ReasonCode.HARMFUL_CONTENT),
            self._fr("c", TerminalSignalStatus.PROCEED_WITH_WARNINGS),
        ]
        signal = ValidatorService._compute_overall_signal(results)
        assert signal.status == TerminalSignalStatus.TERMINATE
        assert signal.reason_code == ReasonCode.HARMFUL_CONTENT

    def test_warnings_propagate_when_no_terminate(self) -> None:
        results = [
            self._fr("a", TerminalSignalStatus.PROCEED),
            self._fr("b", TerminalSignalStatus.PROCEED_WITH_WARNINGS, ReasonCode.PII_DETECTED),
        ]
        signal = ValidatorService._compute_overall_signal(results)
        assert signal.status == TerminalSignalStatus.PROCEED_WITH_WARNINGS
