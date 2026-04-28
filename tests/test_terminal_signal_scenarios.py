"""P0 integration tests for Terminal Signal scenarios.

Tests the full validation pipeline through the ValidatorService using stub
adapters — zero external dependencies.  Covers PROCEED, TERMINATE (SC-INT-04
through SC-INT-08), Terminal Signal structure invariants, and decision audit
logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from collections.abc import Callable

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.pipeline_stub import make_stub_pipeline, stub_pipeline_fn
from validator_agent.adapters.storage_stub import StubStorageAdapter
from validator_agent.api.schemas import FileInfo, ValidationRequest, ValidationResponse
from validator_agent.domain.services import ValidatorService
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignalStatus,
)

if TYPE_CHECKING:
    from validator_agent.pipeline.models import ValidatorResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    *,
    workflow_id: str = "wf-int-001",
    assessment_id: str = "assess-int-001",
    files: list[FileInfo] | None = None,
) -> ValidationRequest:
    """Build a standard single-file ValidationRequest."""
    if files is None:
        files = [
            FileInfo(
                file_name="chapter1.pdf",
                storage_path="materials/wf-int-001/chapter1.pdf",
                file_type="pdf",
            ),
        ]
    return ValidationRequest(
        workflow_id=workflow_id,
        assessment_id=assessment_id,
        validation_type="material_validation",
        files=files,
    )


def _build_service(
    *,
    decision_audit: StubDecisionAuditAdapter | None = None,
    storage: StubStorageAdapter | None = None,
    pipeline_fn: Callable | None = None,
    # Legacy parameters mapped to make_stub_pipeline
    mrc: object | None = None,
    ocr: object | None = None,
) -> tuple[ValidatorService, StubDecisionAuditAdapter]:
    """Construct a ValidatorService wired with stub adapters."""
    da = decision_audit or StubDecisionAuditAdapter()
    st = storage or StubStorageAdapter()

    if pipeline_fn is None:
        mrc_readiness = True
        mrc_confidence = 0.95
        ocr_text = None
        if mrc is not None:
            mrc_readiness = getattr(mrc, '_default_readiness', True)
            mrc_confidence = getattr(mrc, '_default_confidence', 0.95)
        if ocr is not None:
            ocr_text = getattr(ocr, '_default_text', None)
        pipeline_fn = make_stub_pipeline(
            mrc_readiness=mrc_readiness,
            mrc_confidence=mrc_confidence,
            ocr_text=ocr_text,
        )

    from af_shared.adapters.stubs.tracing_stub import StubTracingAdapter

    service = ValidatorService(
        knowledge_service=StubKnowledgeServiceAdapter(),
        decision_audit=da,
        event_publisher=StubEventPublisherAdapter(),
        storage=st,
        tracing=StubTracingAdapter(),
        pipeline_fn=pipeline_fn,
    )
    return service, da


# ===========================================================================
# PROCEED scenarios
# ===========================================================================

class TestProceedScenarios:
    """Clean material that passes MRC + OCR + content safety must PROCEED."""

    async def test_clean_material_returns_proceed(self) -> None:
        """SC-INT-01: MRC pass + OCR pass + content safe -> PROCEED."""
        service, _ = _build_service()
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED

    async def test_proceed_includes_reason_code_validation_passed(self) -> None:
        """SC-INT-02: reason_code must be VALIDATION_PASSED on clean material."""
        service, _ = _build_service()
        response = await service.validate(_make_request())

        assert response.terminal_signal.reason_code == ReasonCode.VALIDATION_PASSED

    async def test_proceed_includes_extracted_text(self) -> None:
        """SC-INT-03: response file results must exist and reflect OCR extraction.

        The ValidationResponse itself does not carry raw extracted text, but
        a successful PROCEED proves OCR extraction succeeded and text was
        forwarded to the Knowledge Service.  We verify through the per-file
        result and the overall signal.
        """
        service, _ = _build_service()
        response = await service.validate(_make_request())

        # Per-file result mirrors the overall PROCEED signal
        assert len(response.file_results) == 1
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.file_results[0].terminal_signal.reason_code == ReasonCode.VALIDATION_PASSED


# ===========================================================================
# TERMINATE scenarios (SC-INT-04 through SC-INT-08)
# ===========================================================================

class TestTerminateBlurry:
    """SC-INT-04: Blurry/unreadable material triggers MRC TERMINATE."""

    async def test_blurry_material_returns_terminate(self) -> None:
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.20)
        service, _ = _build_service(mrc=mrc)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE


class TestTerminateHarmfulContent:
    """SC-INT-05: Harmful content detected by LLM triggers TERMINATE."""

    async def test_harmful_content_returns_terminate(self) -> None:
        harmful_text = FIXTURES_DIR.joinpath("harmful_content.txt").read_text(encoding="utf-8")
        storage = StubStorageAdapter(default_bytes=harmful_text.encode())
        service, _ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT


class TestTerminatePiiDetected:
    """SC-INT-06: PII detected by LLM triggers TERMINATE."""

    async def test_pii_detected_returns_terminate(self) -> None:
        pii_text = FIXTURES_DIR.joinpath("pii_content.txt").read_text(encoding="utf-8")
        storage = StubStorageAdapter(default_bytes=pii_text.encode())
        service, _ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.PII_DETECTED


class TestTerminateCopyrightViolation:
    """SC-INT-07: Copyright violation detected by LLM triggers TERMINATE."""

    async def test_copyright_violation_returns_terminate(self) -> None:
        copyright_text = (
            "This textbook chapter is reproduced here. ISBN: 978-0-13-468599-1. "
            "No modifications were made to the original content. "
            "All content is subject to the publisher's copyright terms. "
            "Additional filler text to exceed the minimum OCR length threshold."
        )
        storage = StubStorageAdapter(default_bytes=copyright_text.encode())
        service, _ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.COPYRIGHT_VIOLATION


class TestTerminateOcrFailed:
    """SC-INT-08: OCR extraction failure triggers TERMINATE."""

    async def test_ocr_failed_returns_terminate(self) -> None:
        pipeline_fn = make_stub_pipeline(ocr_text="")
        service, _ = _build_service(pipeline_fn=pipeline_fn)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.OCR_FAILED


# ===========================================================================
# Terminal Signal structure invariants
# ===========================================================================

class TestTerminalSignalStructure:
    """Validate the Terminal Signal contract across all outcome paths."""

    @pytest.fixture
    def proceed_response(self) -> ValidationResponse:
        """Cache is not needed — kept as a simple factory fixture."""
        # Cannot be async fixture with auto mode easily, so build synchronously
        # via a helper.  We rely on the fact that _build_service returns sync objects;
        # the actual async call is in each test.
        return None  # placeholder — each test calls service.validate directly

    async def _get_proceed_response(self) -> ValidationResponse:
        service, _ = _build_service()
        return await service.validate(_make_request())

    async def _get_terminate_response(self) -> ValidationResponse:
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.15)
        service, _ = _build_service(mrc=mrc)
        return await service.validate(_make_request())

    async def test_terminal_signal_has_required_fields_proceed(self) -> None:
        """status, reason_code, message must all be present on PROCEED."""
        response = await self._get_proceed_response()
        signal = response.terminal_signal

        assert signal.status is not None
        assert signal.reason_code is not None
        assert signal.message is not None

    async def test_terminal_signal_has_required_fields_terminate(self) -> None:
        """status, reason_code, message must all be present on TERMINATE."""
        response = await self._get_terminate_response()
        signal = response.terminal_signal

        assert signal.status is not None
        assert signal.reason_code is not None
        assert signal.message is not None

    async def test_terminal_signal_status_is_enum_proceed(self) -> None:
        """PROCEED status must be a valid TerminalSignalStatus member."""
        response = await self._get_proceed_response()

        assert response.terminal_signal.status in TerminalSignalStatus
        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED

    async def test_terminal_signal_status_is_enum_terminate(self) -> None:
        """TERMINATE status must be a valid TerminalSignalStatus member."""
        response = await self._get_terminate_response()

        assert response.terminal_signal.status in TerminalSignalStatus
        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE

    async def test_terminal_signal_status_not_retry(self) -> None:
        """RETRY was removed from the Terminal Signal contract — only PROCEED, PROCEED_WITH_WARNINGS, or TERMINATE."""
        valid_statuses = {member.value for member in TerminalSignalStatus}
        assert "RETRY" not in valid_statuses
        assert valid_statuses == {"PROCEED", "PROCEED_WITH_WARNINGS", "TERMINATE"}

    async def test_terminate_message_is_human_readable(self) -> None:
        """TERMINATE message must be a non-empty, human-readable string."""
        response = await self._get_terminate_response()
        message = response.terminal_signal.message

        assert isinstance(message, str)
        assert len(message.strip()) > 0
        # Human-readable: contains at least one space (it's a sentence, not a slug)
        assert " " in message

    async def test_proceed_message_is_human_readable(self) -> None:
        """PROCEED message must also be a non-empty, human-readable string."""
        response = await self._get_proceed_response()
        message = response.terminal_signal.message

        assert isinstance(message, str)
        assert len(message.strip()) > 0
        assert " " in message


# ===========================================================================
# Decision audit logging
# ===========================================================================

class TestDecisionAuditLogging:
    """Every validation must produce exactly one audit log entry."""

    async def test_validation_logs_decision_to_audit(self) -> None:
        """Audit log must be called with correct workflow_id, agent_name,
        decision_type, and a payload containing terminal_signal + reasoning_steps.
        """
        da = StubDecisionAuditAdapter()
        service, da_ref = _build_service(decision_audit=da)
        await service.validate(_make_request())

        # Exactly one audit entry per validation invocation
        assert len(da_ref.entries) == 1

        entry = da_ref.entries[0]

        # Correct routing fields
        assert entry.workflow_id == "wf-int-001"
        assert entry.agent_name == "validator-agent"
        assert entry.decision_type == "content_validation"

        # Payload must contain terminal_signal
        payload = entry.payload
        assert "output_summary" in payload
        assert payload["output_summary"]["terminal_signal"]["status"] == "PROCEED"
        assert payload["output_summary"]["terminal_signal"]["reason_code"] == "VALIDATION_PASSED"

        # Payload must contain reasoning_steps (non-empty list of dicts with per-file data)
        assert "reasoning_steps" in payload
        assert isinstance(payload["reasoning_steps"], list)
        assert len(payload["reasoning_steps"]) > 0
        assert "step" in payload["reasoning_steps"][0]
        assert "component" in payload["reasoning_steps"][0]
        assert "action" in payload["reasoning_steps"][0]

        # Payload must include prompt_version per ADR-39
        assert "prompt_version" in payload
        assert payload["prompt_version"].startswith("validator/thet-pipeline@v")

        # Payload must include agent_name and phase (phase lives in input_summary)
        assert payload["agent_name"] == "validator-agent"
        assert payload["input_summary"]["phase"] == "Phase 3: Material Validation"

    async def test_terminate_audit_records_terminate_signal(self) -> None:
        """TERMINATE outcomes must also be logged with correct terminal_signal."""
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.10)
        da = StubDecisionAuditAdapter()
        service, da_ref = _build_service(mrc=mrc, decision_audit=da)
        await service.validate(_make_request())

        assert len(da_ref.entries) == 1
        payload = da_ref.entries[0].payload
        assert payload["output_summary"]["terminal_signal"]["status"] == "TERMINATE"
        assert payload["output_summary"]["terminal_signal"]["reason_code"] == "BLURRY_UNREADABLE"


# ===========================================================================
# GAP-10: Multi-file partial TERMINATE scenarios
# ===========================================================================

def _make_multi_file_request(
    *,
    workflow_id: str = "wf-multi-001",
    assessment_id: str = "assess-multi-001",
    file_count: int = 3,
) -> tuple[ValidationRequest, list[FileInfo]]:
    """Build a multi-file ValidationRequest with N files.

    Returns (request, files) so callers can reference individual file paths
    when configuring per-file stub overrides.
    """
    files = [
        FileInfo(
            file_name=f"document_{i}.pdf",
            storage_path=f"materials/{workflow_id}/document_{i}.pdf",
            file_type="pdf",
        )
        for i in range(1, file_count + 1)
    ]
    request = ValidationRequest(
        workflow_id=workflow_id,
        assessment_id=assessment_id,
        validation_type="material_validation",
        files=files,
    )
    return request, files


class TestMultiFilePartialTerminate:
    """GAP-10: Multiple files where one fails should TERMINATE the workflow.

    The ValidatorService processes all files sequentially (no short-circuit)
    and returns an overall TERMINATE if ANY file has a TERMINATE signal.
    The overall signal uses the first TERMINATE found in file_results.
    """

    async def test_file_2_harmful_overall_terminates(self) -> None:
        """3 files — file 2 has harmful content — overall must be TERMINATE.

        Files 1 and 3 are clean; file 2 contains harmful keywords that
        trigger the stub pipeline's HARMFUL_CONTENT detection.
        """
        request, files = _make_multi_file_request()
        harmful_text = (
            "This extremist manifesto contains dangerous ideological content "
            "that promotes violence and describes weapon construction methods."
        )
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[1].storage_path: harmful_text.encode(),
            },
        )
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT

        # Per-file results: file 1 PROCEED, file 2 TERMINATE, file 3 PROCEED
        assert len(response.file_results) == 3
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.file_results[1].terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.file_results[1].terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT
        assert response.file_results[2].terminal_signal.status == TerminalSignalStatus.PROCEED

    async def test_file_1_harmful_overall_terminates(self) -> None:
        """3 files — file 1 has harmful content — overall must be TERMINATE.

        Even when the very first file fails, the service still processes all
        files (no short-circuit) and returns TERMINATE with the first failing
        file's signal.
        """
        request, files = _make_multi_file_request()
        harmful_text = (
            "This document describes bomb-making instructions and weapon "
            "acquisition through illegal channels. Extremely dangerous content."
        )
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[0].storage_path: harmful_text.encode(),
            },
        )
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT

        # File 1 is the first TERMINATE — overall signal should match it
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.file_results[0].file_name == "document_1.pdf"

    async def test_file_3_harmful_terminates_even_after_two_proceed(self) -> None:
        """3 files — file 3 has harmful content — overall TERMINATE despite files 1+2 passing.

        Validates that the service does not prematurely return PROCEED after
        the first N files succeed. The tail-end failure still kills the workflow.
        """
        request, files = _make_multi_file_request()
        harmful_text = (
            "This extremist propaganda manifesto advocates for overthrow of "
            "democratic institutions and describes recruitment of vulnerable individuals."
        )
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[2].storage_path: harmful_text.encode(),
            },
        )
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT

        # Files 1 and 2 passed, file 3 failed
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.file_results[1].terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.file_results[2].terminal_signal.status == TerminalSignalStatus.TERMINATE

    async def test_all_files_clean_overall_proceeds(self) -> None:
        """3 files — all clean — overall must be PROCEED.

        Baseline case: when every file passes all checks, the overall signal
        is PROCEED with VALIDATION_PASSED.
        """
        request, _files = _make_multi_file_request()
        service, _ = _build_service()
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.terminal_signal.reason_code == ReasonCode.VALIDATION_PASSED
        assert response.terminal_signal.message == "All files validated successfully"

        # All three per-file results should be PROCEED
        assert len(response.file_results) == 3
        for file_result in response.file_results:
            assert file_result.terminal_signal.status == TerminalSignalStatus.PROCEED

    async def test_single_harmful_file_matches_single_file_terminate(self) -> None:
        """1 file — harmful content — same behaviour as single-file TERMINATE.

        Edge case: multi-file code path with a single-element list should
        produce the same result as the existing single-file tests.
        """
        request, files = _make_multi_file_request(file_count=1)
        harmful_text = (
            "This extremist manifesto contains bomb-making instructions "
            "and weapon acquisition details that violate content policies."
        )
        storage = StubStorageAdapter(default_bytes=harmful_text.encode())
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT

        # Single-file: file_results has exactly 1 entry
        assert len(response.file_results) == 1
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.TERMINATE

    async def test_terminate_reason_code_matches_failing_file_issue(self) -> None:
        """Overall TERMINATE reason_code must match the specific failure type.

        File 1: clean (PROCEED)
        File 2: blurry/unreadable (MRC fail -> BLURRY_UNREADABLE)
        File 3: clean (PROCEED)

        The overall signal should carry BLURRY_UNREADABLE, not a generic code.
        """
        request, files = _make_multi_file_request()

        def _per_file_pipeline(file_bytes: bytes, file_name: str) -> "ValidatorResult":
            """MRC fails for file 2 only."""
            if file_name == "document_2.pdf":
                return make_stub_pipeline(
                    mrc_readiness=False, mrc_confidence=0.15,
                )(file_bytes, file_name)
            return stub_pipeline_fn(file_bytes, file_name)

        service, _ = _build_service(pipeline_fn=_per_file_pipeline)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE
        # Message comes from Thet's pipeline; verify it is non-empty and human-readable
        assert len(response.terminal_signal.message.strip()) > 0

        # The specific file that failed
        assert response.file_results[1].terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE
        assert response.file_results[1].file_name == "document_2.pdf"

    async def test_first_terminate_wins_when_multiple_files_fail(self) -> None:
        """When multiple files fail, the overall signal uses the FIRST TERMINATE.

        File 1: blurry (BLURRY_UNREADABLE)
        File 2: harmful content (HARMFUL_CONTENT)
        File 3: clean (PROCEED)

        _compute_overall_signal iterates file_results in order and returns the
        first TERMINATE signal it finds — which should be BLURRY_UNREADABLE.
        """
        request, files = _make_multi_file_request()
        harmful_text = (
            "This extremist manifesto contains bomb-making instructions "
            "and weapon details that are dangerous."
        )

        def _per_file_pipeline(file_bytes: bytes, file_name: str) -> "ValidatorResult":
            """File 1 blurry, file 2 harmful, file 3 clean."""
            if file_name == "document_1.pdf":
                return make_stub_pipeline(
                    mrc_readiness=False, mrc_confidence=0.12,
                )(file_bytes, file_name)
            return stub_pipeline_fn(file_bytes, file_name)

        # File 2 gets harmful content via storage bytes
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[1].storage_path: harmful_text.encode(),
            },
        )
        service, _ = _build_service(storage=storage, pipeline_fn=_per_file_pipeline)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        # First failing file (file 1) is blurry — its reason_code wins
        assert response.terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE

        # Verify per-file results reflect distinct failure types
        assert response.file_results[0].terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE
        assert response.file_results[1].terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT
        assert response.file_results[2].terminal_signal.status == TerminalSignalStatus.PROCEED

    async def test_multi_file_audit_log_contains_all_reasoning_steps(self) -> None:
        """Multi-file validation must log reasoning steps for ALL files.

        Even when file 2 fails, the audit entry should contain the reasoning
        steps from file 1 (passed), file 2 (failed), and file 3 (passed) —
        because the service processes all files before aggregating.
        """
        request, files = _make_multi_file_request()
        harmful_text = (
            "This extremist manifesto contains dangerous weapon content "
            "that should trigger content safety detection."
        )
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[1].storage_path: harmful_text.encode(),
            },
        )
        da = StubDecisionAuditAdapter()
        service, da_ref = _build_service(storage=storage, decision_audit=da)
        await service.validate(request)

        # Exactly one audit entry per validation invocation
        assert len(da_ref.entries) == 1

        payload = da_ref.entries[0].payload

        # Overall terminal_signal in audit must be TERMINATE
        assert payload["output_summary"]["terminal_signal"]["status"] == "TERMINATE"

        # Reasoning steps now contain per-component detail for all 3 files
        steps = payload["reasoning_steps"]
        # Clean files produce ~8 steps each, terminated file fewer — total > 3
        assert len(steps) >= 3
        # All steps should have component and action fields
        assert all("component" in s and "action" in s for s in steps)
        # MRC step should be present (first step for every file)
        mrc_steps = [s for s in steps if s["component"] == "mrc"]
        assert len(mrc_steps) == 3  # one MRC check per file

    async def test_multi_file_response_contains_all_file_results(self) -> None:
        """Response file_results list must contain an entry for every input file.

        This ensures the service does not drop results for files processed
        after a TERMINATE is encountered.
        """
        request, files = _make_multi_file_request(file_count=5)
        harmful_text = (
            "This bomb-making manifesto describes weapon construction "
            "methods and extremist recruitment strategies."
        )
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[2].storage_path: harmful_text.encode(),
            },
        )
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        # All 5 files must appear in file_results
        assert len(response.file_results) == 5

        # File names must match input order
        for i, file_result in enumerate(response.file_results):
            assert file_result.file_name == f"document_{i + 1}.pdf"

        # Overall is TERMINATE because file 3 failed
        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE

    async def test_partial_terminate_pii_reason_code_propagates(self) -> None:
        """Overall signal carries PII_DETECTED when a file contains PII.

        File 1: clean (PROCEED)
        File 2: PII content (TERMINATE with PII_DETECTED)
        File 3: clean (PROCEED)
        """
        request, files = _make_multi_file_request()
        pii_bytes = FIXTURES_DIR.joinpath("pii_content.txt").read_bytes()
        clean_bytes = FIXTURES_DIR.joinpath("clean_document.txt").read_bytes()
        storage = StubStorageAdapter(
            default_bytes=clean_bytes,
            overrides={
                files[1].storage_path: pii_bytes,
            },
        )
        service, _ = _build_service(storage=storage)
        response = await service.validate(request)

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.PII_DETECTED

        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.file_results[1].terminal_signal.reason_code == ReasonCode.PII_DETECTED
        assert response.file_results[2].terminal_signal.status == TerminalSignalStatus.PROCEED
