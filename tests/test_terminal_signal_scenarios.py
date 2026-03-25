"""P0 integration tests for Terminal Signal scenarios.

Tests the full validation pipeline through the ValidatorService using stub
adapters — zero external dependencies.  Covers PROCEED, TERMINATE (SC-INT-04
through SC-INT-08), Terminal Signal structure invariants, and decision audit
logging.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter
from validator_agent.api.schemas import FileInfo, ValidationRequest, ValidationResponse
from validator_agent.domain.content_safety import ContentSafetyReasoner
from validator_agent.domain.services import ValidatorService
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)

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
    mrc: StubMrcAdapter | None = None,
    ocr: StubOcrAdapter | None = None,
    model_broker: StubModelBrokerAdapter | None = None,
    decision_audit: StubDecisionAuditAdapter | None = None,
) -> tuple[ValidatorService, StubDecisionAuditAdapter]:
    """Construct a ValidatorService wired with stub adapters.

    Returns the service and the decision-audit stub so callers can inspect
    logged entries.
    """
    mrc = mrc or StubMrcAdapter()
    ocr = ocr or StubOcrAdapter(
        default_text=FIXTURES_DIR.joinpath("clean_document.txt").read_text(encoding="utf-8"),
    )
    model_broker = model_broker or StubModelBrokerAdapter()
    da = decision_audit or StubDecisionAuditAdapter()

    content_safety = ContentSafetyReasoner(model_broker=model_broker)
    service = ValidatorService(
        mrc=mrc,
        ocr=ocr,
        content_safety=content_safety,
        knowledge_service=StubKnowledgeServiceAdapter(),
        decision_audit=da,
        event_publisher=StubEventPublisherAdapter(),
        storage=StubStorageAdapter(),
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
        ocr = StubOcrAdapter(default_text=harmful_text)
        service, _ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT


class TestTerminatePiiDetected:
    """SC-INT-06: PII detected by LLM triggers TERMINATE."""

    async def test_pii_detected_returns_terminate(self) -> None:
        pii_text = FIXTURES_DIR.joinpath("pii_content.txt").read_text(encoding="utf-8")
        ocr = StubOcrAdapter(default_text=pii_text)
        service, _ = _build_service(ocr=ocr)
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
        ocr = StubOcrAdapter(default_text=copyright_text)
        service, _ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.COPYRIGHT_VIOLATION


class TestTerminateOcrFailed:
    """SC-INT-08: OCR extraction failure triggers TERMINATE."""

    async def test_ocr_failed_returns_terminate(self) -> None:
        ocr = StubOcrAdapter(default_text="")
        service, _ = _build_service(ocr=ocr)
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
        """RETRY was removed from the Terminal Signal contract — only PROCEED or TERMINATE."""
        valid_statuses = {member.value for member in TerminalSignalStatus}
        assert "RETRY" not in valid_statuses
        assert valid_statuses == {"PROCEED", "TERMINATE"}

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
        assert "terminal_signal" in payload
        assert payload["terminal_signal"]["status"] == "PROCEED"
        assert payload["terminal_signal"]["reason_code"] == "VALIDATION_PASSED"

        # Payload must contain reasoning_steps (non-empty list of dicts with step numbers)
        assert "reasoning_steps" in payload
        assert isinstance(payload["reasoning_steps"], list)
        assert len(payload["reasoning_steps"]) > 0
        assert payload["reasoning_steps"][0]["step"] == 1

        # Payload must include prompt_version per ADR-39
        assert "prompt_version" in payload
        assert payload["prompt_version"].startswith("validator/content_safety@v")

        # Payload must include decision_id, agent_name, phase
        assert "decision_id" in payload
        assert payload["agent_name"] == "validator-agent"
        assert payload["phase"] == "Phase 3: Material Validation"

    async def test_terminate_audit_records_terminate_signal(self) -> None:
        """TERMINATE outcomes must also be logged with correct terminal_signal."""
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.10)
        da = StubDecisionAuditAdapter()
        service, da_ref = _build_service(mrc=mrc, decision_audit=da)
        await service.validate(_make_request())

        assert len(da_ref.entries) == 1
        payload = da_ref.entries[0].payload
        assert payload["terminal_signal"]["status"] == "TERMINATE"
        assert payload["terminal_signal"]["reason_code"] == "BLURRY_UNREADABLE"
