"""Tests for the ValidatorService — end-to-end validation pipeline.

Uses stub_pipeline_fn to simulate Thet's 3-component pipeline without
external calls. The stub uses keyword detection matching the old
StubModelBrokerAdapter behavior (harmful/PII/copyright detection).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.pipeline_stub import make_stub_pipeline, stub_pipeline_fn
from validator_agent.adapters.storage_stub import StubStorageAdapter
from validator_agent.api.schemas import FileInfo, ValidationRequest
from validator_agent.domain.services import ValidatorService
from validator_agent.domain.terminal_signal import ReasonCode, TerminalSignalStatus
from validator_agent.pipeline.models import ValidatorResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_request(
    *,
    files: list[FileInfo] | None = None,
    workflow_id: str = "wf-test-001",
    assessment_id: str = "assess-001",
) -> ValidationRequest:
    if files is None:
        files = [
            FileInfo(
                file_name="chapter1.pdf",
                storage_path="materials/wf-test-001/chapter1.pdf",
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
    knowledge_service: StubKnowledgeServiceAdapter | None = None,
    decision_audit: StubDecisionAuditAdapter | None = None,
    event_publisher: StubEventPublisherAdapter | None = None,
    storage: StubStorageAdapter | None = None,
    pipeline_fn: Callable | None = None,
    # Legacy parameters mapped to make_stub_pipeline
    mrc: object | None = None,
    ocr: object | None = None,
    model_broker: object | None = None,
) -> tuple[
    ValidatorService,
    StubKnowledgeServiceAdapter,
    StubDecisionAuditAdapter,
    StubStorageAdapter,
]:
    """Build a ValidatorService with configurable stub pipeline.

    Legacy mrc/ocr params are mapped to make_stub_pipeline overrides:
    - mrc with default_readiness=False → pipeline returns TERMINATE/BLURRY
    - ocr with custom default_text → pipeline uses that text for content checks
    """
    ks = knowledge_service or StubKnowledgeServiceAdapter()
    da = decision_audit or StubDecisionAuditAdapter()
    ep = event_publisher or StubEventPublisherAdapter()
    st = storage or StubStorageAdapter()

    # Build pipeline_fn from legacy params if not explicitly provided
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
        knowledge_service=ks,
        decision_audit=da,
        event_publisher=ep,
        storage=st,
        tracing=StubTracingAdapter(),
        pipeline_fn=pipeline_fn,
    )
    return service, ks, da, st


class TestValidationPipelineClean:
    """Clean document should pass all checks and return PROCEED."""

    async def test_clean_document_proceeds(self) -> None:
        service, *_ = _build_service()
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED
        assert response.terminal_signal.reason_code == ReasonCode.VALIDATION_PASSED

    async def test_clean_document_file_result(self) -> None:
        service, *_ = _build_service()
        response = await service.validate(_make_request())

        assert len(response.file_results) == 1
        assert response.file_results[0].file_name == "chapter1.pdf"
        assert response.file_results[0].terminal_signal.status == TerminalSignalStatus.PROCEED


class TestValidationPipelineMrcFailure:
    """Blurry/unreadable documents should TERMINATE at MRC stage."""

    async def test_blurry_document_terminates(self) -> None:
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.25)
        service, *_ = _build_service(mrc=mrc)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.BLURRY_UNREADABLE

    async def test_blurry_document_message_includes_confidence(self) -> None:
        mrc = StubMrcAdapter(default_readiness=False, default_confidence=0.25)
        service, *_ = _build_service(mrc=mrc)
        response = await service.validate(_make_request())

        assert "0.25" in response.terminal_signal.message

    async def test_low_confidence_but_readable_proceeds(self) -> None:
        """MRC says readable with low confidence — still proceeds (contextual reasoning)."""
        mrc = StubMrcAdapter(default_readiness=True, default_confidence=0.45)
        service, *_ = _build_service(mrc=mrc)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED


class TestValidationPipelineOcrFailure:
    """Documents with insufficient OCR text should TERMINATE."""

    async def test_short_ocr_text_terminates(self) -> None:
        ocr = StubOcrAdapter(default_text="Too short")
        service, *_ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.OCR_FAILED

    async def test_empty_ocr_text_terminates(self) -> None:
        ocr = StubOcrAdapter(default_text="")
        service, *_ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.OCR_FAILED

    async def test_ocr_exactly_at_threshold_terminates(self) -> None:
        """49 chars (below 50 threshold) should fail."""
        ocr = StubOcrAdapter(default_text="x" * 49)
        service, *_ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.OCR_FAILED

    async def test_ocr_at_threshold_proceeds(self) -> None:
        """Exactly 50 chars should pass OCR check."""
        ocr = StubOcrAdapter(default_text="x" * 50)
        service, *_ = _build_service(ocr=ocr)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED


class TestValidationPipelineContentSafety:
    """Content safety failures should TERMINATE with appropriate reason codes."""

    async def test_harmful_content_terminates(self) -> None:
        harmful = (FIXTURES_DIR / "harmful_content.txt").read_text()
        storage = StubStorageAdapter(default_bytes=harmful.encode())
        service, *_ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_pii_content_terminates(self) -> None:
        pii = (FIXTURES_DIR / "pii_content.txt").read_text()
        storage = StubStorageAdapter(default_bytes=pii.encode())
        service, *_ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.PII_DETECTED

    async def test_copyright_content_terminates(self) -> None:
        copyright_text = (
            "This textbook chapter is reproduced here. ISBN: 978-0-13-468599-1. "
            "No modifications were made to the original content. "
            "All content is subject to the publisher's copyright terms."
        )
        storage = StubStorageAdapter(default_bytes=copyright_text.encode())
        service, *_ = _build_service(storage=storage)
        response = await service.validate(_make_request())

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE
        assert response.terminal_signal.reason_code == ReasonCode.COPYRIGHT_VIOLATION


class TestValidationPipelineMultipleFiles:
    """Multi-file validation: one TERMINATE means overall TERMINATE."""

    async def test_all_files_pass(self) -> None:
        service, *_ = _build_service()
        files = [
            FileInfo(file_name="ch1.pdf", storage_path="materials/wf-test-001/ch1.pdf", file_type="pdf"),
            FileInfo(file_name="ch2.pdf", storage_path="materials/wf-test-001/ch2.pdf", file_type="pdf"),
        ]
        response = await service.validate(_make_request(files=files))

        assert response.terminal_signal.status == TerminalSignalStatus.PROCEED
        assert len(response.file_results) == 2
        assert all(r.terminal_signal.status == TerminalSignalStatus.PROCEED for r in response.file_results)

    async def test_one_file_fails_overall_terminates(self) -> None:
        harmful = (FIXTURES_DIR / "harmful_content.txt").read_text()
        storage = StubStorageAdapter(overrides={
            "materials/wf-test-001/bad.pdf": harmful.encode(),
        })
        service, *_ = _build_service(storage=storage)
        files = [
            FileInfo(file_name="good.pdf", storage_path="materials/wf-test-001/good.pdf", file_type="pdf"),
            FileInfo(file_name="bad.pdf", storage_path="materials/wf-test-001/bad.pdf", file_type="pdf"),
        ]
        response = await service.validate(_make_request(files=files))

        assert response.terminal_signal.status == TerminalSignalStatus.TERMINATE

        # One file should pass, one should fail
        statuses = {r.file_name: r.terminal_signal.status for r in response.file_results}
        assert statuses["good.pdf"] == TerminalSignalStatus.PROCEED
        assert statuses["bad.pdf"] == TerminalSignalStatus.TERMINATE


class TestKnowledgeServiceIntegration:
    """Knowledge Service should be called on PROCEED and NOT on TERMINATE."""

    async def test_knowledge_service_called_on_proceed(self) -> None:
        ks = StubKnowledgeServiceAdapter()
        service, ks_ref, _, _ = _build_service(knowledge_service=ks)
        await service.validate(_make_request())

        assert len(ks_ref.calls) == 1
        assert ks_ref.calls[0].workflow_id == "wf-test-001"
        assert ks_ref.calls[0].source_type == "ocr_extracted"

    async def test_knowledge_service_not_called_on_mrc_failure(self) -> None:
        mrc = StubMrcAdapter(default_readiness=False)
        ks = StubKnowledgeServiceAdapter()
        service, ks_ref, _, _ = _build_service(mrc=mrc, knowledge_service=ks)
        await service.validate(_make_request())

        assert len(ks_ref.calls) == 0

    async def test_knowledge_service_not_called_on_ocr_failure(self) -> None:
        ocr = StubOcrAdapter(default_text="short")
        ks = StubKnowledgeServiceAdapter()
        service, ks_ref, _, _ = _build_service(ocr=ocr, knowledge_service=ks)
        await service.validate(_make_request())

        assert len(ks_ref.calls) == 0

    async def test_knowledge_service_not_called_on_safety_failure(self) -> None:
        harmful = (FIXTURES_DIR / "harmful_content.txt").read_text()
        ks = StubKnowledgeServiceAdapter()
        storage = StubStorageAdapter(default_bytes=harmful.encode())
        service, ks_ref, _, _ = _build_service(knowledge_service=ks, storage=storage)
        await service.validate(_make_request())

        assert len(ks_ref.calls) == 0


class TestDecisionAuditLogging:
    """Every validation must log to the Decision Audit Service."""

    async def test_audit_logged_on_proceed(self) -> None:
        da = StubDecisionAuditAdapter()
        service, _, da_ref, _ = _build_service(decision_audit=da)
        await service.validate(_make_request())

        assert len(da_ref.entries) == 1
        entry = da_ref.entries[0]
        assert entry.workflow_id == "wf-test-001"
        assert entry.agent_name == "validator-agent"
        assert entry.decision_type == "content_validation"

    async def test_audit_logged_on_terminate(self) -> None:
        mrc = StubMrcAdapter(default_readiness=False)
        da = StubDecisionAuditAdapter()
        service, _, da_ref, _ = _build_service(mrc=mrc, decision_audit=da)
        await service.validate(_make_request())

        assert len(da_ref.entries) == 1
        assert da_ref.entries[0].payload["output_summary"]["terminal_signal"]["status"] == "TERMINATE"

    async def test_audit_contains_reasoning_steps(self) -> None:
        da = StubDecisionAuditAdapter()
        service, _, da_ref, _ = _build_service(decision_audit=da)
        await service.validate(_make_request())

        payload = da_ref.entries[0].payload
        assert "reasoning_steps" in payload
        assert len(payload["reasoning_steps"]) > 0
        assert "step" in payload["reasoning_steps"][0]
        assert "component" in payload["reasoning_steps"][0]
        assert "action" in payload["reasoning_steps"][0]

    async def test_audit_contains_prompt_version(self) -> None:
        da = StubDecisionAuditAdapter()
        service, _, da_ref, _ = _build_service(decision_audit=da)
        await service.validate(_make_request())

        payload = da_ref.entries[0].payload
        assert payload["prompt_version"] == "validator/thet-pipeline@v1"

    async def test_audit_contains_terminal_signal(self) -> None:
        da = StubDecisionAuditAdapter()
        service, _, da_ref, _ = _build_service(decision_audit=da)
        await service.validate(_make_request())

        payload = da_ref.entries[0].payload
        assert "output_summary" in payload
        assert payload["output_summary"]["terminal_signal"]["status"] == "PROCEED"
        assert payload["output_summary"]["terminal_signal"]["reason_code"] == "VALIDATION_PASSED"


class TestStorageDownload:
    """Storage adapter should be called to download files."""

    async def test_file_downloaded(self) -> None:
        st = StubStorageAdapter()
        service, _, _, st_ref = _build_service(storage=st)
        await service.validate(_make_request())

        assert "materials/wf-test-001/chapter1.pdf" in st_ref.downloaded
