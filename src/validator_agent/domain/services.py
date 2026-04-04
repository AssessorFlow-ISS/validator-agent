"""ValidatorService — core validation pipeline.

Bridges Thet's 3-component pipeline (validate_file) to Dale's hexagonal
infrastructure (ports for Knowledge Service, Decision Audit, Pub/Sub, Langfuse).

Thet's pipeline (validator_agent.pipeline):
  Component 1: MRC (Vertex AI EfficientNet-B0) — per-page readability
  Component 2: OCR (Document AI + GPT-4o-mini classification + GPT-4o visual understanding)
  Component 3: Content Safety (Moderation API + 4 parallel GPT-4o analyzers + synthesizer)

Dale's infrastructure layer (this file):
  - Download files via StoragePort
  - Call Thet's pipeline (sync → async bridge)
  - Map ValidatorResult → Terminal Signal
  - Forward cleaned_text to Knowledge Service
  - Log decisions to Decision Audit (dual-sink)
  - Trace to Langfuse via TracingPort
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import structlog

from validator_agent.api.schemas import (
    FileInfo,
    FileResult,
    ValidationRequest,
    ValidationResponse,
)
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)
from validator_agent.ports.decision_audit_port import DecisionAuditPort
from validator_agent.ports.event_publisher_port import EventPublisherPort
from validator_agent.ports.knowledge_service_port import KnowledgeServicePort
from validator_agent.ports.storage_port import StoragePort

# Shared ports and models from af-shared (B3 unification)
from af_shared.models.domain import DecisionLogEntry
from af_shared.ports.tracing import TracingPort

logger = structlog.get_logger(__name__)

# Map Thet's termination_reason strings to Dale's ReasonCode enum
_REASON_CODE_MAP: dict[str | None, ReasonCode] = {
    "BLURRY_UNREADABLE": ReasonCode.BLURRY_UNREADABLE,
    "OCR_FAILED": ReasonCode.OCR_FAILED,
    "HARMFUL_IMAGE": ReasonCode.HARMFUL_IMAGE,
    "HARMFUL_CONTENT": ReasonCode.HARMFUL_CONTENT,
    "RELIGIOUS_POLITICAL_VIOLATION": ReasonCode.RELIGIOUS_POLITICAL_VIOLATION,
    "CONTENT_SAFETY_UNAVAILABLE": ReasonCode.CONTENT_SAFETY_UNAVAILABLE,
    "PII_DETECTED": ReasonCode.PII_DETECTED,
    "COPYRIGHT_VIOLATION": ReasonCode.COPYRIGHT_VIOLATION,
    "CONTENT_POLICY_VIOLATION": ReasonCode.CONTENT_POLICY_VIOLATION,
    "VALIDATION_PASSED": ReasonCode.VALIDATION_PASSED,
    None: ReasonCode.VALIDATION_PASSED,
}

# Map Thet's overall_status strings to Dale's TerminalSignalStatus
_STATUS_MAP: dict[str, TerminalSignalStatus] = {
    "PROCEED": TerminalSignalStatus.PROCEED,
    "PROCEED_WITH_WARNINGS": TerminalSignalStatus.PROCEED_WITH_WARNINGS,
    "PROCEED_WITH_EXCLUSIONS": TerminalSignalStatus.PROCEED_WITH_WARNINGS,
    "TERMINATE": TerminalSignalStatus.TERMINATE,
}


class ValidatorService:
    """Core validation pipeline for the Validator Agent (#12).

    Wraps Thet's 3-component pipeline with Dale's infrastructure layer.
    This service is stateless — all state is carried by the request/response
    objects. External dependencies are injected via port interfaces.

    The pipeline_fn parameter allows injecting a stub pipeline for testing
    (returns canned ValidatorResult) or using Thet's real pipeline in production.
    """

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeServicePort,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        storage: StoragePort,
        tracing: TracingPort,
        pipeline_fn: Callable | None = None,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._decision_audit = decision_audit
        self._event_publisher = event_publisher
        self._storage = storage
        self._tracing = tracing
        self._pipeline_fn = pipeline_fn

    async def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Run the full validation pipeline for all files in the request."""
        file_results: list[FileResult] = []
        all_cleaned_text: list[str] = []
        all_warnings: list[dict] = []
        reasoning_steps: list[dict] = []

        for file_info in request.files:
            result = await self._validate_single_file(
                file_info=file_info,
                workflow_id=request.workflow_id,
            )
            file_results.append(result)

            if result.cleaned_text:
                all_cleaned_text.append(result.cleaned_text)
            all_warnings.extend(result.assessor_warnings)

            # Build reasoning steps from pipeline result
            reasoning_steps.append({
                "file": file_info.file_name,
                "status": result.terminal_signal.status.value,
                "reason_code": result.terminal_signal.reason_code.value,
                "terminated_at": result.terminated_at_component,
                "time_ms": result.total_time_ms,
            })

        overall_signal = self._compute_overall_signal(file_results)

        # Forward cleaned text to Knowledge Service on PROCEED
        if overall_signal.status != TerminalSignalStatus.TERMINATE:
            combined_text = "\n\n".join(all_cleaned_text)
            if combined_text.strip():
                await self._knowledge_service.process_material(
                    workflow_id=request.workflow_id,
                    content_text=combined_text,
                    source_type="ocr_extracted",
                )

        # Log audit decision (dual-sink)
        await self._log_audit_decision(
            request=request,
            overall_signal=overall_signal,
            file_results=file_results,
            reasoning_steps=reasoning_steps,
        )

        return ValidationResponse(
            workflow_id=request.workflow_id,
            terminal_signal=overall_signal,
            file_results=file_results,
            cleaned_text="\n\n".join(all_cleaned_text),
            assessor_warnings=all_warnings,
        )

    async def _validate_single_file(
        self,
        *,
        file_info: FileInfo,
        workflow_id: str,
    ) -> FileResult:
        """Validate a single file through Thet's 3-component pipeline."""
        # Download file bytes from storage
        file_bytes = await self._storage.download_file(file_info.storage_path)

        # Run pipeline (Thet's real pipeline or injected stub)
        if self._pipeline_fn is not None:
            fn = self._pipeline_fn
        else:
            from validator_agent.pipeline import validate_file
            fn = validate_file

        pipeline_result = await asyncio.to_thread(
            fn, file_bytes, file_info.file_name,
        )

        # Map Thet's result to Dale's Terminal Signal
        status = _STATUS_MAP.get(
            pipeline_result.overall_status,
            TerminalSignalStatus.TERMINATE,
        )
        reason_code = _REASON_CODE_MAP.get(
            pipeline_result.termination_reason,
            ReasonCode.VALIDATION_PASSED if status != TerminalSignalStatus.TERMINATE
            else ReasonCode.CONTENT_POLICY_VIOLATION,
        )
        message = (
            pipeline_result.termination_detail
            or f"Document {file_info.file_name} validated: {pipeline_result.overall_status}"
        )

        signal = TerminalSignal(
            status=status,
            reason_code=reason_code,
            message=message[:500],
        )

        # Trace the pipeline result
        await self._tracing.trace_tool_call(
            workflow_id=workflow_id,
            agent_name="validator-agent",
            tool_name="thet-pipeline",
            input_params={
                "file_name": file_info.file_name,
                "file_type": file_info.file_type,
            },
            output_summary={
                "overall_status": pipeline_result.overall_status,
                "termination_reason": pipeline_result.termination_reason,
                "terminated_at_component": pipeline_result.terminated_at_component,
                "total_time_ms": pipeline_result.total_time_ms,
                "mrc_status": pipeline_result.mrc.overall_status if pipeline_result.mrc else None,
                "ocr_word_count": pipeline_result.ocr.total_word_count if pipeline_result.ocr else 0,
                "safety_status": pipeline_result.content_safety.overall_status if pipeline_result.content_safety else None,
            },
            latency_ms=pipeline_result.total_time_ms,
        )

        logger.info(
            "validation_complete",
            file_name=file_info.file_name,
            status=pipeline_result.overall_status,
            reason=pipeline_result.termination_reason,
            time_ms=pipeline_result.total_time_ms,
        )

        return FileResult(
            file_name=file_info.file_name,
            terminal_signal=signal,
            cleaned_text=pipeline_result.cleaned_text,
            assessor_warnings=pipeline_result.assessor_warnings,
            total_time_ms=pipeline_result.total_time_ms,
            terminated_at_component=pipeline_result.terminated_at_component,
        )

    @staticmethod
    def _compute_overall_signal(file_results: list[FileResult]) -> TerminalSignal:
        """Compute the overall Terminal Signal from per-file results.

        If ANY file has TERMINATE, the overall result is TERMINATE (first one).
        If any file has PROCEED_WITH_WARNINGS, overall is PROCEED_WITH_WARNINGS.
        Otherwise PROCEED.
        """
        if len(file_results) == 1:
            return file_results[0].terminal_signal

        for result in file_results:
            if result.terminal_signal.status == TerminalSignalStatus.TERMINATE:
                return result.terminal_signal

        has_warnings = any(
            result.terminal_signal.status == TerminalSignalStatus.PROCEED_WITH_WARNINGS
            for result in file_results
        )
        if has_warnings:
            return TerminalSignal(
                status=TerminalSignalStatus.PROCEED_WITH_WARNINGS,
                reason_code=ReasonCode.VALIDATION_PASSED,
                message="All files validated with warnings",
            )

        return TerminalSignal(
            status=TerminalSignalStatus.PROCEED,
            reason_code=ReasonCode.VALIDATION_PASSED,
            message="All files validated successfully",
        )

    async def _log_audit_decision(
        self,
        *,
        request: ValidationRequest,
        overall_signal: TerminalSignal,
        file_results: list[FileResult],
        reasoning_steps: list[dict],
    ) -> None:
        """Log the validation decision to both sinks using ONE canonical format."""
        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            input_summary={
                "file_count": len(request.files),
                "files": [f.file_name for f in request.files],
                "phase": "Phase 3: Material Validation",
                "tools_used": [
                    "mrc-vertex-ai", "documentai-ocr", "page-classifier",
                    "visual-understanding", "moderation-api",
                    "content-analyzers-x4", "content-synthesizer",
                ],
            },
            output_summary={
                "terminal_signal": overall_signal.model_dump(),
                "files_validated": len(file_results),
                "files_passed": sum(
                    1 for f in file_results
                    if f.terminal_signal.status != TerminalSignalStatus.TERMINATE
                ),
                "assessor_warnings_count": sum(
                    len(f.assessor_warnings) for f in file_results
                ),
            },
            reasoning_steps=reasoning_steps,
            confidence_score=None,
            prompt_version="validator/thet-pipeline@v1",
            model_id="gpt-4o",
            grounding_sources=[f.file_name for f in file_results],
        )

        # Sink 1: PostgreSQL (via Pub/Sub → Decision Audit Service)
        await self._decision_audit.log_decision(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            payload=entry.model_dump(),
        )

        # Sink 2: Langfuse (via TracingPort → OTel SDK)
        await self._tracing.trace_decision(entry)
