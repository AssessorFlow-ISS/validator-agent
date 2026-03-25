"""ValidatorService — core validation pipeline.

Orchestrates the three-step validation flow for each file:
  1. MRC blur/readability check
  2. OCR text extraction
  3. Content safety LLM reasoning

On PROCEED, extracted text is forwarded to the Knowledge Service for
chunking and embedding.  On TERMINATE (any file), the overall result is
TERMINATE and the Orchestrator halts the workflow.

Every validation decision is logged to the Decision Audit Service for the
immutable audit trail (Invariant #5).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog

from validator_agent.api.schemas import (
    FileInfo,
    FileResult,
    ValidationRequest,
    ValidationResponse,
)
from validator_agent.domain.content_safety import ContentSafetyReasoner
from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)
from validator_agent.ports.decision_audit_port import DecisionAuditPort
from validator_agent.ports.event_publisher_port import EventPublisherPort
from validator_agent.ports.knowledge_service_port import KnowledgeServicePort
from validator_agent.ports.mrc_port import MrcPort
from validator_agent.ports.ocr_port import OcrPort
from validator_agent.ports.storage_port import StoragePort

logger = structlog.get_logger(__name__)

# Minimum viable text length from OCR to consider extraction successful
_MIN_OCR_TEXT_LENGTH = 50


class ValidatorService:
    """Core validation pipeline for the Validator Agent (#12).

    This service is stateless — all state is carried by the request/response
    objects.  External dependencies are injected via port interfaces.
    """

    def __init__(
        self,
        *,
        mrc: MrcPort,
        ocr: OcrPort,
        content_safety: ContentSafetyReasoner,
        knowledge_service: KnowledgeServicePort,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        storage: StoragePort,
    ) -> None:
        self._mrc = mrc
        self._ocr = ocr
        self._content_safety = content_safety
        self._knowledge_service = knowledge_service
        self._decision_audit = decision_audit
        self._event_publisher = event_publisher
        self._storage = storage

    async def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Run the full validation pipeline for all files in the request."""
        file_results: list[FileResult] = []
        reasoning_steps_all: list[dict] = []
        step_counter = 0

        for file_info in request.files:
            signal, steps, step_counter = await self._validate_single_file(
                file_info=file_info,
                workflow_id=request.workflow_id,
                step_counter=step_counter,
            )
            file_results.append(FileResult(
                file_name=file_info.file_name,
                terminal_signal=signal,
            ))
            reasoning_steps_all.extend(steps)

        overall_signal = self._compute_overall_signal(file_results)

        await self._log_audit_decision(
            request=request,
            overall_signal=overall_signal,
            file_results=file_results,
            reasoning_steps=reasoning_steps_all,
        )

        return ValidationResponse(
            workflow_id=request.workflow_id,
            terminal_signal=overall_signal,
            file_results=file_results,
        )

    async def _validate_single_file(
        self,
        *,
        file_info: FileInfo,
        workflow_id: str,
        step_counter: int,
    ) -> tuple[TerminalSignal, list[dict], int]:
        """Validate a single file through the three-stage pipeline.

        Returns (terminal_signal, reasoning_steps, updated_step_counter).
        """
        steps: list[dict] = []

        # -- Stage 1: Download file from storage ----------------------------
        step_counter += 1
        await self._storage.download_file(file_info.storage_path)
        steps.append({
            "step": step_counter,
            "action": f"Downloaded file {file_info.file_name} from {file_info.storage_path}",
        })

        # -- Stage 2: MRC blur/readability check ----------------------------
        step_counter += 1
        mrc_result = await self._mrc.predict(file_info.storage_path)
        steps.append({
            "step": step_counter,
            "action": (
                f"MRC blur check: readiness={mrc_result.readiness}, "
                f"confidence={mrc_result.confidence}"
            ),
        })

        if not mrc_result.readiness:
            signal = TerminalSignal(
                status=TerminalSignalStatus.TERMINATE,
                reason_code=ReasonCode.BLURRY_UNREADABLE,
                message=(
                    f"Document {file_info.file_name} is blurry/unreadable "
                    f"(confidence={mrc_result.confidence})"
                ),
            )
            logger.info(
                "validation_terminated",
                file_name=file_info.file_name,
                reason="blurry_unreadable",
                confidence=mrc_result.confidence,
            )
            return signal, steps, step_counter

        # -- Stage 3: OCR text extraction ------------------------------------
        step_counter += 1
        ocr_result = await self._ocr.extract_text(file_info.storage_path)
        extracted_length = len(ocr_result.text.strip()) if ocr_result.text else 0
        steps.append({
            "step": step_counter,
            "action": (
                f"OCR extracted {extracted_length} characters "
                f"(source_type: {ocr_result.source_type})"
            ),
        })

        if extracted_length < _MIN_OCR_TEXT_LENGTH:
            signal = TerminalSignal(
                status=TerminalSignalStatus.TERMINATE,
                reason_code=ReasonCode.OCR_FAILED,
                message=(
                    f"OCR extracted insufficient text from {file_info.file_name} "
                    f"({extracted_length} chars, minimum {_MIN_OCR_TEXT_LENGTH})"
                ),
            )
            logger.info(
                "validation_terminated",
                file_name=file_info.file_name,
                reason="ocr_failed",
                extracted_length=extracted_length,
            )
            return signal, steps, step_counter

        # -- Stage 4: Content safety LLM reasoning --------------------------
        step_counter += 1
        safety_result = await self._content_safety.check(
            ocr_result.text, file_info.file_name,
        )
        steps.append({
            "step": step_counter,
            "action": (
                f"Content safety LLM: "
                f"{'no harmful content detected' if safety_result.is_safe else safety_result.message}"
            ),
        })

        if not safety_result.is_safe:
            signal = TerminalSignal(
                status=TerminalSignalStatus.TERMINATE,
                reason_code=safety_result.reason_code,
                message=safety_result.message,
            )
            logger.info(
                "validation_terminated",
                file_name=file_info.file_name,
                reason=safety_result.reason_code,
            )
            return signal, steps, step_counter

        # -- Stage 5: All checks passed — forward to Knowledge Service ------
        step_counter += 1
        await self._knowledge_service.process_material(
            workflow_id=workflow_id,
            content_text=ocr_result.text,
            source_type=ocr_result.source_type,
        )
        steps.append({
            "step": step_counter,
            "action": (
                f"Forwarded extracted text to Knowledge Service "
                f"(workflow_id={workflow_id}, source_type={ocr_result.source_type})"
            ),
        })

        signal = TerminalSignal(
            status=TerminalSignalStatus.PROCEED,
            reason_code=ReasonCode.VALIDATION_PASSED,
            message=f"Document {file_info.file_name} passed all validation checks",
        )
        logger.info(
            "validation_passed",
            file_name=file_info.file_name,
        )
        return signal, steps, step_counter

    @staticmethod
    def _compute_overall_signal(file_results: list[FileResult]) -> TerminalSignal:
        """Compute the overall Terminal Signal from per-file results.

        If ANY file has TERMINATE, the overall result is TERMINATE (using the
        first TERMINATE signal).  Otherwise PROCEED.
        """
        if len(file_results) == 1:
            return file_results[0].terminal_signal

        for result in file_results:
            if result.terminal_signal.status == TerminalSignalStatus.TERMINATE:
                return result.terminal_signal

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
        """Log the validation decision to the Decision Audit Service."""
        decision_payload = {
            "decision_id": str(uuid.uuid4()),
            "agent_name": "validator-agent",
            "decision_type": "content_validation",
            "phase": "Phase 3: Material Validation",
            "model_id": "stub",
            "confidence_score": None,
            "prompt_version": self._content_safety.prompt_version,
            "reasoning_steps": reasoning_steps,
            "terminal_signal": overall_signal.model_dump(),
            "grounding_sources": [],
            "quality_iteration_count": None,
            "convergence_stall": None,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        await self._decision_audit.log_decision(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            payload=decision_payload,
        )
