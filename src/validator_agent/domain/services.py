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
import os
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
        # Keep raw pipeline results for reasoning_steps extraction
        pipeline_results: list[tuple[FileInfo, object]] = []

        for file_info in request.files:
            result, raw_pipeline = await self._validate_single_file(
                file_info=file_info,
                workflow_id=request.workflow_id,
            )
            file_results.append(result)
            pipeline_results.append((file_info, raw_pipeline))

            if result.cleaned_text:
                all_cleaned_text.append(result.cleaned_text)
            all_warnings.extend(result.assessor_warnings)

        # Build detailed per-component reasoning steps
        for fi, raw in pipeline_results:
            reasoning_steps.extend(self._build_reasoning_steps(fi, raw))

        overall_signal = self._compute_overall_signal(file_results)

        # Forward cleaned text to Knowledge Service on PROCEED
        chunk_ids: list[str] = []
        if overall_signal.status != TerminalSignalStatus.TERMINATE:
            combined_text = "\n\n".join(all_cleaned_text)
            if combined_text.strip():
                chunk_ids = await self._knowledge_service.process_material(
                    workflow_id=request.workflow_id,
                    content_text=combined_text,
                    source_type="ocr_extracted",
                )

        # Log audit decision (dual-sink) — grounding_sources are real chunk IDs
        await self._log_audit_decision(
            request=request,
            overall_signal=overall_signal,
            file_results=file_results,
            reasoning_steps=reasoning_steps,
            grounding_chunk_ids=chunk_ids,
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
    ) -> tuple[FileResult, object]:
        """Validate a single file through Thet's 3-component pipeline.

        Returns (FileResult, raw_pipeline_result) — the raw result is
        needed for per-component reasoning_steps extraction.
        """
        # Download file bytes from storage
        file_bytes = await self._storage.download_file(file_info.storage_path)

        # Set workflow_id for Model Broker session tracking (token budgets)
        os.environ["CURRENT_WORKFLOW_ID"] = workflow_id

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

        # Emit per-component traces to Langfuse (Sink 2)
        await self._emit_component_traces(workflow_id, file_info, pipeline_result)

        logger.info(
            "validation_complete",
            file_name=file_info.file_name,
            status=pipeline_result.overall_status,
            reason=pipeline_result.termination_reason,
            time_ms=pipeline_result.total_time_ms,
        )

        file_result = FileResult(
            file_name=file_info.file_name,
            terminal_signal=signal,
            cleaned_text=pipeline_result.cleaned_text,
            assessor_warnings=pipeline_result.assessor_warnings,
            total_time_ms=pipeline_result.total_time_ms,
            terminated_at_component=pipeline_result.terminated_at_component,
        )
        return file_result, pipeline_result

    async def _emit_component_traces(
        self,
        workflow_id: str,
        file_info: FileInfo,
        result: object,
    ) -> None:
        """Emit per-component Langfuse traces from ValidatorResult.

        Each component gets its own TOOL span in Langfuse, providing
        step-by-step visibility in the Agent Trace page.
        """
        mrc = getattr(result, "mrc", None)
        ocr = getattr(result, "ocr", None)
        safety = getattr(result, "content_safety", None)

        # 1. MRC readability check
        if mrc is not None:
            await self._tracing.trace_tool_call(
                workflow_id=workflow_id,
                agent_name="validator-agent",
                tool_name="mrc-vertex-ai",
                input_params={"file_name": file_info.file_name},
                output_summary={
                    "overall_status": mrc.overall_status,
                    "total_pages": mrc.total_pages,
                    "readable_pages": mrc.readable_pages,
                    "blurry_ratio": mrc.blurry_ratio,
                    "excluded_pages": mrc.excluded_pages,
                    "overall_confidence": mrc.overall_confidence,
                },
                latency_ms=mrc.inference_time_ms,
            )

        # 2. Document AI OCR
        if ocr is not None:
            text_pages = sum(1 for p in ocr.pages if p.classification == "TEXT")
            visual_pages = sum(1 for p in ocr.pages if p.classification == "VISUAL")

            await self._tracing.trace_tool_call(
                workflow_id=workflow_id,
                agent_name="validator-agent",
                tool_name="documentai-ocr",
                input_params={"file_name": file_info.file_name, "readable_pages": mrc.readable_page_numbers if mrc else []},
                output_summary={
                    "total_pages": ocr.total_pages,
                    "total_word_count": ocr.total_word_count,
                    "processing_mode": ocr.processing_mode,
                    "text_pages": text_pages,
                    "visual_pages": visual_pages,
                    "visual_pages_processed": ocr.visual_pages_processed,
                    "harmful_image_detected": ocr.harmful_image_detected,
                },
                latency_ms=ocr.ocr_time_ms,
            )

            # 3. Page classification detail
            if ocr.pages:
                page_details = [
                    {"page": p.page_number, "classification": p.classification, "source": p.source, "words": p.word_count}
                    for p in ocr.pages
                ]
                await self._tracing.trace_tool_call(
                    workflow_id=workflow_id,
                    agent_name="validator-agent",
                    tool_name="page-classifier",
                    input_params={"page_count": len(ocr.pages)},
                    output_summary={"pages": page_details, "text_count": text_pages, "visual_count": visual_pages},
                    latency_ms=0,
                )

            # 4. Image moderation (if harmful image detected)
            if ocr.harmful_image_detected:
                await self._tracing.trace_tool_call(
                    workflow_id=workflow_id,
                    agent_name="validator-agent",
                    tool_name="openai-image-moderation",
                    input_params={"page": ocr.harmful_image_page},
                    output_summary={"flagged": True, "detail": ocr.harmful_image_detail},
                    latency_ms=0,
                )

        # 5. Content safety pipeline
        if safety is not None:
            findings_summary = {
                "harmful": len(safety.harmful_findings),
                "pii": len(safety.pii_findings),
                "religious_political": len(safety.religious_political_findings),
                "copyright": len(safety.copyright_findings),
                "misinformation": len(safety.misinformation_findings),
            }
            await self._tracing.trace_tool_call(
                workflow_id=workflow_id,
                agent_name="validator-agent",
                tool_name="content-safety-pipeline",
                input_params={"page_count": ocr.total_pages if ocr else 0},
                output_summary={
                    "overall_status": safety.overall_status,
                    "findings": findings_summary,
                    "total_findings": sum(findings_summary.values()),
                    "pii_redacted": len(safety.pii_findings),
                    "assessor_warnings": len(safety.assessor_warnings),
                    "cleaned_text_length": len(safety.cleaned_text),
                },
                latency_ms=0,
            )

    @staticmethod
    def _build_reasoning_steps(file_info: FileInfo, result: object) -> list[dict]:
        """Extract per-component reasoning steps from ValidatorResult.

        These populate the reasoning chain panel in the Agent Trace page.
        """
        steps: list[dict] = []
        step_num = 0

        mrc = getattr(result, "mrc", None)
        ocr = getattr(result, "ocr", None)
        safety = getattr(result, "content_safety", None)
        terminated_at = getattr(result, "terminated_at_component", None)

        # Step 1: MRC
        if mrc is not None:
            step_num += 1
            excluded = f", excluded pages: {mrc.excluded_pages}" if mrc.excluded_pages else ""
            steps.append({
                "step": step_num,
                "component": "mrc",
                "action": (
                    f"MRC readability check: {mrc.readable_pages}/{mrc.total_pages} pages readable, "
                    f"confidence {mrc.overall_confidence:.3f}, blurry_ratio {mrc.blurry_ratio:.0%}{excluded}"
                ),
                "status": mrc.overall_status,
                "latency_ms": mrc.inference_time_ms,
            })
            if terminated_at == "mrc":
                return steps

        # Step 2: OCR
        if ocr is not None:
            step_num += 1
            steps.append({
                "step": step_num,
                "component": "ocr",
                "action": (
                    f"Document AI OCR: {ocr.total_word_count} words from {ocr.total_pages} pages "
                    f"(mode: {ocr.processing_mode or 'unknown'})"
                ),
                "status": ocr.overall_status,
                "latency_ms": ocr.ocr_time_ms,
            })

            # Step 3: Page classification
            if ocr.pages:
                text_count = sum(1 for p in ocr.pages if p.classification == "TEXT")
                visual_count = sum(1 for p in ocr.pages if p.classification == "VISUAL")
                step_num += 1
                steps.append({
                    "step": step_num,
                    "component": "page_classification",
                    "action": f"Page classification: {text_count} TEXT, {visual_count} VISUAL",
                    "pages": [
                        {"page": p.page_number, "class": p.classification, "source": p.source}
                        for p in ocr.pages
                    ],
                })

            # Step 3b: Image moderation (if harmful)
            if ocr.harmful_image_detected:
                step_num += 1
                steps.append({
                    "step": step_num,
                    "component": "image_moderation",
                    "action": f"Image moderation: FLAGGED on page {ocr.harmful_image_page}. {ocr.harmful_image_detail or ''}",
                    "status": "TERMINATE",
                })
                return steps

            # Step 3c: Visual understanding (if visual pages processed)
            if ocr.visual_pages_processed and ocr.visual_pages_processed > 0:
                step_num += 1
                visual_details = [
                    f"page {p.page_number}: {p.source} (attempts: {p.visual_attempts})"
                    for p in ocr.pages if p.classification == "VISUAL"
                ]
                steps.append({
                    "step": step_num,
                    "component": "visual_understanding",
                    "action": f"Visual understanding: {ocr.visual_pages_processed} pages enhanced. {', '.join(visual_details)}",
                })

            if terminated_at == "ocr":
                return steps

        # Step 4: Text moderation pre-filter
        if safety is not None:
            step_num += 1
            if safety.overall_status == "TERMINATE" and safety.termination_reason == "HARMFUL_CONTENT" and not safety.harmful_findings:
                steps.append({
                    "step": step_num,
                    "component": "text_moderation",
                    "action": f"OpenAI Moderation pre-filter: FLAGGED ({safety.termination_detail or 'harmful content'})",
                    "status": "TERMINATE",
                })
                return steps
            steps.append({
                "step": step_num,
                "component": "text_moderation",
                "action": "OpenAI Moderation pre-filter: PASSED (free)",
                "status": "PASSED",
            })

            # Step 5: Content analyzers
            findings = {
                "A": len(safety.harmful_findings),
                "B": len(safety.misinformation_findings),
                "C": len(safety.pii_findings) + len(safety.copyright_findings),
                "D": len(safety.religious_political_findings),
            }
            step_num += 1
            steps.append({
                "step": step_num,
                "component": "content_analyzers",
                "action": (
                    f"4 parallel analyzers: "
                    f"A({findings['A']} harmful) B({findings['B']} misinfo) "
                    f"C({findings['C']} PII/copyright) D({findings['D']} religious/political)"
                ),
                "findings_total": sum(findings.values()),
            })

            # Step 6: Synthesizer (inferred from findings presence)
            total = sum(findings.values())
            step_num += 1
            steps.append({
                "step": step_num,
                "component": "synthesizer",
                "action": f"Synthesizer: {total} finding(s) confirmed after voting",
            })

            # Step 7: Content safety decision
            hard_gates = []
            if safety.harmful_detected:
                hard_gates.append("harmful")
            if safety.religious_political_detected:
                hard_gates.append("religious/political")

            soft_items = []
            if safety.pii_detected:
                soft_items.append(f"{len(safety.pii_findings)} PII redacted")
            if safety.copyright_detected:
                soft_items.append(f"{len(safety.copyright_findings)} copyright warned")
            if safety.misinformation_detected:
                soft_items.append(f"{len(safety.misinformation_findings)} misinformation warned")

            step_num += 1
            if hard_gates:
                steps.append({
                    "step": step_num,
                    "component": "safety_decision",
                    "action": f"Content safety decision: TERMINATE. Hard gate(s): {', '.join(hard_gates)}",
                    "status": "TERMINATE",
                })
                return steps

            # TERMINATE without hard gates = analyzers unavailable
            if safety.overall_status == "TERMINATE":
                reason = safety.termination_reason or safety.error_message or "Content safety unavailable"
                steps.append({
                    "step": step_num,
                    "component": "safety_decision",
                    "action": f"Content safety decision: TERMINATE. {reason}",
                    "status": "TERMINATE",
                })
                return steps

            soft_desc = ". ".join(soft_items) if soft_items else "No issues detected"
            steps.append({
                "step": step_num,
                "component": "safety_decision",
                "action": f"Content safety decision: {safety.overall_status}. {soft_desc}",
                "status": safety.overall_status,
            })

        # Step 8: Knowledge Service write (on PROCEED)
        cleaned_len = len(getattr(result, "cleaned_text", "") or "")
        if cleaned_len > 0 and getattr(result, "overall_status", "") != "TERMINATE":
            step_num += 1
            steps.append({
                "step": step_num,
                "component": "knowledge_service",
                "action": f"Forwarded {cleaned_len} chars (PII-redacted) to Knowledge Service for chunking and embedding",
            })

        return steps

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
        grounding_chunk_ids: list[str] | None = None,
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
            grounding_sources=grounding_chunk_ids or [f.file_name for f in file_results],
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
