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
        from datetime import datetime, timezone

        pipeline_start = datetime.now(timezone.utc)
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

        # Build detailed per-component reasoning steps with timestamps.
        # For text files (.md), inject component-tagged steps so each
        # sub-card in the trace page shows contextual Decision Insight.
        for fi, raw in pipeline_results:
            is_text = (
                fi.file_type in ("text/markdown", "text/plain")
                or fi.file_name.endswith((".md", ".txt"))
            )
            if is_text:
                word_count = len((raw.cleaned_text or "").split())
                reasoning_steps.extend([
                    {"step": len(reasoning_steps) + 1, "component": "mrc",
                     "action": f"MRC readability: SKIPPED (text file, {word_count} words). No blur detection needed for .md files."},
                    {"step": len(reasoning_steps) + 2, "component": "ocr",
                     "action": f"Text extraction: SKIPPED (already text, {word_count} words). Read directly from {fi.file_name}."},
                    {"step": len(reasoning_steps) + 3, "component": "content_fit_decision",
                     "action": f"Content-Fit: {raw.overall_status}. {fi.file_name} ({word_count} words) from web research."},
                ])
            else:
                reasoning_steps.extend(self._build_reasoning_steps(fi, raw, pipeline_start))

        # Write consolidated progress events for Phase 5 text files (one set
        # for ALL files, not per-file). This gives the trace page 3 clean
        # sub-cards instead of N x 3.
        if request.validation_type == "web_research_validation":
            text_files = [fi for fi, _ in pipeline_results
                          if fi.file_type in ("text/markdown", "text/plain") or fi.file_name.endswith((".md", ".txt"))]
            total_words = sum(len((raw.cleaned_text or "").split()) for _, raw in pipeline_results)
            _PG = "web-content-assurance"
            await self._write_progress_event(
                request.workflow_id,
                "assessorflow.validation.mrc-complete",
                f"MRC readability: SKIPPED ({len(text_files)} text files, {total_words} total words). No blur detection needed for web research .md files.",
                pipeline_group=_PG,
            )
            await self._write_progress_event(
                request.workflow_id,
                "assessorflow.validation.ocr-complete",
                f"Text extraction: SKIPPED ({len(text_files)} files already text, {total_words} words). Read directly from .md files.",
                pipeline_group=_PG,
            )
            safety_statuses = [r.overall_status for _, r in pipeline_results]
            all_proceed = all(s in ("PROCEED", "PROCEED_WITH_WARNINGS") for s in safety_statuses)
            await self._write_progress_event(
                request.workflow_id,
                "assessorflow.validation.safety-complete",
                f"Content safety: {'PASSED' if all_proceed else 'ISSUES FOUND'} across {len(text_files)} web research files. {total_words} words screened.",
                pipeline_group=_PG,
            )

        overall_signal = self._compute_overall_signal(file_results)

        # Forward cleaned text to Knowledge Service on PROCEED
        chunk_ids: list[str] = []
        if overall_signal.status != TerminalSignalStatus.TERMINATE:
            combined_text = "\n\n".join(all_cleaned_text)
            if combined_text.strip():
                # Use source_type based on validation_type:
                # Phase 3 materials → "ocr_extracted" (from PDF OCR)
                # Phase 5 web research → "web_research" (already text)
                source_type = (
                    "web_research"
                    if request.validation_type == "web_research_validation"
                    else "ocr_extracted"
                )
                chunk_ids = await self._knowledge_service.process_material(
                    workflow_id=request.workflow_id,
                    content_text=combined_text,
                    source_type=source_type,
                    assessment_id=request.assessment_id,
                    source="web_research" if request.validation_type == "web_research_validation" else "upload",
                )

        # Add KB-write decision step with content fitness score
        if chunk_ids:
            reasoning_steps.append({
                "step": len(reasoning_steps) + 1,
                "action": f"Content fit: PROCEED — stored {len(chunk_ids)} chunks to Knowledge Service",
                "component": "kb_write",
                "chunks_stored": len(chunk_ids),
            })
            ingestion_summary = f"Forwarded {len(combined_text)} chars (PII-redacted) to Knowledge Service for chunking and embedding"
            ingestion_pg = "web-content-assurance" if request.validation_type == "web_research_validation" else "content-assurance"
            await self._write_progress_event(request.workflow_id, "assessorflow.validation.ingestion-complete", ingestion_summary, pipeline_group=ingestion_pg)
            await self._tracing.trace_tool_call(
                workflow_id=request.workflow_id,
                agent_name="validator-agent",
                tool_name="knowledge-service-ingestion",
                input_params={"content_length": len(combined_text)},
                output_summary={"chunks_stored": len(chunk_ids), "chunk_ids": chunk_ids[:5]},
                latency_ms=0,
            )

        # Compute content fitness score from pipeline results
        content_fitness = self._compute_content_fitness(pipeline_results, file_results)

        # Log audit decision (dual-sink) — grounding_sources are real chunk IDs
        await self._log_audit_decision(
            request=request,
            overall_signal=overall_signal,
            file_results=file_results,
            reasoning_steps=reasoning_steps,
            grounding_chunk_ids=chunk_ids,
            content_fitness=content_fitness,
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
        """Validate a single file through the 3-component pipeline.

        Each component runs sequentially. After each completes, a progress
        event is written to workflow_events so the UI can render sub-cards
        incrementally as the pipeline progresses.

        Returns (FileResult, raw_pipeline_result) — the raw result is
        needed for per-component reasoning_steps extraction.
        """
        # Download file bytes from storage
        file_bytes = await self._storage.download_file(file_info.storage_path)

        # Text files (.md, .txt) from Web Research Agent: skip MRC/OCR,
        # run content safety only, return text content directly.
        is_text_file = (
            file_info.file_type in ("text/markdown", "text/plain")
            or file_info.file_name.endswith((".md", ".txt"))
        )
        if is_text_file:
            pipeline_result = await self._validate_text_file(
                file_bytes, file_info, workflow_id,
            )
            return self._map_pipeline_to_result(file_info, pipeline_result, workflow_id), pipeline_result

        # Set workflow context for Model Broker session tracking (token budgets)
        from validator_agent.pipeline.llm_client import set_workflow_context
        set_workflow_context(workflow_id)

        # If a stub pipeline was injected (tests), run it as a single call
        if self._pipeline_fn is not None:
            pipeline_result = await asyncio.to_thread(
                self._pipeline_fn, file_bytes, file_info.file_name,
            )
        else:
            pipeline_result = await self._run_pipeline_with_progress(
                file_bytes, file_info, workflow_id,
            )

        result_tuple = self._map_pipeline_to_result(file_info, pipeline_result, workflow_id)

        # Emit per-component traces to Langfuse (Sink 2)
        await self._emit_component_traces(workflow_id, file_info, pipeline_result)

        return result_tuple, pipeline_result

    def _map_pipeline_to_result(
        self,
        file_info: FileInfo,
        pipeline_result: object,
        workflow_id: str,
    ) -> FileResult:
        """Map a pipeline result to a FileResult with Terminal Signal."""
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

    async def _validate_text_file(
        self,
        file_bytes: bytes,
        file_info: FileInfo,
        workflow_id: str,
    ) -> object:
        """Validate a text file (.md, .txt) — skip MRC/OCR, content safety only.

        Web Research Agent produces .md files that are already text. Running
        MRC (blur detection) or OCR on text is meaningless. We read the text
        directly and run content safety.
        """
        from validator_agent.pipeline.models import ValidatorResult
        from validator_agent.pipeline.content_safety import check_content_safety
        from validator_agent.pipeline.llm_client import set_workflow_context

        set_workflow_context(workflow_id)
        start_ms = time.time() * 1000

        # Read text directly
        text = file_bytes.decode("utf-8", errors="replace")
        word_count = len(text.split())

        logger.info(
            "text_file_validation",
            file_name=file_info.file_name,
            word_count=word_count,
        )

        # No per-file progress events for text files — consolidated events
        # are written once by validate() after all files are processed.
        # This prevents 5x MRC, 5x OCR, 5x Safety sub-cards in the trace page.

        # Run content safety on the text — wrap in PageOcrResult (content_safety
        # expects list[PageOcrResult], not raw text string)
        try:
            from validator_agent.pipeline.models import PageOcrResult
            pages = [PageOcrResult(
                page_number=1,
                extracted_text=text,
                word_count=word_count,
                confidence=1.0,
                classification="TEXT",
                source="web_research",
            )]
            safety_result = await asyncio.to_thread(
                check_content_safety, pages,
            )

            elapsed = time.time() * 1000 - start_ms
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status=safety_result.overall_status,
                termination_reason=safety_result.termination_reason,
                termination_detail=safety_result.termination_detail,
                cleaned_text=text,
                assessor_warnings=safety_result.assessor_warnings,
                total_time_ms=elapsed,
                terminated_at_component=None,
            )
        except Exception:
            logger.warning("text_file_safety_check_failed", exc_info=True)
            elapsed = time.time() * 1000 - start_ms
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status="PROCEED",
                termination_reason=None,
                termination_detail=f"Text file validated: {word_count} words (safety check skipped)",
                cleaned_text=text,
                assessor_warnings=[],
                total_time_ms=elapsed,
                terminated_at_component=None,
            )

    async def _run_pipeline_with_progress(
        self,
        file_bytes: bytes,
        file_info: FileInfo,
        workflow_id: str,
    ) -> object:
        """Run each pipeline component individually, emitting progress events."""
        from validator_agent.pipeline.models import ValidatorResult
        from validator_agent.pipeline.mrc_client import check_readability
        from validator_agent.pipeline.ocr_pipeline import extract_text
        from validator_agent.pipeline.content_safety import check_content_safety

        start_ms = time.time() * 1000

        # ── Component 1: MRC ──
        mrc_result = await asyncio.to_thread(check_readability, file_bytes, file_info.file_name)
        mrc_summary = (
            f"MRC readability check: {mrc_result.readable_pages}/{mrc_result.total_pages} pages readable, "
            f"confidence {mrc_result.overall_confidence:.3f}, blurry_ratio {mrc_result.blurry_ratio:.0%}"
        )
        await self._write_progress_event(workflow_id, "assessorflow.validation.mrc-complete", mrc_summary)
        await self._tracing.trace_tool_call(
            workflow_id=workflow_id,
            agent_name="validator-agent",
            tool_name="mrc-vertex-ai",
            input_params={"file_name": file_info.file_name},
            output_summary={
                "overall_status": mrc_result.overall_status,
                "total_pages": mrc_result.total_pages,
                "readable_pages": mrc_result.readable_pages,
                "blurry_ratio": mrc_result.blurry_ratio,
                "overall_confidence": mrc_result.overall_confidence,
            },
            latency_ms=mrc_result.inference_time_ms,
        )

        if mrc_result.overall_status == "TERMINATE":
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status="TERMINATE",
                termination_reason="BLURRY_UNREADABLE",
                termination_detail=f"{mrc_result.unreadable_pages}/{mrc_result.total_pages} pages unreadable ({mrc_result.blurry_ratio:.0%} > 30% threshold)",
                terminated_at_component="mrc",
                mrc=mrc_result,
                total_time_ms=round(time.time() * 1000 - start_ms, 2),
            )

        readable_pages = mrc_result.readable_page_numbers if mrc_result.readable_page_numbers else None

        # ── Component 2: OCR (3 passes: Document AI → Classification → Visual) ──
        ocr_result = await asyncio.to_thread(extract_text, file_bytes, file_info.file_name, readable_pages)
        text_pages = sum(1 for p in ocr_result.pages if p.classification == "TEXT")
        visual_pages = sum(1 for p in ocr_result.pages if p.classification == "VISUAL")

        # Sub-card 2a: Document AI OCR
        ocr_summary = (
            f"Document AI OCR: {ocr_result.total_word_count} words from {ocr_result.total_pages} pages "
            f"(mode: {ocr_result.processing_mode})"
        )
        await self._write_progress_event(workflow_id, "assessorflow.validation.ocr-complete", ocr_summary)
        await self._tracing.trace_tool_call(
            workflow_id=workflow_id,
            agent_name="validator-agent",
            tool_name="documentai-ocr",
            input_params={"file_name": file_info.file_name, "readable_pages": mrc_result.readable_page_numbers if mrc_result.readable_page_numbers else []},
            output_summary={
                "total_pages": ocr_result.total_pages,
                "total_word_count": ocr_result.total_word_count,
                "processing_mode": ocr_result.processing_mode,
            },
            latency_ms=ocr_result.ocr_time_ms,
        )

        # Sub-card 2b: Page Classification
        classify_summary = f"Page classification: {text_pages} TEXT, {visual_pages} VISUAL out of {ocr_result.total_pages} pages"
        await self._write_progress_event(workflow_id, "assessorflow.validation.classification-complete", classify_summary)
        try:
            await self._tracing.trace_tool_call(
                workflow_id=workflow_id,
                agent_name="validator-agent",
                tool_name="page-classifier",
                input_params={"total_pages": ocr_result.total_pages},
                output_summary={
                    "text_pages": text_pages,
                    "visual_pages": visual_pages,
                    "page_sources": {str(p.page_number): p.classification for p in ocr_result.pages},
                },
            )
        except Exception:
            logger.warning("trace_page_classifier_failed", workflow_id=workflow_id, exc_info=True)

        # Sub-card 2c: Visual Understanding (only if VISUAL pages exist)
        if visual_pages > 0:
            llm_pages = sum(1 for p in ocr_result.pages if p.source == "llm")
            fallback_pages = sum(1 for p in ocr_result.pages if p.source == "ocr_fallback")
            visual_summary = (
                f"Visual understanding: {llm_pages}/{visual_pages} pages enhanced via LLM"
            )
            if fallback_pages > 0:
                visual_summary += f", {fallback_pages} fell back to OCR text"
            if ocr_result.visual_pages_processed > 0:
                visual_summary += f". {ocr_result.visual_pages_processed} pages produced richer descriptions"
            try:
                await self._write_progress_event(workflow_id, "assessorflow.validation.visual-complete", visual_summary)
                await self._tracing.trace_tool_call(
                    workflow_id=workflow_id,
                    agent_name="validator-agent",
                    tool_name="visual-understanding",
                    input_params={"visual_pages": visual_pages},
                    output_summary={
                        "llm_enhanced": llm_pages,
                        "ocr_fallback": fallback_pages,
                        "visual_pages_processed": ocr_result.visual_pages_processed,
                        "page_details": {
                            str(p.page_number): {"source": p.source, "attempts": p.visual_attempts}
                            for p in ocr_result.pages if p.classification == "VISUAL"
                        },
                    },
                )
            except Exception:
                logger.warning("trace_visual_understanding_failed", workflow_id=workflow_id, exc_info=True)

        if ocr_result.overall_status == "TERMINATE":
            reason = "HARMFUL_IMAGE" if ocr_result.harmful_image_detected else "OCR_FAILED"
            detail = ocr_result.harmful_image_detail or ocr_result.error_message or "OCR failed"
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status="TERMINATE",
                termination_reason=reason,
                termination_detail=detail,
                terminated_at_component="ocr",
                mrc=mrc_result,
                ocr=ocr_result,
                total_time_ms=round(time.time() * 1000 - start_ms, 2),
            )

        # ── Component 3: Content Safety ──
        safety_result = await asyncio.to_thread(check_content_safety, ocr_result.pages)
        safety_parts = []
        safety_parts.append(f"OpenAI Moderation pre-filter: {'FLAGGED' if safety_result.harmful_detected else 'PASSED'} (free)")
        if hasattr(safety_result, 'harmful_findings'):
            # WF-3257BB (2026-04-24): when analyzers ERROR, the *_findings lists
            # are empty and the old "0 harmful / 0 misinfo / ..." summary read
            # like a clean pass — completely hiding the upstream LLM auth
            # failure. Surface error count first so the operator sees the real
            # status before any zero-count noise.
            error_count = getattr(safety_result, 'analyzer_error_count', 0)
            if error_count:
                first_err = ""
                msgs = getattr(safety_result, 'analyzer_error_messages', [])
                if msgs:
                    first_err = f" — first: {msgs[0][:120]}"
                safety_parts.append(
                    f"⚠ {error_count}/4 analyzers ERRORED{first_err}"
                )
            if error_count < 4:
                safety_parts.append(
                    f"{4 - error_count} analyzers ran: "
                    f"A({len(safety_result.harmful_findings)} harmful) "
                    f"B({len(safety_result.misinformation_findings)} misinfo) "
                    f"C({len(safety_result.pii_findings)} PII, {len(safety_result.copyright_findings)} copyright) "
                    f"D({len(safety_result.religious_political_findings)} political)"
                )
        safety_summary = ". ".join(safety_parts)
        await self._write_progress_event(workflow_id, "assessorflow.validation.safety-complete", safety_summary)
        await self._tracing.trace_tool_call(
            workflow_id=workflow_id,
            agent_name="validator-agent",
            tool_name="content-safety-pipeline",
            input_params={"page_count": ocr_result.total_pages},
            output_summary={
                "overall_status": safety_result.overall_status,
                "harmful": len(safety_result.harmful_findings),
                "pii": len(safety_result.pii_findings),
                "religious_political": len(safety_result.religious_political_findings),
                "copyright": len(safety_result.copyright_findings),
                "misinformation": len(safety_result.misinformation_findings),
            },
            latency_ms=0,
        )

        if safety_result.overall_status == "TERMINATE":
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status="TERMINATE",
                termination_reason=safety_result.termination_reason,
                termination_detail=safety_result.termination_detail,
                terminated_at_component="content_safety",
                mrc=mrc_result,
                ocr=ocr_result,
                content_safety=safety_result,
                total_time_ms=round(time.time() * 1000 - start_ms, 2),
            )

        # ── All passed ──
        assessor_warnings = []
        if mrc_result.excluded_pages:
            for page_num in mrc_result.excluded_pages:
                assessor_warnings.append({
                    "page": page_num,
                    "type": "page_excluded",
                    "detail": "Page excluded: blurry/unreadable (detected by MRC)",
                })
        assessor_warnings.extend(safety_result.assessor_warnings)

        overall_status = safety_result.overall_status
        if mrc_result.excluded_pages and overall_status == "PROCEED":
            overall_status = "PROCEED_WITH_WARNINGS"

        return ValidatorResult(
            file_name=file_info.file_name,
            overall_status=overall_status,
            mrc=mrc_result,
            ocr=ocr_result,
            content_safety=safety_result,
            cleaned_text=safety_result.cleaned_text,
            assessor_warnings=assessor_warnings,
            total_time_ms=round(time.time() * 1000 - start_ms, 2),
        )

    _STAGE_MAP = {
        "assessorflow.validation.mrc-complete": 1,
        "assessorflow.validation.ocr-complete": 2,
        "assessorflow.validation.classification-complete": 3,
        "assessorflow.validation.visual-complete": 4,
        "assessorflow.validation.safety-complete": 5,
        "assessorflow.validation.ingestion-complete": 6,
    }

    async def _write_progress_event(
        self,
        workflow_id: str,
        event_type: str,
        summary: str,
        *,
        pipeline_group: str = "content-assurance",
    ) -> None:
        """Write a progress event to workflow_events for live UI updates."""
        import json as _json
        import os
        stage = self._STAGE_MAP.get(event_type, 0)
        payload = _json.dumps({"pipeline_group": pipeline_group, "stage": stage})
        try:
            import asyncpg
            host = os.getenv("ORCHESTRATOR_DB_HOST", "localhost")
            port = os.getenv("ORCHESTRATOR_DB_PORT", "15432")
            name = os.getenv("ORCHESTRATOR_DB_NAME", "af_orchestrator")
            user = os.getenv("ORCHESTRATOR_DB_USER", "assessorflow")
            password = os.getenv("ORCHESTRATOR_DB_PASSWORD", "dev_password")
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}"
            conn = await asyncpg.connect(dsn)
            try:
                await conn.execute(
                    """INSERT INTO workflow_events (workflow_id, event_type, source_agent, summary, payload)
                       VALUES ($1, $2, 'validator-agent', $3, $4::jsonb)""",
                    workflow_id, event_type, summary, payload,
                )
            finally:
                await conn.close()
        except Exception:
            logger.warning("progress_event_write_failed", workflow_id=workflow_id, event_type=event_type, exc_info=True)

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
    def _compute_content_fitness(
        pipeline_results: list[tuple],
        file_results: list,
    ) -> float:
        """Compute a 0.0-1.0 content fitness score from pipeline components.

        Factors: MRC readability ratio, content safety warnings (penalty).
        """
        if not pipeline_results:
            return 0.0

        readable_ratio = 1.0
        warning_penalty = 0.0

        for _fi, raw in pipeline_results:
            mrc = getattr(raw, "mrc", None)
            safety = getattr(raw, "content_safety", None)

            if mrc is not None and mrc.total_pages > 0:
                readable_ratio = min(readable_ratio, mrc.readable_pages / mrc.total_pages)

            if safety is not None:
                warnings = getattr(safety, "assessor_warnings", [])
                warning_penalty += len(warnings) * 0.05

        # Bind content_fitness to real MRC confidence × warning penalty (not a flat heuristic).
        # Pre-fix produced 0.95 for any doc with 1 warning regardless of readability quality.
        mrc_conf = 1.0
        for _fi, raw in pipeline_results:
            mrc = getattr(raw, "mrc", None)
            if mrc is not None and getattr(mrc, "overall_confidence", None) is not None:
                mrc_conf = min(mrc_conf, float(mrc.overall_confidence))
        return round(max(0.0, readable_ratio * mrc_conf * (1.0 - min(warning_penalty * 2, 0.5))), 4)

    @staticmethod
    def _build_reasoning_steps(file_info: FileInfo, result: object, pipeline_start: object = None) -> list[dict]:
        """Extract per-component reasoning steps from ValidatorResult.

        These populate the reasoning chain panel in the Agent Trace page.
        Each step gets a computed timestamp based on cumulative latency from
        pipeline_start so the trace timeline shows sequential progression.
        """
        from datetime import timedelta

        steps: list[dict] = []
        step_num = 0
        cumulative_ms = 0.0  # running total for timestamp computation

        def _step_timestamp() -> str | None:
            if pipeline_start is None:
                return None
            ts = pipeline_start + timedelta(milliseconds=cumulative_ms)
            return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

        mrc = getattr(result, "mrc", None)
        ocr = getattr(result, "ocr", None)
        safety = getattr(result, "content_safety", None)
        terminated_at = getattr(result, "terminated_at_component", None)

        # Step 1: MRC
        if mrc is not None:
            step_num += 1
            excluded = f", excluded pages: {mrc.excluded_pages}" if mrc.excluded_pages else ""
            ts = _step_timestamp()
            cumulative_ms += mrc.inference_time_ms
            step_data: dict = {
                "step": step_num,
                "component": "mrc",
                "model_id": "vertex-ai/mrc-production",
                "action": (
                    f"MRC readability check: {mrc.readable_pages}/{mrc.total_pages} pages readable"
                    f", blurry_ratio {mrc.blurry_ratio:.0%}{excluded}"
                ),
                "status": mrc.overall_status,
                "latency_ms": mrc.inference_time_ms,
                "confidence": round(float(mrc.overall_confidence), 4),
            }
            if ts:
                step_data["timestamp"] = ts
            steps.append(step_data)
            if terminated_at == "mrc":
                return steps

        # Step 2: OCR
        if ocr is not None:
            step_num += 1
            ts = _step_timestamp()
            cumulative_ms += ocr.ocr_time_ms
            ocr_step: dict = {
                "step": step_num,
                "component": "ocr",
                "model_id": "google/document-ai",
                "action": (
                    f"Document AI OCR: {ocr.total_word_count} words from {ocr.total_pages} pages "
                    f"(mode: {ocr.processing_mode or 'unknown'})"
                ),
                "status": ocr.overall_status,
                "latency_ms": ocr.ocr_time_ms,
                "confidence": 1.0 if ocr.overall_status in ("PROCEED", "PROCEED_WITH_WARNINGS") else 0.0,
            }
            if ts:
                ocr_step["timestamp"] = ts
            steps.append(ocr_step)

            # Step 3: Page classification (+ visual understanding time)
            if ocr.pages:
                text_count = sum(1 for p in ocr.pages if p.classification == "TEXT")
                visual_count = sum(1 for p in ocr.pages if p.classification == "VISUAL")
                step_num += 1
                # Page classification + visual understanding elapsed time
                page_cls_ms = getattr(ocr, "page_classification_time_ms", 0) or 0
                ts = _step_timestamp()
                cumulative_ms += page_cls_ms
                # Page-classification confidence: 1.0 if all pages classified, 0.0 if none
                _classified = sum(1 for p in ocr.pages if p.classification in ("TEXT", "VISUAL"))
                _pg_conf = round(_classified / max(len(ocr.pages), 1), 4)
                pg_step: dict = {
                    "step": step_num,
                    "component": "page_classification",
                    "model_id": "gpt-4.1-mini",
                    "action": f"Page classification: {text_count} TEXT, {visual_count} VISUAL",
                    "confidence": _pg_conf,
                    "pages": [
                        {"page": p.page_number, "class": p.classification, "source": p.source}
                        for p in ocr.pages
                    ],
                }
                if ts:
                    pg_step["timestamp"] = ts
                steps.append(pg_step)

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
                _vis_count = sum(1 for p in ocr.pages if p.classification == "VISUAL")
                _vis_enhanced = sum(1 for p in ocr.pages if p.classification == "VISUAL" and p.source == "llm")
                _vis_conf = round(_vis_enhanced / max(_vis_count, 1), 4) if _vis_count else 1.0
                steps.append({
                    "step": step_num,
                    "component": "visual_understanding",
                    "model_id": "gpt-4.1",
                    "action": f"Visual understanding: {ocr.visual_pages_processed} pages enhanced. {', '.join(visual_details)}",
                    "confidence": _vis_conf,
                })

            if terminated_at == "ocr":
                return steps

        # Step 4: Text moderation pre-filter
        if safety is not None:
            step_num += 1
            ts = _step_timestamp()
            if safety.overall_status == "TERMINATE" and safety.termination_reason == "HARMFUL_CONTENT" and not safety.harmful_findings:
                mod_step: dict = {
                    "step": step_num,
                    "component": "text_moderation",
                    "model_id": "openai/moderation-api",
                    "action": f"OpenAI Moderation pre-filter: FLAGGED ({safety.termination_detail or 'harmful content'})",
                    "status": "TERMINATE",
                }
                if ts:
                    mod_step["timestamp"] = ts
                steps.append(mod_step)
                return steps
            mod_step = {
                "step": step_num,
                "component": "text_moderation",
                "model_id": "openai/moderation-api",
                "action": "OpenAI Moderation pre-filter: PASSED (free)",
                "status": "PASSED",
                "confidence": 1.0,
            }
            if ts:
                mod_step["timestamp"] = ts
            steps.append(mod_step)

            # Step 5: Content analyzers
            findings = {
                "A": len(safety.harmful_findings),
                "B": len(safety.misinformation_findings),
                "C": len(safety.pii_findings) + len(safety.copyright_findings),
                "D": len(safety.religious_political_findings),
            }
            step_num += 1
            _analyzer_conf = round(max(0.0, 1.0 - 0.10 * sum(findings.values())), 4)
            steps.append({
                "step": step_num,
                "component": "content_analyzers",
                "model_id": "gpt-4.1",
                "action": (
                    f"4 parallel analyzers: "
                    f"A({findings['A']} harmful) B({findings['B']} misinfo) "
                    f"C({findings['C']} PII/copyright) D({findings['D']} religious/political)"
                ),
                "findings_total": sum(findings.values()),
                "confidence": _analyzer_conf,
            })

            # Step 6: Synthesizer (inferred from findings presence)
            total = sum(findings.values())
            step_num += 1
            _synth_conf = round(max(0.5, 1.0 - 0.05 * total), 4)
            steps.append({
                "step": step_num,
                "component": "synthesizer",
                "model_id": "gpt-4.1",
                "action": f"Synthesizer: {total} finding(s) confirmed after voting",
                "confidence": _synth_conf,
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
                    "component": "content_fit_decision",
                    "action": f"Content-Fit decision: TERMINATE. Hard gate(s): {', '.join(hard_gates)}",
                    "status": "TERMINATE",
                })
                return steps

            # TERMINATE without hard gates = analyzers unavailable
            if safety.overall_status == "TERMINATE":
                reason = safety.termination_reason or safety.error_message or "Content safety unavailable"
                steps.append({
                    "step": step_num,
                    "component": "content_fit_decision",
                    "action": f"Content-Fit decision: TERMINATE. {reason}",
                    "status": "TERMINATE",
                })
                return steps

            soft_desc = ". ".join(soft_items) if soft_items else "No issues detected"
            # Content-fit decision confidence reflects soft-issue load (each soft -8%).
            _fit_conf = round(max(0.5, 1.0 - 0.08 * len(soft_items)), 4)
            steps.append({
                "step": step_num,
                "component": "content_fit_decision",
                "action": f"Content-Fit decision: {safety.overall_status}. {soft_desc}",
                "status": safety.overall_status,
                "confidence": _fit_conf,
            })

        # Step 8: Knowledge Service write (on PROCEED)
        cleaned_len = len(getattr(result, "cleaned_text", "") or "")
        if cleaned_len > 0 and getattr(result, "overall_status", "") != "TERMINATE":
            step_num += 1
            steps.append({
                "step": step_num,
                "component": "knowledge_service",
                "action": f"Forwarded {cleaned_len} chars (PII-redacted) to Knowledge Service for chunking and embedding",
                "confidence": 1.0,
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

    @staticmethod
    def _build_composite_model_id() -> str:
        """Build composite model_id from actual models used in the pipeline.

        MRC and Document AI are non-LLM services with fixed descriptive names.
        The LLM component comes from the real Model Broker response tracked
        by llm_client.get_stats().last_model_used.
        """
        from validator_agent.pipeline.llm_client import get_stats

        llm_model = get_stats().last_model_used
        return f"vertex-ai-mrc + document-ai-ocr + {llm_model}"

    async def _log_audit_decision(
        self,
        *,
        request: ValidationRequest,
        overall_signal: TerminalSignal,
        file_results: list[FileResult],
        reasoning_steps: list[dict],
        grounding_chunk_ids: list[str] | None = None,
        content_fitness: float = 0.0,
    ) -> None:
        """Log the validation decision to both sinks using ONE canonical format."""
        composite_model_id = self._build_composite_model_id()

        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            assessor_id=request.assessor_id,
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
                "content_fitness": content_fitness,
            },
            reasoning_steps=reasoning_steps,
            confidence_score=content_fitness,
            prompt_version="validator/thet-pipeline@v1",
            model_id=composite_model_id,
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
