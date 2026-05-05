"""ValidatorService — orchestration + infrastructure layer.

Responsibilities (infrastructure only — does NOT re-implement the pipeline):
  1. Download files via StoragePort
  2. Call validate_file() from pipeline/validator_pipeline.py
  3. Map ValidatorResult -> Terminal Signal
  4. Forward cleaned_text to Knowledge Service
  5. Log decisions to Decision Audit (dual-sink via Pub/Sub)
  6. Trace to Langfuse via TracingPort

The ML/AI pipeline logic lives in pipeline/validator_pipeline.py.
This file only handles infrastructure concerns around it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from validator_agent.domain.audit_models import DecisionLogEntry
from validator_agent.ports.tracing_port import TracingPort

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

logger = structlog.get_logger(__name__)

# Map pipeline status strings to Terminal Signal enums
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

_STATUS_MAP: dict[str, TerminalSignalStatus] = {
    "PROCEED": TerminalSignalStatus.PROCEED,
    "PROCEED_WITH_WARNINGS": TerminalSignalStatus.PROCEED_WITH_WARNINGS,
    "PROCEED_WITH_EXCLUSIONS": TerminalSignalStatus.PROCEED_WITH_WARNINGS,
    "TERMINATE": TerminalSignalStatus.TERMINATE,
}


class ValidatorService:
    """Orchestration layer for the Validator Agent.

    Calls the pipeline (validate_file or pipeline_fn stub), then handles
    all infrastructure: tracing, audit, KB forwarding, progress events.

    Stateless — all state in request/response objects.
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

    # ─── Public API ───────────────────────────────────────────────

    async def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Run the full validation pipeline for all files."""
        pipeline_start = datetime.now(UTC)
        file_results: list[FileResult] = []
        all_cleaned_text: list[str] = []
        all_warnings: list[dict] = []
        reasoning_steps: list[dict] = []
        pipeline_results: list[tuple[FileInfo, object]] = []

        for file_info in request.files:
            result, raw = await self._validate_single_file(
                file_info=file_info,
                workflow_id=request.workflow_id,
            )
            file_results.append(result)
            pipeline_results.append((file_info, raw))
            if result.cleaned_text:
                all_cleaned_text.append(result.cleaned_text)
            all_warnings.extend(result.assessor_warnings)

        # Build reasoning steps from pipeline results
        for fi, raw in pipeline_results:
            if self._is_text_file(fi):
                word_count = len((raw.cleaned_text or "").split())
                reasoning_steps.extend(self._text_file_reasoning(fi, raw, word_count, len(reasoning_steps)))
            else:
                reasoning_steps.extend(self._build_reasoning_steps(fi, raw, pipeline_start))

        overall_signal = self._compute_overall_signal(file_results)

        # Forward cleaned text to Knowledge Service on PROCEED
        chunk_ids = await self._forward_to_knowledge_service(
            request, overall_signal, all_cleaned_text, reasoning_steps,
        )

        # Content fitness score
        content_fitness = self._compute_content_fitness(pipeline_results, file_results)

        # Audit decision (dual-sink)
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

    # ─── Single File Validation ───────────────────────────────────

    async def _validate_single_file(
        self, *, file_info: FileInfo, workflow_id: str,
    ) -> tuple[FileResult, object]:
        """Validate a single file: download -> pipeline -> map result -> trace."""
        file_bytes = await self._storage.download_file(file_info.storage_path)

        # Text files (.md, .txt): content safety only, skip MRC/OCR
        if self._is_text_file(file_info):
            raw = await self._validate_text_file(file_bytes, file_info, workflow_id)
            return self._map_to_result(file_info, raw), raw

        # Set workflow context for LLM token tracking
        from validator_agent.pipeline.llm_client import set_workflow_context
        set_workflow_context(workflow_id)

        # Run pipeline (stub for tests, real for production)
        if self._pipeline_fn is not None:
            raw = await asyncio.to_thread(self._pipeline_fn, file_bytes, file_info.file_name)
        else:
            from validator_agent.pipeline.validator_pipeline import validate_file
            raw = await asyncio.to_thread(validate_file, file_bytes, file_info.file_name)

        result = self._map_to_result(file_info, raw)
        return result, raw

    async def _validate_text_file(
        self, file_bytes: bytes, file_info: FileInfo, workflow_id: str,
    ) -> object:
        """Validate text file (.md, .txt) — content safety only."""
        from validator_agent.pipeline.content_safety import check_content_safety
        from validator_agent.pipeline.llm_client import set_workflow_context
        from validator_agent.pipeline.models import PageOcrResult, ValidatorResult

        set_workflow_context(workflow_id)
        start_ms = time.time() * 1000
        text = file_bytes.decode("utf-8", errors="replace")
        word_count = len(text.split())

        logger.info("text_file_validation", file_name=file_info.file_name, word_count=word_count)

        try:
            pages = [PageOcrResult(
                page_number=1, extracted_text=text, word_count=word_count,
                confidence=1.0, classification="TEXT", source="web_research",
            )]
            safety_result = await asyncio.to_thread(check_content_safety, pages)
            elapsed = time.time() * 1000 - start_ms

            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status=safety_result.overall_status,
                termination_reason=safety_result.termination_reason,
                termination_detail=safety_result.termination_detail,
                cleaned_text=text,
                assessor_warnings=safety_result.assessor_warnings,
                total_time_ms=elapsed,
            )
        except Exception:
            logger.warning("text_file_safety_check_failed", exc_info=True)
            elapsed = time.time() * 1000 - start_ms
            return ValidatorResult(
                file_name=file_info.file_name,
                overall_status="PROCEED",
                termination_detail=f"Text file validated: {word_count} words (safety check skipped)",
                cleaned_text=text,
                assessor_warnings=[],
                total_time_ms=elapsed,
            )

    # ─── Result Mapping ───────────────────────────────────────────

    def _map_to_result(self, file_info: FileInfo, raw: object) -> FileResult:
        """Map ValidatorResult -> FileResult with Terminal Signal."""
        status = _STATUS_MAP.get(raw.overall_status, TerminalSignalStatus.TERMINATE)
        reason_code = _REASON_CODE_MAP.get(
            raw.termination_reason,
            ReasonCode.VALIDATION_PASSED if status != TerminalSignalStatus.TERMINATE
            else ReasonCode.CONTENT_POLICY_VIOLATION,
        )
        message = raw.termination_detail or f"Document {file_info.file_name} validated: {raw.overall_status}"

        signal = TerminalSignal(status=status, reason_code=reason_code, message=message[:500])

        logger.info(
            "validation_complete", file_name=file_info.file_name,
            status=raw.overall_status, reason=raw.termination_reason,
            time_ms=raw.total_time_ms,
        )

        return FileResult(
            file_name=file_info.file_name,
            terminal_signal=signal,
            cleaned_text=raw.cleaned_text,
            assessor_warnings=raw.assessor_warnings,
            total_time_ms=raw.total_time_ms,
            terminated_at_component=getattr(raw, "terminated_at_component", None),
        )

    @staticmethod
    def _compute_overall_signal(file_results: list[FileResult]) -> TerminalSignal:
        """ANY TERMINATE -> overall TERMINATE. Any warnings -> overall PROCEED_WITH_WARNINGS."""
        if len(file_results) == 1:
            return file_results[0].terminal_signal

        for result in file_results:
            if result.terminal_signal.status == TerminalSignalStatus.TERMINATE:
                return result.terminal_signal

        has_warnings = any(
            r.terminal_signal.status == TerminalSignalStatus.PROCEED_WITH_WARNINGS
            for r in file_results
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

    # ─── Knowledge Service Forwarding ─────────────────────────────

    async def _forward_to_knowledge_service(
        self, request: ValidationRequest, overall_signal: TerminalSignal,
        all_cleaned_text: list[str], reasoning_steps: list[dict],
    ) -> list[str]:
        """Forward cleaned text to Knowledge Service on PROCEED."""
        if overall_signal.status == TerminalSignalStatus.TERMINATE:
            return []

        combined_text = "\n\n".join(all_cleaned_text)
        if not combined_text.strip():
            return []

        source_type = "web_research" if request.validation_type == "web_research_validation" else "ocr_extracted"
        source = "web_research" if request.validation_type == "web_research_validation" else "upload"

        chunk_ids = await self._knowledge_service.process_material(
            workflow_id=request.workflow_id,
            content_text=combined_text,
            source_type=source_type,
            assessment_id=request.assessment_id,
            source=source,
        )

        if chunk_ids:
            reasoning_steps.append({
                "step": len(reasoning_steps) + 1,
                "action": f"Content fit: PROCEED — stored {len(chunk_ids)} chunks to Knowledge Service",
                "component": "kb_write",
                "chunks_stored": len(chunk_ids),
            })
            await self._tracing.trace_tool_call(
                workflow_id=request.workflow_id, agent_name="validator-agent",
                tool_name="knowledge-service-ingestion",
                input_params={"content_length": len(combined_text)},
                output_summary={"chunks_stored": len(chunk_ids), "chunk_ids": chunk_ids[:5]},
                latency_ms=0,
            )

        return chunk_ids

    # ─── Reasoning Steps ──────────────────────────────────────────

    @staticmethod
    def _text_file_reasoning(fi: FileInfo, raw: object, word_count: int, offset: int) -> list[dict]:
        """Reasoning steps for text file validation (Phase 5)."""
        return [
            {"step": offset + 1, "component": "mrc", "action": f"MRC: SKIPPED (text file, {word_count} words)"},
            {"step": offset + 2, "component": "ocr", "action": f"OCR: SKIPPED (already text, read from {fi.file_name})"},
            {"step": offset + 3, "component": "content_fit_decision", "action": f"Content-Fit: {raw.overall_status}. {fi.file_name} ({word_count} words)"},
        ]

    @staticmethod
    def _build_reasoning_steps(file_info: FileInfo, result: object, pipeline_start: object = None) -> list[dict]:
        """Extract per-component reasoning steps from ValidatorResult."""
        from datetime import timedelta

        steps: list[dict] = []
        step_num = 0
        cumulative_ms = 0.0

        def _ts() -> str | None:
            if pipeline_start is None:
                return None
            ts = pipeline_start + timedelta(milliseconds=cumulative_ms)
            return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

        mrc = getattr(result, "mrc", None)
        ocr = getattr(result, "ocr", None)
        safety = getattr(result, "content_safety", None)
        terminated_at = getattr(result, "terminated_at_component", None)

        # MRC
        if mrc is not None:
            step_num += 1
            excluded = f", excluded pages: {mrc.excluded_pages}" if mrc.excluded_pages else ""
            ts = _ts()
            cumulative_ms += mrc.inference_time_ms
            step: dict = {
                "step": step_num, "component": "mrc", "model_id": "vertex-ai/mrc-production",
                "action": f"MRC: {mrc.readable_pages}/{mrc.total_pages} pages readable, blurry_ratio {mrc.blurry_ratio:.0%}{excluded}",
                "status": mrc.overall_status, "latency_ms": mrc.inference_time_ms,
                "confidence": round(float(mrc.overall_confidence), 4),
            }
            if ts:
                step["timestamp"] = ts
            steps.append(step)
            if terminated_at == "mrc":
                return steps

        # OCR
        if ocr is not None:
            step_num += 1
            ts = _ts()
            cumulative_ms += ocr.ocr_time_ms
            ocr_step: dict = {
                "step": step_num, "component": "ocr", "model_id": "google/document-ai",
                "action": f"OCR: {ocr.total_word_count} words from {ocr.total_pages} pages ({ocr.processing_mode or 'unknown'})",
                "status": ocr.overall_status, "latency_ms": ocr.ocr_time_ms,
            }
            if ts:
                ocr_step["timestamp"] = ts
            steps.append(ocr_step)

            # Page classification
            if ocr.pages:
                text_count = sum(1 for p in ocr.pages if p.classification == "TEXT")
                visual_count = sum(1 for p in ocr.pages if p.classification == "VISUAL")
                step_num += 1
                steps.append({
                    "step": step_num, "component": "page_classification", "model_id": "gpt-4.1-mini",
                    "action": f"Classification: {text_count} TEXT, {visual_count} VISUAL",
                })

            # Harmful image
            if ocr.harmful_image_detected:
                step_num += 1
                steps.append({
                    "step": step_num, "component": "image_moderation",
                    "action": f"Image moderation: FLAGGED page {ocr.harmful_image_page}. {ocr.harmful_image_detail or ''}",
                    "status": "TERMINATE",
                })
                return steps

            # Visual understanding
            if ocr.visual_pages_processed and ocr.visual_pages_processed > 0:
                step_num += 1
                steps.append({
                    "step": step_num, "component": "visual_understanding", "model_id": "gpt-4.1",
                    "action": f"Visual understanding: {ocr.visual_pages_processed} pages enhanced",
                })

            if terminated_at == "ocr":
                return steps

        # Content safety
        if safety is not None:
            # Text moderation pre-filter
            step_num += 1
            if safety.overall_status == "TERMINATE" and safety.termination_reason == "HARMFUL_CONTENT" and not safety.harmful_findings:
                steps.append({
                    "step": step_num, "component": "text_moderation", "model_id": "openai/moderation-api",
                    "action": f"Moderation pre-filter: FLAGGED ({safety.termination_detail or 'harmful'})",
                    "status": "TERMINATE",
                })
                return steps

            steps.append({
                "step": step_num, "component": "text_moderation", "model_id": "openai/moderation-api",
                "action": "Moderation pre-filter: PASSED (free)", "status": "PASSED",
            })

            # Analyzers
            findings = {
                "A": len(safety.harmful_findings), "B": len(safety.misinformation_findings),
                "C": len(safety.pii_findings) + len(safety.copyright_findings),
                "D": len(safety.religious_political_findings),
            }
            step_num += 1
            steps.append({
                "step": step_num, "component": "content_analyzers", "model_id": "gpt-4.1",
                "action": f"4 analyzers: A({findings['A']} harmful) B({findings['B']} misinfo) C({findings['C']} PII/copyright) D({findings['D']} political)",
                "findings_total": sum(findings.values()),
            })

            # Synthesizer
            step_num += 1
            steps.append({
                "step": step_num, "component": "synthesizer", "model_id": "gpt-4.1",
                "action": f"Synthesizer: {sum(findings.values())} finding(s) confirmed",
            })

            # Content fit decision
            step_num += 1
            hard_gates = []
            if safety.harmful_detected:
                hard_gates.append("harmful")
            if safety.religious_political_detected:
                hard_gates.append("religious/political")

            if hard_gates:
                steps.append({
                    "step": step_num, "component": "content_fit_decision",
                    "action": f"Content-Fit: TERMINATE. Hard gate(s): {', '.join(hard_gates)}",
                    "status": "TERMINATE",
                })
                return steps

            if safety.overall_status == "TERMINATE":
                reason = safety.termination_reason or safety.error_message or "unavailable"
                steps.append({
                    "step": step_num, "component": "content_fit_decision",
                    "action": f"Content-Fit: TERMINATE. {reason}", "status": "TERMINATE",
                })
                return steps

            soft_items = []
            if safety.pii_detected:
                soft_items.append(f"{len(safety.pii_findings)} PII redacted")
            if safety.copyright_detected:
                soft_items.append(f"{len(safety.copyright_findings)} copyright warned")
            if safety.misinformation_detected:
                soft_items.append(f"{len(safety.misinformation_findings)} misinformation warned")

            soft_desc = ". ".join(soft_items) if soft_items else "No issues"
            steps.append({
                "step": step_num, "component": "content_fit_decision",
                "action": f"Content-Fit: {safety.overall_status}. {soft_desc}",
                "status": safety.overall_status,
            })

        # KB write
        cleaned_len = len(getattr(result, "cleaned_text", "") or "")
        if cleaned_len > 0 and getattr(result, "overall_status", "") != "TERMINATE":
            step_num += 1
            steps.append({
                "step": step_num, "component": "knowledge_service",
                "action": f"Forwarded {cleaned_len} chars (PII-redacted) to Knowledge Service",
            })

        return steps

    # ─── Content Fitness Score ────────────────────────────────────

    @staticmethod
    def _compute_content_fitness(pipeline_results: list[tuple], file_results: list) -> float:
        """0.0-1.0 score from MRC confidence x readable_ratio x warning_penalty."""
        if not pipeline_results:
            return 0.0

        readable_ratio = 1.0
        warning_penalty = 0.0
        mrc_conf = 1.0

        for _fi, raw in pipeline_results:
            mrc = getattr(raw, "mrc", None)
            safety = getattr(raw, "content_safety", None)

            if mrc is not None and mrc.total_pages > 0:
                readable_ratio = min(readable_ratio, mrc.readable_pages / mrc.total_pages)
            if mrc is not None and getattr(mrc, "overall_confidence", None) is not None:
                mrc_conf = min(mrc_conf, float(mrc.overall_confidence))
            if safety is not None:
                warning_penalty += len(getattr(safety, "assessor_warnings", [])) * 0.05

        return round(max(0.0, readable_ratio * mrc_conf * (1.0 - min(warning_penalty * 2, 0.5))), 4)

    # ─── Audit Decision (Dual-Sink) ───────────────────────────────

    async def _log_audit_decision(
        self, *, request: ValidationRequest, overall_signal: TerminalSignal,
        file_results: list[FileResult], reasoning_steps: list[dict],
        grounding_chunk_ids: list[str] | None = None, content_fitness: float = 0.0,
    ) -> None:
        """Log decision to Decision Audit Service (Pub/Sub) + Langfuse."""
        composite_model_id = self._build_composite_model_id()

        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            assessor_id=request.assessor_id,
            input={
                "file_count": len(request.files),
                "files": [f.file_name for f in request.files],
                "phase": "Phase 3: Material Validation" if request.validation_type != "web_research_validation" else "Phase 5: Web Research Validation",
                "tools_used": ["mrc-vertex-ai", "documentai-ocr", "page-classifier", "visual-understanding", "moderation-api", "content-analyzers-x4", "content-synthesizer"],
            },
            output={
                "terminal_signal": overall_signal.model_dump(),
                "files_validated": len(file_results),
                "files_passed": sum(1 for f in file_results if f.terminal_signal.status != TerminalSignalStatus.TERMINATE),
                "assessor_warnings_count": sum(len(f.assessor_warnings) for f in file_results),
                "content_fitness": content_fitness,
            },
            reasoning_steps=reasoning_steps,
            confidence_score=content_fitness,
            prompt_version="validator/thet-pipeline@v1",
            model_id=composite_model_id,
            grounding_sources=grounding_chunk_ids or [f.file_name for f in file_results],
        )

        # Sink 1: Decision Audit Service (via Pub/Sub)
        await self._decision_audit.log_decision(
            workflow_id=request.workflow_id,
            agent_name="validator-agent",
            decision_type="content_validation",
            payload=entry.model_dump(),
        )

        # Sink 2: Langfuse
        await self._tracing.trace_decision(entry)

    @staticmethod
    def _build_composite_model_id() -> str:
        """Build composite model_id from actual models used."""
        from validator_agent.pipeline.llm_client import get_stats
        llm_model = get_stats().last_model_used
        return f"vertex-ai-mrc + document-ai-ocr + {llm_model}"

    # ─── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _is_text_file(fi: FileInfo) -> bool:
        return fi.file_type in ("text/markdown", "text/plain") or fi.file_name.endswith((".md", ".txt"))
