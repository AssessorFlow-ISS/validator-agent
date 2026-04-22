"""FastAPI application entry point for the Validator Agent (#12).

Wires up port implementations based on environment configuration and
registers the API routes.  The ValidatorService is stored on app.state
so routes can access it via request.app.state.

In event-driven mode (EVENT_ADAPTER=pubsub), the agent subscribes to
assessorflow.validation.trigger on startup and processes messages
automatically — no HTTP call needed from the Orchestrator.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.material_validation_stub import StubMaterialValidationAdapter
from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter
from af_shared.adapters.factory import get_tracing
from validator_agent.api.routes import router
from validator_agent.api.schemas import FileInfo, ValidationRequest
from validator_agent.config import ValidatorConfig
from validator_agent.domain.services import ValidatorService
from validator_agent.domain.terminal_signal import TerminalSignalStatus
from validator_agent.ports.material_validation_port import MaterialValidationPort

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


def _build_service(config: ValidatorConfig) -> tuple[ValidatorService, Any, Any, Any, MaterialValidationPort]:
    """Construct the ValidatorService with adapter implementations.

    Returns (service, event_publisher, decision_audit, knowledge_service,
    material_validation) — event_publisher is returned separately so the
    lifespan can set up Pub/Sub subscriptions; decision_audit,
    knowledge_service, and material_validation are returned for graceful
    pool/channel shutdown.
    """
    # -- MRC adapter --------------------------------------------------------
    StubMrcAdapter()

    # -- OCR adapter --------------------------------------------------------
    StubOcrAdapter()

    # -- Model Broker adapter -----------------------------------------------
    if config.llm_provider == "stub":
        StubModelBrokerAdapter()
    elif config.llm_provider in ("http", "google_ai_studio", "vertex_ai"):
        from validator_agent.adapters.model_broker_http import ModelBrokerHttpAdapter
        ModelBrokerHttpAdapter()
        logger.info("using_real_model_broker", url=os.environ.get("MODEL_BROKER_URL", "localhost:8010"))
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider}")

    # -- Knowledge Service adapter ------------------------------------------
    if config.knowledge_adapter == "http":
        from validator_agent.adapters.knowledge_service_http import KnowledgeServiceHttpAdapter
        knowledge_service = KnowledgeServiceHttpAdapter()
        logger.info("using_http_knowledge_service", url=os.environ.get("KNOWLEDGE_SERVICE_URL", "http://localhost:8020"))
    else:
        knowledge_service = StubKnowledgeServiceAdapter()

    # -- Decision Audit adapter ---------------------------------------------
    if config.audit_adapter == "postgres":
        from validator_agent.adapters.decision_audit_postgres import PostgresDecisionAuditAdapter
        decision_audit = PostgresDecisionAuditAdapter()
        logger.info("using_postgres_decision_audit")
    else:
        decision_audit = StubDecisionAuditAdapter()

    # -- Event Publisher adapter --------------------------------------------
    if config.event_adapter == "stub":
        event_publisher = StubEventPublisherAdapter()
    elif config.event_adapter in ("pubsub", "real", "emulator"):
        from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter
        event_publisher = PubSubEventPublisherAdapter()
        logger.info("using_real_pubsub_adapter")
    else:
        raise ValueError(f"Unknown EVENT_ADAPTER: {config.event_adapter}")

    # -- Material Validation adapter (Submission Service gRPC) -------------
    # Phase 6C.1 — replaces the previous HTTP-to-API-Server read path and
    # direct-DB writes to assessment_materials.readiness_status.
    material_validation: MaterialValidationPort
    if config.material_adapter == "grpc":
        from validator_agent.adapters.material_validation_grpc import (
            GrpcMaterialValidationAdapter,
        )
        material_validation = GrpcMaterialValidationAdapter()
        logger.info(
            "using_grpc_material_validation",
            url=os.environ.get(
                "SUBMISSION_SERVICE_GRPC_URL", "localhost:9001",
            ),
        )
    else:
        material_validation = StubMaterialValidationAdapter()

    # -- Storage adapter ----------------------------------------------------
    if config.storage_adapter == "local":
        from validator_agent.adapters.storage_local import LocalStorageAdapter
        storage = LocalStorageAdapter()
        logger.info("using_local_storage_adapter", upload_dir=os.environ.get("UPLOAD_DIR", "/tmp/assessorflow-uploads"))
    elif config.storage_adapter == "gcs":
        from validator_agent.adapters.storage_gcs import GcsStorageAdapter
        storage = GcsStorageAdapter()
        logger.info("using_gcs_storage_adapter", bucket=os.environ.get("GCS_BUCKET_NAME", "(from URI)"))
    else:
        storage = StubStorageAdapter()

    # -- Tracing adapter (config-driven via TRACING_ADAPTER env var) --------
    # TRACING_ADAPTER=stub → StubTracingAdapter (console)
    # TRACING_ADAPTER=langfuse → LangfuseTracingAdapter (real Langfuse SDK)
    tracing = get_tracing()

    # ValidatorService wraps Thet's 3-component pipeline.
    # In stub mode, use a keyword-based stub pipeline (no external calls).
    # In production, Thet's real pipeline runs MRC + Document AI + Content Safety.
    pipeline_fn = None
    if config.llm_provider == "stub":
        from validator_agent.adapters.pipeline_stub import stub_pipeline_fn
        pipeline_fn = stub_pipeline_fn

    service = ValidatorService(
        knowledge_service=knowledge_service,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        storage=storage,
        tracing=tracing,
        pipeline_fn=pipeline_fn,
    )

    return service, event_publisher, decision_audit, knowledge_service, material_validation


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    config = ValidatorConfig.from_env()
    logger.info("starting_validator_agent", config=config)

    service, event_publisher, decision_audit, knowledge_service, material_validation = _build_service(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        """Startup/shutdown lifecycle — subscribe to Pub/Sub on startup."""
        if config.event_adapter in ("pubsub", "real", "emulator"):
            from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

            if isinstance(event_publisher, PubSubEventPublisherAdapter):
                async def handle_trigger(payload: dict) -> None:
                    """Process a validation trigger from Pub/Sub.

                    The Orchestrator dispatches {workflow_id, assessment_id, phase, agent}.
                    This handler fetches material metadata from the Submission
                    Service via gRPC (``GetMaterials``) and builds FileInfo
                    objects for the validation pipeline, then writes a
                    per-material decision back via ``UpdateMaterialValidation``
                    once validation completes.
                    """
                    workflow_id = payload.get("workflow_id", "unknown")
                    assessment_id = payload.get("assessment_id", "unknown")
                    logger.info(
                        "validation_trigger_received",
                        workflow_id=workflow_id,
                        assessment_id=assessment_id,
                        payload_keys=list(payload.keys()),
                    )

                    validation_type = payload.get("validation_type", "material_validation")
                    file_infos, file_to_material_id = await _load_files_for_validation(
                        material_validation=material_validation,
                        assessment_id=assessment_id,
                        validation_type=validation_type,
                    )

                    if not file_infos:
                        logger.error("no_materials_found", assessment_id=assessment_id)
                        await event_publisher.publish(
                            topic="assessorflow.validation.complete",
                            payload={
                                "workflow_id": workflow_id,
                                "assessment_id": assessment_id,
                                "terminal_signal": {
                                    "status": "TERMINATE",
                                    "reason_code": "NO_MATERIALS",
                                    "message": "No materials found for assessment",
                                },
                                "file_results": [],
                                "source_agent": "validator-agent",
                            },
                        )
                        return

                    request = ValidationRequest(
                        workflow_id=workflow_id,
                        assessment_id=assessment_id,
                        assessor_id=payload.get("assessor_id"),
                        validation_type=validation_type,
                        files=file_infos,
                    )

                    os.environ["CURRENT_WORKFLOW_ID"] = request.workflow_id
                    response = await service.validate(request)

                    await _write_material_decisions(
                        material_validation=material_validation,
                        assessment_id=assessment_id,
                        response=response,
                        file_to_material_id=file_to_material_id,
                    )

                    await event_publisher.publish(
                        topic="assessorflow.validation.complete",
                        payload={
                            "workflow_id": request.workflow_id,
                            "assessment_id": request.assessment_id,
                            "terminal_signal": response.terminal_signal.model_dump(),
                            "file_results": [
                                {"file_name": r.file_name, "terminal_signal": r.terminal_signal.model_dump()}
                                for r in response.file_results
                            ],
                            "source_agent": "validator-agent",
                        },
                    )
                    logger.info("validation_complete_published", workflow_id=request.workflow_id, status=response.terminal_signal.status)

                await event_publisher.subscribe_and_process("assessorflow.validation.trigger.sub", handle_trigger)
                logger.info("validator_listening", topic="assessorflow.validation.trigger")

        yield

        # Shutdown: close adapter pools/clients gracefully.
        if hasattr(knowledge_service, "close"):
            await knowledge_service.close()
        if hasattr(decision_audit, "close"):
            await decision_audit.close()
        if hasattr(material_validation, "close"):
            await material_validation.close()

        logger.info("validator_agent_shutting_down")

    application = FastAPI(
        title="Validator Agent",
        description="Centralized content safety gate for AssessorFlow (#12)",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    application.state.validator_service = service
    application.include_router(router)

    # -- HTTP trigger endpoint (golden eval, no Pub/Sub) -------------------
    @application.post("/trigger")
    async def trigger(body: dict) -> dict:
        """HTTP trigger — same as Pub/Sub handler, returns completion payload."""
        workflow_id = body.get("workflow_id", "unknown")
        assessment_id = body.get("assessment_id", "unknown")
        validation_type = body.get("validation_type", "material_validation")

        file_infos, file_to_material_id = await _load_files_for_validation(
            material_validation=material_validation,
            assessment_id=assessment_id,
            validation_type=validation_type,
        )

        if not file_infos:
            return {
                "workflow_id": workflow_id,
                "assessment_id": assessment_id,
                "terminal_signal": {
                    "status": "TERMINATE",
                    "reason_code": "NO_MATERIALS",
                    "message": "No materials found for assessment",
                },
                "file_results": [],
                "source_agent": "validator-agent",
            }

        request = ValidationRequest(
            workflow_id=workflow_id,
            assessment_id=assessment_id,
            assessor_id=body.get("assessor_id"),
            validation_type=validation_type,
            files=file_infos,
        )
        os.environ["CURRENT_WORKFLOW_ID"] = workflow_id
        response = await service.validate(request)

        await _write_material_decisions(
            material_validation=material_validation,
            assessment_id=assessment_id,
            response=response,
            file_to_material_id=file_to_material_id,
        )

        return {
            "workflow_id": workflow_id,
            "assessment_id": assessment_id,
            "terminal_signal": response.terminal_signal.model_dump(),
            "file_results": [
                {"file_name": r.file_name, "terminal_signal": r.terminal_signal.model_dump()}
                for r in response.file_results
            ],
            "source_agent": "validator-agent",
        }

    return application


async def _load_files_for_validation(
    *,
    material_validation: MaterialValidationPort,
    assessment_id: str,
    validation_type: str,
) -> tuple[list[FileInfo], dict[str, str]]:
    """Fetch materials via Submission Service gRPC and map to FileInfo.

    Returns ``(file_infos, file_name_to_material_id)``. The mapping lets
    the caller write a per-material decision back via
    ``UpdateMaterialValidation`` after ``ValidatorService.validate``
    returns.
    """
    upload_dir = os.environ.get("UPLOAD_DIR", "/tmp/assessorflow-uploads")
    # Phase 5 (web_research_validation) only processes new web-research
    # materials; the rest of Phase 3's materials were validated earlier.
    source_filter = (
        "web_research" if validation_type == "web_research_validation" else None
    )

    file_infos: list[FileInfo] = []
    file_to_material_id: dict[str, str] = {}
    try:
        # unvalidated_only=true skips previously-PROCEEDED materials so
        # we don't re-run the pipeline on them (mirrors the old
        # API-Server filter but enforced server-side).
        materials = await material_validation.get_materials(
            assessment_id=assessment_id,
            unvalidated_only=(validation_type == "material_validation"),
            source=source_filter,
        )
    except Exception:
        logger.error(
            "materials_fetch_failed",
            assessment_id=assessment_id,
            validation_type=validation_type,
            exc_info=True,
        )
        return [], {}

    for material in materials:
        storage_path = material.storage_path
        if storage_path.startswith("gs://"):
            resolved_path = storage_path
        elif material.source == "web_research" and storage_path:
            resolved_path = os.path.join(upload_dir, storage_path)
        else:
            resolved_path = os.path.join(
                upload_dir, f"{material.material_id}_{material.file_name}",
            )

        file_infos.append(
            FileInfo(
                file_name=material.file_name,
                storage_path=resolved_path,
                file_type=material.file_type or "pdf",
            )
        )
        if material.material_id:
            file_to_material_id[material.file_name] = material.material_id

    return file_infos, file_to_material_id


# -- UpdateMaterialValidation helpers -----------------------------------

# TerminalSignal.status -> readiness_status string (proto field).
_READINESS_MAP: dict[TerminalSignalStatus, str] = {
    TerminalSignalStatus.PROCEED: "PROCEED",
    TerminalSignalStatus.PROCEED_WITH_WARNINGS: "PARTIAL",
    TerminalSignalStatus.TERMINATE: "REJECT",
}


async def _write_material_decisions(
    *,
    material_validation: MaterialValidationPort,
    assessment_id: str,
    response: Any,
    file_to_material_id: dict[str, str],
) -> None:
    """Write an ``UpdateMaterialValidation`` RPC per file in the response.

    Fire-and-forget at the collection level -- individual RPC failures
    are logged but do not block the Pub/Sub / HTTP reply. Materials not
    present in ``file_to_material_id`` (e.g. files synthesized by tests
    without a backing DB row) are skipped.
    """
    for file_result in response.file_results:
        material_id = file_to_material_id.get(file_result.file_name)
        if not material_id:
            continue
        signal = file_result.terminal_signal
        readiness_status = _READINESS_MAP.get(signal.status, "REJECT")
        try:
            await material_validation.update_material_validation(
                assessment_id=assessment_id,
                material_id=material_id,
                readiness_status=readiness_status,
                validation_reason_code=signal.reason_code.value,
                validation_message=signal.message or "",
            )
        except Exception:
            logger.warning(
                "material_validation_write_failed",
                assessment_id=assessment_id,
                material_id=material_id,
                file_name=file_result.file_name,
                exc_info=True,
            )


app = create_app()
