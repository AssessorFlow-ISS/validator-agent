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
from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter
from af_shared.adapters.factory import get_tracing
from validator_agent.api.routes import router
from validator_agent.api.schemas import FileInfo, ValidationRequest
from validator_agent.config import ValidatorConfig
from validator_agent.domain.content_safety import ContentSafetyReasoner
from validator_agent.domain.services import ValidatorService

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


def _build_service(config: ValidatorConfig) -> tuple[ValidatorService, Any, Any, Any]:
    """Construct the ValidatorService with adapter implementations.

    Returns (service, event_publisher, decision_audit, knowledge_service) —
    event_publisher is returned separately so the lifespan can set up Pub/Sub
    subscriptions; decision_audit and knowledge_service are returned for
    graceful pool/client shutdown.
    """
    # -- MRC adapter --------------------------------------------------------
    mrc = StubMrcAdapter()

    # -- OCR adapter --------------------------------------------------------
    ocr = StubOcrAdapter()

    # -- Model Broker adapter -----------------------------------------------
    if config.llm_provider == "stub":
        model_broker = StubModelBrokerAdapter()
    elif config.llm_provider in ("http", "google_ai_studio", "vertex_ai"):
        from validator_agent.adapters.model_broker_http import ModelBrokerHttpAdapter
        model_broker = ModelBrokerHttpAdapter()
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

    return service, event_publisher, decision_audit, knowledge_service


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    config = ValidatorConfig.from_env()
    logger.info("starting_validator_agent", config=config)

    service, event_publisher, decision_audit, knowledge_service = _build_service(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        """Startup/shutdown lifecycle — subscribe to Pub/Sub on startup."""
        if config.event_adapter in ("pubsub", "real", "emulator"):
            from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

            if isinstance(event_publisher, PubSubEventPublisherAdapter):
                async def handle_trigger(payload: dict) -> None:
                    """Process a validation trigger from Pub/Sub.

                    The Orchestrator dispatches {workflow_id, assessment_id, phase, agent}.
                    This handler fetches material metadata from the API Server
                    (GET /api/v1/assessments/{assessment_id}/materials) and builds
                    FileInfo objects for the validation pipeline.
                    """
                    import httpx

                    workflow_id = payload.get("workflow_id", "unknown")
                    assessment_id = payload.get("assessment_id", "unknown")
                    logger.info(
                        "validation_trigger_received",
                        workflow_id=workflow_id,
                        assessment_id=assessment_id,
                        payload_keys=list(payload.keys()),
                    )

                    # Fetch materials from API Server — the Orchestrator trigger
                    # does NOT include files; we must look them up by assessment_id.
                    # For Phase 5 (web_research_validation), filter by source=web_research
                    # to only validate new web research content, not re-process Phase 3 materials.
                    api_base = os.environ.get("API_SERVER_URL", "http://localhost:8001")
                    upload_dir = os.environ.get("UPLOAD_DIR", "/tmp/assessorflow-uploads")
                    validation_type = payload.get("validation_type", "material_validation")
                    file_infos: list[FileInfo] = []
                    try:
                        materials_url = f"{api_base}/api/v1/assessments/{assessment_id}/materials"
                        if validation_type == "web_research_validation":
                            materials_url += "?source=web_research"
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            resp = await client.get(materials_url)
                            resp.raise_for_status()
                            materials = resp.json()
                        logger.info("materials_fetched", assessment_id=assessment_id, count=len(materials), validation_type=validation_type)

                        for m in materials:
                            material_id = m.get("id", "")
                            file_name = m.get("file_name", "unknown")
                            source = m.get("source", "upload")
                            storage_path = m.get("storage_path", "")

                            # Web research .md files are stored at {UPLOAD_DIR}/materials/{workflow_id}/{file_name}
                            # by the Web Research Agent's LocalStorageAdapter.
                            if source == "web_research" and storage_path:
                                local_path = os.path.join(upload_dir, storage_path)
                            else:
                                # Original uploads: API Server stores as {id}_{file_name}
                                local_path = os.path.join(upload_dir, f"{material_id}_{file_name}")

                            file_infos.append(
                                FileInfo(
                                    file_name=file_name,
                                    storage_path=local_path,
                                    file_type=m.get("file_type", "pdf"),
                                )
                            )
                    except Exception:
                        logger.error(
                            "materials_fetch_failed",
                            assessment_id=assessment_id,
                            exc_info=True,
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
                        validation_type=payload.get("validation_type", "material_validation"),
                        files=file_infos,
                    )

                    os.environ["CURRENT_WORKFLOW_ID"] = request.workflow_id
                    response = await service.validate(request)

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

    return application


app = create_app()
