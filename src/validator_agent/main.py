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


def _build_service(config: ValidatorConfig) -> tuple[ValidatorService, Any]:
    """Construct the ValidatorService with adapter implementations.

    Returns (service, event_publisher) — event_publisher is returned
    separately so the lifespan can set up Pub/Sub subscriptions.
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
    knowledge_service = StubKnowledgeServiceAdapter()

    # -- Decision Audit adapter ---------------------------------------------
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
    storage = StubStorageAdapter()

    content_safety = ContentSafetyReasoner(model_broker=model_broker)

    service = ValidatorService(
        mrc=mrc,
        ocr=ocr,
        content_safety=content_safety,
        knowledge_service=knowledge_service,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        storage=storage,
    )

    return service, event_publisher


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    config = ValidatorConfig.from_env()
    logger.info("starting_validator_agent", config=config)

    service, event_publisher = _build_service(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        """Startup/shutdown lifecycle — subscribe to Pub/Sub on startup."""
        if config.event_adapter in ("pubsub", "real", "emulator"):
            from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

            if isinstance(event_publisher, PubSubEventPublisherAdapter):
                async def handle_trigger(payload: dict) -> None:
                    """Process a validation trigger from Pub/Sub."""
                    logger.info("validation_trigger_received", payload_keys=list(payload.keys()))

                    files = payload.get("files", [])
                    file_infos = [
                        FileInfo(
                            file_name=f.get("file_name", "unknown"),
                            storage_path=f.get("storage_path", ""),
                            file_type=f.get("file_type", "pdf"),
                        )
                        for f in files
                    ] if files else [
                        FileInfo(file_name="default.pdf", storage_path="gs://stub/default.pdf", file_type="pdf")
                    ]

                    request = ValidationRequest(
                        workflow_id=payload.get("workflow_id", "unknown"),
                        assessment_id=payload.get("assessment_id", "unknown"),
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
