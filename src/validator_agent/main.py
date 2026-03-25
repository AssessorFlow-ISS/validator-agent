"""FastAPI application entry point for the Validator Agent (#12).

Wires up port implementations based on environment configuration and
registers the API routes.  The ValidatorService is stored on app.state
so routes can access it via request.app.state.
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter
from validator_agent.api.routes import router
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


def _build_service(config: ValidatorConfig) -> ValidatorService:
    """Construct the ValidatorService with adapter implementations.

    Currently only stub adapters are implemented.  Real adapters (gRPC,
    Vertex AI, Pub/Sub, GCS) will be added when external service
    contracts are ready for integration.
    """
    # -- MRC adapter --------------------------------------------------------
    if config.mrc_adapter == "stub":
        mrc = StubMrcAdapter()
    else:
        raise ValueError(f"Unknown MRC_ADAPTER: {config.mrc_adapter}")

    # -- OCR adapter --------------------------------------------------------
    if config.ocr_adapter == "stub":
        ocr = StubOcrAdapter()
    else:
        raise ValueError(f"Unknown OCR_ADAPTER: {config.ocr_adapter}")

    # -- Model Broker adapter -----------------------------------------------
    if config.llm_provider == "stub":
        model_broker = StubModelBrokerAdapter()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.llm_provider}")

    # -- Knowledge Service adapter ------------------------------------------
    if config.knowledge_adapter == "stub":
        knowledge_service = StubKnowledgeServiceAdapter()
    else:
        raise ValueError(f"Unknown KNOWLEDGE_ADAPTER: {config.knowledge_adapter}")

    # -- Decision Audit adapter ---------------------------------------------
    if config.audit_adapter == "stub":
        decision_audit = StubDecisionAuditAdapter()
    else:
        raise ValueError(f"Unknown AUDIT_ADAPTER: {config.audit_adapter}")

    # -- Event Publisher adapter --------------------------------------------
    if config.event_adapter == "stub":
        event_publisher = StubEventPublisherAdapter()
    else:
        raise ValueError(f"Unknown EVENT_ADAPTER: {config.event_adapter}")

    # -- Storage adapter ----------------------------------------------------
    if config.storage_adapter == "stub":
        storage = StubStorageAdapter()
    else:
        raise ValueError(f"Unknown STORAGE_ADAPTER: {config.storage_adapter}")

    content_safety = ContentSafetyReasoner(model_broker=model_broker)

    return ValidatorService(
        mrc=mrc,
        ocr=ocr,
        content_safety=content_safety,
        knowledge_service=knowledge_service,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        storage=storage,
    )


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    config = ValidatorConfig.from_env()
    logger.info("starting_validator_agent", config=config)

    application = FastAPI(
        title="Validator Agent",
        description="Centralized content safety gate for AssessorFlow (#12)",
        version="0.1.0",
    )

    application.state.validator_service = _build_service(config)
    application.include_router(router)

    return application


app = create_app()
