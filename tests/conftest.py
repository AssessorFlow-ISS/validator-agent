"""Shared test fixtures for the Validator Agent test suite.

All fixtures use stub adapters — zero external dependencies required.
The stub_pipeline_fn (from adapters/pipeline_stub.py) replaces Thet's
real 3-component pipeline with a keyword-based stub that returns canned
ValidatorResult objects.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.adapters.pipeline_stub import stub_pipeline_fn
from af_shared.adapters.stubs.tracing_stub import StubTracingAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter
from validator_agent.domain.content_safety import ContentSafetyReasoner
from validator_agent.domain.services import ValidatorService
from validator_agent.main import create_app

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def clean_text() -> str:
    return (FIXTURES_DIR / "clean_document.txt").read_text(encoding="utf-8")


@pytest.fixture
def harmful_text() -> str:
    return (FIXTURES_DIR / "harmful_content.txt").read_text(encoding="utf-8")


@pytest.fixture
def pii_text() -> str:
    return (FIXTURES_DIR / "pii_content.txt").read_text(encoding="utf-8")


@pytest.fixture
def stub_mrc() -> StubMrcAdapter:
    return StubMrcAdapter()


@pytest.fixture
def stub_ocr(clean_text: str) -> StubOcrAdapter:
    return StubOcrAdapter(default_text=clean_text)


@pytest.fixture
def stub_model_broker() -> StubModelBrokerAdapter:
    return StubModelBrokerAdapter()


@pytest.fixture
def stub_knowledge_service() -> StubKnowledgeServiceAdapter:
    return StubKnowledgeServiceAdapter()


@pytest.fixture
def stub_decision_audit() -> StubDecisionAuditAdapter:
    return StubDecisionAuditAdapter()


@pytest.fixture
def stub_event_publisher() -> StubEventPublisherAdapter:
    return StubEventPublisherAdapter()


@pytest.fixture
def stub_storage() -> StubStorageAdapter:
    return StubStorageAdapter()


@pytest.fixture
def content_safety_reasoner(
    stub_model_broker: StubModelBrokerAdapter,
) -> ContentSafetyReasoner:
    return ContentSafetyReasoner(model_broker=stub_model_broker)


@pytest.fixture
def validator_service(
    stub_knowledge_service: StubKnowledgeServiceAdapter,
    stub_decision_audit: StubDecisionAuditAdapter,
    stub_event_publisher: StubEventPublisherAdapter,
    stub_storage: StubStorageAdapter,
) -> ValidatorService:
    return ValidatorService(
        knowledge_service=stub_knowledge_service,
        decision_audit=stub_decision_audit,
        event_publisher=stub_event_publisher,
        storage=stub_storage,
        tracing=StubTracingAdapter(),
        pipeline_fn=stub_pipeline_fn,
    )


@pytest.fixture
async def api_client() -> AsyncClient:
    """HTTP test client using the real FastAPI app with stub adapters."""
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
