"""Tests for stub adapter behavior."""

from __future__ import annotations

from pathlib import Path

from validator_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from validator_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from validator_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from validator_agent.adapters.mrc_stub import StubMrcAdapter
from validator_agent.adapters.ocr_stub import StubOcrAdapter
from validator_agent.adapters.storage_stub import StubStorageAdapter

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestStubMrcAdapter:
    async def test_default_behavior(self) -> None:
        adapter = StubMrcAdapter()
        result = await adapter.predict("any/path.pdf")

        assert result.readiness is True
        assert result.confidence == 0.95

    async def test_custom_defaults(self) -> None:
        adapter = StubMrcAdapter(default_readiness=False, default_confidence=0.3)
        result = await adapter.predict("any/path.pdf")

        assert result.readiness is False
        assert result.confidence == 0.3

    async def test_per_file_override(self) -> None:
        adapter = StubMrcAdapter(
            overrides={
                "bad.pdf": {"readiness": False, "confidence": 0.1},
            },
        )

        good_result = await adapter.predict("good.pdf")
        assert good_result.readiness is True

        bad_result = await adapter.predict("bad.pdf")
        assert bad_result.readiness is False
        assert bad_result.confidence == 0.1

    async def test_partial_override_uses_defaults(self) -> None:
        adapter = StubMrcAdapter(
            default_confidence=0.99,
            overrides={
                "partial.pdf": {"readiness": False},
            },
        )
        result = await adapter.predict("partial.pdf")
        assert result.readiness is False
        assert result.confidence == 0.99  # default confidence used


class TestStubOcrAdapter:
    async def test_default_text(self) -> None:
        adapter = StubOcrAdapter()
        result = await adapter.extract_text("any/path.pdf")

        assert len(result.text) > 50
        assert result.source_type == "ocr_extracted"

    async def test_custom_default_text(self) -> None:
        adapter = StubOcrAdapter(default_text="Custom text content for testing purposes here.")
        result = await adapter.extract_text("any/path.pdf")

        assert result.text == "Custom text content for testing purposes here."

    async def test_per_file_override(self) -> None:
        adapter = StubOcrAdapter(
            overrides={"special.pdf": "Overridden text for this specific file."},
        )
        result = await adapter.extract_text("special.pdf")
        assert result.text == "Overridden text for this specific file."

    async def test_fixture_file_lookup(self) -> None:
        adapter = StubOcrAdapter(fixture_dir=FIXTURES_DIR)
        result = await adapter.extract_text("some/path/clean_document.pdf")

        # Should match clean_document.txt in fixtures
        assert "Software Engineering" in result.text
        assert result.source_type == "ocr_extracted"

    async def test_fixture_not_found_falls_to_default(self) -> None:
        adapter = StubOcrAdapter(fixture_dir=FIXTURES_DIR, default_text="fallback text is here now")
        result = await adapter.extract_text("some/path/nonexistent.pdf")
        assert result.text == "fallback text is here now"

    async def test_override_takes_priority_over_fixture(self) -> None:
        adapter = StubOcrAdapter(
            fixture_dir=FIXTURES_DIR,
            overrides={"some/path/clean_document.pdf": "Override wins"},
        )
        result = await adapter.extract_text("some/path/clean_document.pdf")
        assert result.text == "Override wins"


class TestStubKnowledgeServiceAdapter:
    async def test_records_calls(self) -> None:
        adapter = StubKnowledgeServiceAdapter()
        await adapter.process_material(
            workflow_id="wf-1",
            content_text="some text",
            source_type="ocr_extracted",
        )

        assert len(adapter.calls) == 1
        assert adapter.calls[0].workflow_id == "wf-1"
        assert adapter.calls[0].content_text == "some text"

    async def test_multiple_calls_recorded(self) -> None:
        adapter = StubKnowledgeServiceAdapter()
        await adapter.process_material(workflow_id="wf-1", content_text="text1", source_type="ocr_extracted")
        await adapter.process_material(workflow_id="wf-2", content_text="text2", source_type="native_text")

        assert len(adapter.calls) == 2


class TestStubDecisionAuditAdapter:
    async def test_records_entries(self) -> None:
        adapter = StubDecisionAuditAdapter()
        await adapter.log_decision(
            workflow_id="wf-1",
            agent_name="validator-agent",
            decision_type="content_validation",
            payload={"key": "value"},
        )

        assert len(adapter.entries) == 1
        assert adapter.entries[0].agent_name == "validator-agent"


class TestStubEventPublisherAdapter:
    async def test_records_events(self) -> None:
        adapter = StubEventPublisherAdapter()
        await adapter.publish(
            topic="assessorflow.validation.complete",
            payload={"workflow_id": "wf-1"},
        )

        assert len(adapter.events) == 1
        assert adapter.events[0].topic == "assessorflow.validation.complete"


class TestStubStorageAdapter:
    async def test_default_bytes(self) -> None:
        adapter = StubStorageAdapter()
        data = await adapter.download_file("some/path.pdf")

        assert isinstance(data, bytes)
        assert len(data) > 0

    async def test_records_downloads(self) -> None:
        adapter = StubStorageAdapter()
        await adapter.download_file("path/a.pdf")
        await adapter.download_file("path/b.pdf")

        assert adapter.downloaded == ["path/a.pdf", "path/b.pdf"]

    async def test_per_file_override(self) -> None:
        adapter = StubStorageAdapter(
            overrides={"special.pdf": b"special-content"},
        )
        data = await adapter.download_file("special.pdf")
        assert data == b"special-content"
