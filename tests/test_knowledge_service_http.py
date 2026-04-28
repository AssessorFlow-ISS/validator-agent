"""Unit tests for KnowledgeServiceHttpAdapter.

Mocks the httpx client to cover the happy path, HTTP error, connection error,
and unexpected error branches — all fire-and-forget, never raising.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from validator_agent.adapters.knowledge_service_http import (
    KnowledgeServiceHttpAdapter,
)

_BASE_URL = "http://test-knowledge:8020"
_POST_URL = f"{_BASE_URL}/api/v1/internal/process-material"


def _req() -> httpx.Request:
    return httpx.Request("POST", _POST_URL)


@pytest.fixture
def adapter() -> KnowledgeServiceHttpAdapter:
    return KnowledgeServiceHttpAdapter(base_url=_BASE_URL)


class TestKnowledgeServiceHttpAdapter:
    """HTTP client for Knowledge Service — fire-and-forget semantics."""

    async def test_returns_chunk_ids_on_success(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "material_id": "mat-1",
                "chunks_created": 3,
                "chunk_ids": ["c-1", "c-2", "c-3"],
            },
            request=_req(),
        )
        with patch.object(
            adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            chunk_ids = await adapter.process_material(
                workflow_id="wf-1",
                content_text="sample text",
                source_type="ocr_extracted",
            )

            assert chunk_ids == ["c-1", "c-2", "c-3"]

    async def test_includes_optional_fields_when_provided(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            200, json={"chunk_ids": []}, request=_req()
        )
        with patch.object(
            adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as post_mock:
            await adapter.process_material(
                workflow_id="wf-2",
                content_text="body",
                source_type="ocr_extracted",
                assessment_id="a-1",
                assessor_id="u-1",
                source="web_research",
            )

            body = post_mock.call_args[1]["json"]
            assert body["workflow_id"] == "wf-2"
            assert body["assessment_id"] == "a-1"
            assert body["assessor_id"] == "u-1"
            assert body["source"] == "web_research"

    async def test_omits_optional_fields_when_none(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            200, json={"chunk_ids": []}, request=_req()
        )
        with patch.object(
            adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as post_mock:
            await adapter.process_material(
                workflow_id="wf-3",
                content_text="body",
                source_type="ocr_extracted",
            )

            body = post_mock.call_args[1]["json"]
            assert "assessment_id" not in body
            assert "assessor_id" not in body

    async def test_returns_empty_on_http_status_error(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            500, json={"detail": "boom"}, request=_req()
        )
        with patch.object(
            adapter._client, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            # Fire-and-forget: never raises, returns []
            chunk_ids = await adapter.process_material(
                workflow_id="wf-err",
                content_text="body",
                source_type="ocr_extracted",
            )
            assert chunk_ids == []

    async def test_returns_empty_on_connection_error(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connect refused"),
        ):
            chunk_ids = await adapter.process_material(
                workflow_id="wf-conn",
                content_text="body",
                source_type="ocr_extracted",
            )
            assert chunk_ids == []

    async def test_returns_empty_on_unexpected_error(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ):
            chunk_ids = await adapter.process_material(
                workflow_id="wf-unknown",
                content_text="body",
                source_type="ocr_extracted",
            )
            assert chunk_ids == []

    async def test_close_closes_client(
        self, adapter: KnowledgeServiceHttpAdapter
    ) -> None:
        with patch.object(
            adapter._client, "aclose", new_callable=AsyncMock
        ) as aclose:
            await adapter.close()
            aclose.assert_awaited_once()

    def test_default_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://custom:9999")
        adapter = KnowledgeServiceHttpAdapter()
        assert adapter._base_url == "http://custom:9999"
