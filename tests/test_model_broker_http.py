"""Tests for the ModelBrokerHttpAdapter.

Uses httpx mock transport to test HTTP request/response handling
without a real Model Broker service.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from validator_agent.adapters.model_broker_http import ModelBrokerHttpAdapter

_BROKER_URL = "http://test-broker:8010"
_GENERATE_URL = f"{_BROKER_URL}/api/v1/generate"


def _mock_request() -> httpx.Request:
    return httpx.Request("POST", _GENERATE_URL)


@pytest.fixture
def success_response() -> dict:
    return {
        "content": "No harmful content detected. Document is safe.",
        "model_used": "gemini-2.5-flash-lite",
        "model_tier": "CHEAP",
    }


@pytest.fixture
def adapter() -> ModelBrokerHttpAdapter:
    return ModelBrokerHttpAdapter(base_url=_BROKER_URL)


class TestModelBrokerHttpAdapter:
    """Tests for HTTP adapter bridging to Model Broker."""

    async def test_generate_sends_correct_request_high_tier(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.generate(
                prompt="Analyse this document for content safety",
                model_tier="HIGH",
            )

            adapter._client.post.assert_called_once()
            call_args = adapter._client.post.call_args
            assert call_args[0][0] == "/api/v1/generate"
            body = call_args[1]["json"]
            assert body["task_key"] == "validator.content_safety_reasoning"
            assert body["prompt"] == "Analyse this document for content safety"
            assert body["agent_id"] == "validator-agent"
            assert body["max_tokens"] == 65536
            assert body["temperature"] == 0.2

    async def test_generate_sends_cheap_tier_task_key(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.generate(
                prompt="Interpret MRC results",
                model_tier="CHEAP",
            )

            body = adapter._client.post.call_args[1]["json"]
            assert body["task_key"] == "validator.mrc_interpretation"

    async def test_generate_returns_content_string(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.generate(prompt="prompt", model_tier="CHEAP")
            assert result == success_response["content"]
            assert isinstance(result, str)

    async def test_generate_raises_on_http_error(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            500,
            json={"detail": "Internal Server Error"},
            request=httpx.Request("POST", f"{_BROKER_URL}/api/v1/generate"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.generate(prompt="prompt", model_tier="HIGH")

    async def test_generate_raises_on_connection_error(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await adapter.generate(prompt="prompt", model_tier="HIGH")

    async def test_prompt_version_format(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.generate(prompt="prompt", model_tier="HIGH")
            body = adapter._client.post.call_args[1]["json"]
            assert body["prompt_version"] == "validator/content_safety@v1"

    async def test_default_session_id(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.generate(prompt="prompt", model_tier="CHEAP")
            body = adapter._client.post.call_args[1]["json"]
            assert body["session_id"] == "unknown"

    async def test_session_id_from_env(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with patch.dict("os.environ", {"CURRENT_WORKFLOW_ID": "wf-env-test"}):
                await adapter.generate(prompt="prompt", model_tier="HIGH")
                body = adapter._client.post.call_args[1]["json"]
                assert body["session_id"] == "wf-env-test"

    async def test_close_closes_client(self, adapter: ModelBrokerHttpAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()

    def test_default_base_url(self) -> None:
        adapter = ModelBrokerHttpAdapter()
        assert "localhost:8010" in adapter._base_url
