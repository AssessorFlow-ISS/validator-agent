"""Unit tests for SubmissionClient (gRPC).

Patches the gRPC stub factory so no real channel is opened.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from validator_agent.clients.submission_client import SubmissionClient


def _fake_stub_with_get_materials(response: object | Exception):
    stub = MagicMock()
    if isinstance(response, Exception):
        stub.GetMaterials = AsyncMock(side_effect=response)
    else:
        stub.GetMaterials = AsyncMock(return_value=response)
    stub.UpdateMaterialValidation = AsyncMock(return_value=MagicMock())
    return stub


def _unavailable_error() -> grpc.aio.AioRpcError:
    exc = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="pod gone",
    )
    return exc


def _invalid_argument_error() -> grpc.aio.AioRpcError:
    exc = grpc.aio.AioRpcError(
        code=grpc.StatusCode.INVALID_ARGUMENT,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="bad request",
    )
    return exc


class TestSubmissionClientLifecycle:
    async def test_close_is_idempotent(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        # Close before any RPC — channel is None, nothing to close
        await client.close()
        # Now simulate an RPC → channel created → close again
        fake_channel = MagicMock()
        fake_channel.close = AsyncMock()
        client._channel = fake_channel
        client._stub = MagicMock()
        await client.close()
        fake_channel.close.assert_awaited_once()
        assert client._channel is None

    async def test_ensure_stub_reuses_existing(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        with patch(
            "validator_agent.clients.submission_client.grpc.aio.insecure_channel"
        ) as mock_channel:
            mock_channel.return_value = MagicMock()
            stub1 = client._ensure_stub()
            stub2 = client._ensure_stub()
            assert stub1 is stub2
            mock_channel.assert_called_once()


class TestGetMaterialsRetry:
    async def test_retry_on_unavailable_then_success(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        resp_ok = MagicMock()
        stub = MagicMock()
        stub.GetMaterials = AsyncMock(side_effect=[_unavailable_error(), resp_ok])
        client._stub = stub
        client._channel = MagicMock()

        with patch(
            "validator_agent.clients.submission_client.asyncio.sleep",
            new=AsyncMock(),
        ):
            out = await client.get_materials(assessment_id="a-1")
        assert out is resp_ok
        assert stub.GetMaterials.await_count == 2

    async def test_no_retry_on_domain_error(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        stub = _fake_stub_with_get_materials(_invalid_argument_error())
        client._stub = stub
        client._channel = MagicMock()

        with pytest.raises(grpc.aio.AioRpcError):
            await client.get_materials(assessment_id="a-1")
        assert stub.GetMaterials.await_count == 1

    async def test_exhaust_retries_raises(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        stub = MagicMock()
        stub.GetMaterials = AsyncMock(side_effect=_unavailable_error())
        client._stub = stub
        client._channel = MagicMock()

        with patch(
            "validator_agent.clients.submission_client.asyncio.sleep",
            new=AsyncMock(),
        ):
            with pytest.raises(grpc.aio.AioRpcError):
                await client.get_materials(assessment_id="a-1")
        assert stub.GetMaterials.await_count == 3


class TestUpdateMaterialValidation:
    async def test_builds_request_and_returns_response(self) -> None:
        client = SubmissionClient(grpc_url="localhost:1")
        resp_ok = MagicMock()
        stub = MagicMock()
        stub.UpdateMaterialValidation = AsyncMock(return_value=resp_ok)
        client._stub = stub
        client._channel = MagicMock()

        out = await client.update_material_validation(
            assessment_id="a-1",
            material_id="m-1",
            readiness_status="PROCEED",
            validation_reason_code="VALIDATION_PASSED",
            validation_message="ok",
        )
        assert out is resp_ok
        stub.UpdateMaterialValidation.assert_awaited_once()
