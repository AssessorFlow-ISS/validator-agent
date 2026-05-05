"""Knowledge Service gRPC client.

Manages the gRPC channel lifecycle and provides typed methods
for the ProcessMaterial RPC. Follows the same pattern as
SubmissionClient for consistency.

Environment variables:
    KNOWLEDGE_SERVICE_GRPC_URL: host:port (default: localhost:9002)
"""

from __future__ import annotations

import os

import grpc
import structlog

from validator_agent._grpc import knowledge_pb2, knowledge_pb2_grpc

logger = structlog.get_logger(__name__)

_DEFAULT_URL = "localhost:9002"


class KnowledgeClient:
    """gRPC client for the Knowledge Service."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.environ.get(
            "KNOWLEDGE_SERVICE_GRPC_URL", _DEFAULT_URL
        )
        self._channel = grpc.aio.insecure_channel(self._url)
        self._stub = knowledge_pb2_grpc.KnowledgeServiceStub(self._channel)
        logger.info("knowledge_client_init", url=self._url)

    async def process_material(
        self,
        *,
        workflow_id: str,
        content_text: str,
        source_type: str,
        assessment_id: str = "",
        assessor_id: str = "",
        source_file: str = "",
    ) -> knowledge_pb2.ProcessMaterialResponse:
        """Call ProcessMaterial RPC."""
        request = knowledge_pb2.ProcessMaterialRequest(
            workflow_id=workflow_id,
            content_text=content_text,
            source_type=source_type,
            assessment_id=assessment_id,
            assessor_id=assessor_id,
            source_file=source_file,
        )
        return await self._stub.ProcessMaterial(request, timeout=30.0)

    async def close(self) -> None:
        """Gracefully close the gRPC channel."""
        await self._channel.close()
