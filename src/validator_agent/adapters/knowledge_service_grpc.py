"""KnowledgeServiceGrpcAdapter — real gRPC client for Knowledge Service.

Bridges the Validator's KnowledgeServicePort to the real Knowledge Service
gRPC endpoint (ProcessMaterial RPC).

Fire-and-forget: the Validator Agent does not block on chunking completion.
Exceptions are caught and logged so validation flow is never blocked by
a Knowledge Service outage.

Environment Variables:
    KNOWLEDGE_SERVICE_GRPC_URL: host:port (default: localhost:9002)
        In-cluster: knowledge-service.af-data.svc.cluster.local:9002
"""

from __future__ import annotations

import structlog

from validator_agent.clients.knowledge_client import KnowledgeClient
from validator_agent.ports.knowledge_service_port import KnowledgeServicePort

logger = structlog.get_logger(__name__)


class KnowledgeServiceGrpcAdapter(KnowledgeServicePort):
    """gRPC adapter calling the real Knowledge Service."""

    def __init__(self, *, client: KnowledgeClient | None = None) -> None:
        self._client = client or KnowledgeClient()

    async def process_material(
        self,
        *,
        workflow_id: str,
        content_text: str,
        source_type: str,
        assessment_id: str | None = None,
        assessor_id: str | None = None,
        source: str = "upload",
    ) -> list[str]:
        """Send extracted text to Knowledge Service via gRPC ProcessMaterial.

        Returns list of chunk IDs on success, empty list on failure.
        Fire-and-forget: logs a warning on failure but never raises.
        """
        try:
            logger.info(
                "knowledge_service_grpc_request",
                workflow_id=workflow_id,
                source_type=source_type,
                content_length=len(content_text),
            )

            response = await self._client.process_material(
                workflow_id=workflow_id,
                content_text=content_text,
                source_type=source_type,
                assessment_id=assessment_id or "",
                assessor_id=assessor_id or "",
            )

            chunk_ids = list(response.chunk_ids)
            logger.info(
                "knowledge_service_grpc_response",
                workflow_id=workflow_id,
                chunks_created=response.chunks_created,
                status=response.status,
                chunk_ids_count=len(chunk_ids),
            )
            return chunk_ids

        except Exception as exc:
            logger.warning(
                "knowledge_service_grpc_error",
                workflow_id=workflow_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    async def close(self) -> None:
        """Gracefully close the gRPC channel."""
        await self._client.close()
