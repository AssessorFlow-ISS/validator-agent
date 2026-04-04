"""KnowledgeServiceHttpAdapter — real HTTP client for Knowledge Service.

Bridges the Validator's KnowledgeServicePort to the real Knowledge Service
FastAPI endpoint (POST /api/v1/internal/process-material).

Fire-and-forget: the Validator Agent does not wait for chunking to complete.
Exceptions are caught and logged as warnings so validation flow is never
blocked by a Knowledge Service outage.

Environment Variables:
    KNOWLEDGE_SERVICE_URL: Base URL (default: http://localhost:8020)
"""
from __future__ import annotations

import os

import httpx
import structlog

from validator_agent.ports.knowledge_service_port import KnowledgeServicePort

logger = structlog.get_logger(__name__)


class KnowledgeServiceHttpAdapter(KnowledgeServicePort):
    """HTTP client adapter calling the real Knowledge Service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or os.environ.get(
            "KNOWLEDGE_SERVICE_URL", "http://localhost:8020"
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def process_material(
        self,
        *,
        workflow_id: str,
        content_text: str,
        source_type: str,
        assessor_id: str | None = None,
    ) -> None:
        """POST extracted text to Knowledge Service for chunking/embedding.

        Fire-and-forget: logs a warning on failure but never raises, so the
        validation pipeline is not blocked by Knowledge Service issues.
        """
        request_body: dict[str, str] = {
            "workflow_id": workflow_id,
            "content_text": content_text,
            "source_type": source_type,
        }
        if assessor_id is not None:
            request_body["assessor_id"] = assessor_id

        try:
            logger.info(
                "knowledge_service_request",
                workflow_id=workflow_id,
                source_type=source_type,
                content_length=len(content_text),
            )

            response = await self._client.post(
                "/api/v1/internal/process-material",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            logger.info(
                "knowledge_service_response",
                workflow_id=workflow_id,
                material_id=data.get("material_id"),
                chunks_created=data.get("chunks_created"),
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "knowledge_service_http_error",
                workflow_id=workflow_id,
                status_code=exc.response.status_code,
                detail=exc.response.text[:500],
            )
        except httpx.RequestError as exc:
            logger.warning(
                "knowledge_service_connection_error",
                workflow_id=workflow_id,
                error=str(exc),
            )
        except Exception as exc:
            logger.warning(
                "knowledge_service_unexpected_error",
                workflow_id=workflow_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        await self._client.aclose()
