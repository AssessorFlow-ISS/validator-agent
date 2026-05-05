"""GrpcMaterialValidationAdapter -- Submission Service gRPC (Phase 6C.1).

Thin mapping layer between the agent's domain-oriented
:class:`MaterialValidationPort` interface and the
:class:`SubmissionClient`. All cross-cutting concerns (retry, timeouts,
channel lifecycle, logging) live in the client so this file stays
focused on translation: proto messages <-> agent dataclasses.

Environment variables:
    SUBMISSION_SERVICE_GRPC_URL: host:port of the submission service gRPC
        endpoint (required in production). Defaults to ``localhost:9001``
        for local dev. In-cluster value:
        ``submission-service.af-submission.svc.cluster.local:9001``.
"""

from __future__ import annotations

import structlog

from validator_agent.clients.submission_client import SubmissionClient
from validator_agent.ports.material_validation_port import (
    Material,
    MaterialValidationPort,
)

logger = structlog.get_logger(__name__)


class GrpcMaterialValidationAdapter(MaterialValidationPort):
    """Submission Service gRPC adapter for material read + validation write."""

    def __init__(
        self,
        *,
        client: SubmissionClient | None = None,
    ) -> None:
        # Allow tests to inject a preconfigured client; otherwise we
        # build the default one that reads SUBMISSION_SERVICE_GRPC_URL.
        self._client = client or SubmissionClient()

    async def close(self) -> None:
        await self._client.close()

    async def get_materials(
        self,
        *,
        assessment_id: str,
        unvalidated_only: bool = False,
        source: str | None = None,
    ) -> list[Material]:
        response = await self._client.get_materials(
            assessment_id=assessment_id,
            unvalidated_only=unvalidated_only,
        )
        materials = [_from_proto_material(m) for m in response.materials]
        if source:
            materials = [m for m in materials if m.source == source]
        logger.info(
            "materials_fetched",
            assessment_id=assessment_id,
            unvalidated_only=unvalidated_only,
            source=source,
            count=len(materials),
        )
        return materials

    async def update_material_validation(
        self,
        *,
        assessment_id: str,
        material_id: str,
        readiness_status: str,
        validation_reason_code: str = "",
        validation_message: str = "",
    ) -> None:
        response = await self._client.update_material_validation(
            assessment_id=assessment_id,
            material_id=material_id,
            readiness_status=readiness_status,
            validation_reason_code=validation_reason_code,
            validation_message=validation_message,
        )
        logger.info(
            "material_validation_updated",
            assessment_id=assessment_id,
            material_id=material_id,
            readiness_status=response.readiness_status,
            status=response.status,
        )


# -- Proto <-> dataclass mapping helpers ----------------------------------


def _from_proto_material(material_info: object) -> Material:
    """Map a ``MaterialInfo`` proto message onto the domain dataclass."""
    return Material(
        material_id=material_info.material_id,  # type: ignore[attr-defined]
        file_name=material_info.file_name,  # type: ignore[attr-defined]
        storage_path=material_info.storage_path,  # type: ignore[attr-defined]
        file_type=material_info.file_type,  # type: ignore[attr-defined]
        readiness_status=material_info.readiness_status,  # type: ignore[attr-defined]
        source=material_info.source,  # type: ignore[attr-defined]
        source_url=material_info.source_url,  # type: ignore[attr-defined]
    )
