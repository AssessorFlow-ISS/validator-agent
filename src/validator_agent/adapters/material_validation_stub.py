"""StubMaterialValidationAdapter -- in-memory stub for tests/local dev.

Mirrors :class:`GrpcMaterialValidationAdapter` without any network calls.
Pre-seed ``materials`` via the constructor; validation writes are
recorded in ``updates`` for test assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from validator_agent.ports.material_validation_port import (
    Material,
    MaterialValidationPort,
)


@dataclass
class MaterialValidationUpdate:
    """A single recorded update_material_validation call."""

    assessment_id: str
    material_id: str
    readiness_status: str
    validation_reason_code: str
    validation_message: str


@dataclass
class StubMaterialValidationAdapter(MaterialValidationPort):
    """In-memory stub. Pre-seed ``materials`` and inspect ``updates``."""

    materials: list[Material] = field(default_factory=list)
    updates: list[MaterialValidationUpdate] = field(default_factory=list)

    async def get_materials(
        self,
        *,
        assessment_id: str,
        unvalidated_only: bool = False,
        source: str | None = None,
    ) -> list[Material]:
        result = list(self.materials)
        if unvalidated_only:
            result = [m for m in result if m.readiness_status in ("", "pending")]
        if source:
            result = [m for m in result if m.source == source]
        return result

    async def update_material_validation(
        self,
        *,
        assessment_id: str,
        material_id: str,
        readiness_status: str,
        validation_reason_code: str = "",
        validation_message: str = "",
    ) -> None:
        self.updates.append(
            MaterialValidationUpdate(
                assessment_id=assessment_id,
                material_id=material_id,
                readiness_status=readiness_status,
                validation_reason_code=validation_reason_code,
                validation_message=validation_message,
            )
        )
