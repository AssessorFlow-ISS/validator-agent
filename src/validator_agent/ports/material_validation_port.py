"""MaterialValidationPort -- abstract interface for reading materials and
writing validation decisions via the Submission Service.

The Validator Agent reads assessment materials (by default only those
still pending validation) and writes a decision per material after the
MRC + OCR + Content Safety pipeline. Phase 6C.1 routes both flows
through the Submission Service gRPC API (``GetMaterials`` and
``UpdateMaterialValidation``), replacing the previous direct-DB /
HTTP-to-API-Server pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Domain representation of a single assessment material row."""

    material_id: str
    file_name: str
    storage_path: str
    file_type: str
    readiness_status: str
    source: str
    source_url: str


class MaterialValidationPort(ABC):
    """Port for reading materials and writing validation decisions."""

    @abstractmethod
    async def get_materials(
        self,
        *,
        assessment_id: str,
        unvalidated_only: bool = False,
        source: str | None = None,
    ) -> list[Material]:
        """Fetch materials for an assessment.

        Args:
            assessment_id: Assessment to read materials for.
            unvalidated_only: If true, return only materials whose
                readiness_status is still ``pending`` (or the service-side
                equivalent). Used in Phase 3 to avoid re-validating
                previously-PROCEEDED materials.
            source: Optional filter by source (e.g. ``"web_research"``)
                for Phase 5 web-research validation. Applied client-side
                against the Material.source field.
        """

    @abstractmethod
    async def update_material_validation(
        self,
        *,
        assessment_id: str,
        material_id: str,
        readiness_status: str,
        validation_reason_code: str = "",
        validation_message: str = "",
    ) -> None:
        """Write a validation decision for a single material.

        Args:
            assessment_id: Scoping for the material row.
            material_id: Which material to update.
            readiness_status: PROCEED / REJECT / PARTIAL.
            validation_reason_code: Short analytics code (e.g.
                ``VALIDATION_PASSED``, ``OCR_LOW_CONFIDENCE``).
            validation_message: Free-text audit message.
        """

    async def close(self) -> None:  # pragma: no cover -- default no-op
        """Release resources (connection pools, gRPC channels, etc.)."""
        return None
