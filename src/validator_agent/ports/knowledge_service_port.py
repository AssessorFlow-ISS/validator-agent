"""KnowledgeServicePort — abstract interface for the Knowledge Service.

On PROCEED, the Validator Agent forwards extracted text to the Knowledge
Service via ProcessMaterial (gRPC 3.2.1) for chunking and embedding.
The Knowledge Service stores chunks in the Document KB (pgvector).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class KnowledgeServicePort(ABC):
    """Port for forwarding validated content to the Knowledge Service (#3)."""

    @abstractmethod
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
        """Send extracted text to the Knowledge Service for chunking/embedding.

        Args:
            workflow_id: The workflow identifier.
            content_text: Pre-extracted text content from OCR or web research.
            source_type: How the text was extracted (e.g. "ocr_extracted", "web_research").
            assessment_id: Assessment scoping for cross-context isolation.
            assessor_id: CR-RAG-001 — assessor identity for cumulative knowledge.
            source: Material source ("upload" for Phase 3, "web_research" for Phase 5).

        Returns:
            List of chunk IDs created by the Knowledge Service.
            Empty list on failure (fire-and-forget).
        """
