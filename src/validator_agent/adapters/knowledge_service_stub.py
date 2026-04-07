"""StubKnowledgeServiceAdapter — records calls for test assertion.

Stores all process_material calls in a list so tests can verify that
validated content was (or was not) forwarded to the Knowledge Service.
"""
from __future__ import annotations

from dataclasses import dataclass

from validator_agent.ports.knowledge_service_port import KnowledgeServicePort


@dataclass
class ProcessMaterialCall:
    """Record of a single process_material invocation."""

    workflow_id: str
    content_text: str
    source_type: str
    assessment_id: str | None = None
    assessor_id: str | None = None


class StubKnowledgeServiceAdapter(KnowledgeServicePort):
    """Stub adapter that records process_material calls for test inspection."""

    def __init__(self) -> None:
        self.calls: list[ProcessMaterialCall] = []

    async def process_material(
        self,
        *,
        workflow_id: str,
        content_text: str,
        source_type: str,
        assessment_id: str | None = None,
        assessor_id: str | None = None,
    ) -> list[str]:
        """Record the call and return stub chunk IDs."""
        self.calls.append(ProcessMaterialCall(
            workflow_id=workflow_id,
            content_text=content_text,
            source_type=source_type,
            assessment_id=assessment_id,
            assessor_id=assessor_id,
        ))
        return [f"stub-chunk-{i}" for i in range(1, 4)]
