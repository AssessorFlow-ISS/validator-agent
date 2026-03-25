"""EventPublisherPort — abstract interface for Pub/Sub event publishing.

The Validator Agent publishes completion events to
assessorflow.validation.complete (Topic #3) for the Orchestrator to
consume and resume the workflow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventPublisherPort(ABC):
    """Port for publishing events to Pub/Sub."""

    @abstractmethod
    async def publish(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish an event to a Pub/Sub topic.

        Args:
            topic: The Pub/Sub topic name.
            payload: The event payload to publish.
        """
