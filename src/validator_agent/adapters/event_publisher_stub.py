"""StubEventPublisherAdapter — records published events for test assertion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validator_agent.ports.event_publisher_port import EventPublisherPort


@dataclass
class PublishedEvent:
    """Record of a single publish invocation."""

    topic: str
    payload: dict[str, Any]


class StubEventPublisherAdapter(EventPublisherPort):
    """Stub adapter that records published events for test inspection."""

    def __init__(self) -> None:
        self.events: list[PublishedEvent] = []

    async def publish(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """Record the event without publishing."""
        self.events.append(PublishedEvent(
            topic=topic,
            payload=payload,
        ))
