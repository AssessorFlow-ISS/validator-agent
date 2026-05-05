"""Real Pub/Sub event publisher for the Validator Agent.

Publishes completion events to assessorflow.validation.complete (Topic #3)
for the Orchestrator to consume.

Also provides subscribe() for receiving trigger events from
assessorflow.validation.trigger (Topic #2).

Inherits all publish/subscribe/envelope logic from the shared
AgentPubSubSubscriber base class.
"""

from __future__ import annotations

from typing import Any

from validator_agent.adapters.pubsub_subscriber import AgentPubSubSubscriber

from validator_agent.ports.event_publisher_port import EventPublisherPort


class PubSubEventPublisherAdapter(AgentPubSubSubscriber, EventPublisherPort):
    """Real Pub/Sub adapter for the Validator Agent.

    Inherits publish(), subscribe_and_process(), envelope wrapping, and
    poll-based subscription from AgentPubSubSubscriber. Only the agent
    name differs.
    """

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__(
            agent_name="validator-agent",
            project_id=project_id,
        )

    async def publish(self, *, topic: str, payload: dict[str, Any]) -> None:
        """Publish with keyword-only args matching the Validator port interface."""
        await super().publish(topic, payload)
