"""Real Pub/Sub event publisher for the Validator Agent.

Publishes completion events to assessorflow.validation.complete (Topic #3)
for the Orchestrator to consume.

Also provides subscribe() for receiving trigger events from
assessorflow.validation.trigger (Topic #2).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from validator_agent.ports.event_publisher_port import EventPublisherPort

logger = structlog.get_logger(__name__)


class PubSubEventPublisherAdapter(EventPublisherPort):
    """Real Pub/Sub adapter for the Validator Agent."""

    def __init__(
        self,
        project_id: str | None = None,
        emulator_host: str | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get(
            "PUBSUB_PROJECT_ID", "assessorflow-local"
        )
        self._emulator_host = emulator_host or os.environ.get(
            "PUBSUB_EMULATOR_HOST", "localhost:18085"
        )
        if self._emulator_host:
            os.environ["PUBSUB_EMULATOR_HOST"] = self._emulator_host

        from google.cloud import pubsub_v1

        self._publisher = pubsub_v1.PublisherClient()
        self._subscriber = pubsub_v1.SubscriberClient()
        self._poll_tasks: list[asyncio.Task] = []

    @staticmethod
    def _build_envelope(
        event_type: str, payload: dict[str, Any], source_agent: str,
    ) -> dict[str, Any]:
        """Wrap a flat payload in the standard Pub/Sub envelope (pubsub.md 5.1)."""
        workflow_id = payload.get("workflow_id", "unknown")
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "workflow_id": workflow_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "source_agent": source_agent,
            "correlation_id": payload.get("workflow_id", str(uuid.uuid4())),
            "payload": payload,
        }

    @staticmethod
    def _derive_event_type(topic: str) -> str:
        """Derive event_type from topic name.

        e.g. 'assessorflow.validation.complete' -> 'validation.complete'
        """
        parts = topic.split(".")
        return ".".join(parts[1:]) if len(parts) > 1 else topic

    async def publish(self, *, topic: str, payload: dict[str, Any]) -> None:
        """Publish a completion event to Pub/Sub.

        Automatically wraps the payload in the standard Pub/Sub envelope
        per pubsub.md Section 5.1.
        """
        topic_path = self._publisher.topic_path(self._project_id, topic)

        event_type = self._derive_event_type(topic)
        envelope = self._build_envelope(event_type, payload, "validator-agent")
        message_data = json.dumps(envelope).encode("utf-8")

        workflow_id = envelope["workflow_id"]

        logger.info(
            "pubsub_publish",
            topic=topic,
            workflow_id=workflow_id,
            event_type=event_type,
        )

        future = self._publisher.publish(
            topic_path,
            data=message_data,
            workflow_id=str(workflow_id),
            source_agent="validator-agent",
        )
        message_id = await asyncio.to_thread(future.result, timeout=10)

        logger.info("pubsub_published", topic=topic, message_id=message_id)

    async def subscribe_and_process(
        self,
        subscription: str,
        handler: Any,
    ) -> None:
        """Subscribe to a topic via polling and process messages.

        Uses periodic synchronous pull instead of streaming pull for
        better reliability with the Pub/Sub emulator.
        """
        subscription_path = self._subscriber.subscription_path(
            self._project_id, subscription
        )

        async def _poll_loop() -> None:
            """Background task that polls for messages every 2 seconds."""
            from google.cloud.pubsub_v1 import types as pubsub_types

            while True:
                try:
                    response = await asyncio.to_thread(
                        self._subscriber.pull,
                        request=pubsub_types.PullRequest(
                            subscription=subscription_path,
                            max_messages=1,
                        ),
                        timeout=5.0,
                    )
                    for msg in response.received_messages:
                        raw = json.loads(msg.message.data.decode("utf-8"))
                        # Unwrap standard envelope — pass inner payload to handler
                        payload = raw.get("payload", raw)
                        logger.info(
                            "pubsub_received",
                            subscription=subscription,
                            envelope_event_type=raw.get("event_type"),
                            workflow_id=payload.get("workflow_id", "?"),
                        )
                        try:
                            await handler(payload)
                            self._subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": [msg.ack_id],
                                }
                            )
                        except Exception:
                            logger.exception("pubsub_handler_error")
                            try:
                                self._subscriber.modify_ack_deadline(
                                    request={
                                        "subscription": subscription_path,
                                        "ack_ids": [msg.ack_id],
                                        "ack_deadline_seconds": 0,
                                    }
                                )
                            except Exception:
                                pass
                except Exception as exc:
                    exc_name = type(exc).__name__
                    if exc_name not in ("DeadlineExceeded", "_InactiveRpcError"):
                        logger.warning("pubsub_poll_error", error=str(exc), error_type=exc_name)

        task = asyncio.create_task(_poll_loop())
        self._poll_tasks.append(task)
        logger.info("pubsub_subscribed", subscription=subscription)
