"""Base Pub/Sub adapter for stateless agents.

Provides publish/subscribe/envelope logic for Google Cloud Pub/Sub.
Always connects to real GCP Pub/Sub (emulator host is stripped if set).

Inlined from af_shared.pubsub.agent_subscriber — no external dependency.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AgentPubSubSubscriber:
    """Reusable Pub/Sub publish + subscribe base for stateless agents."""

    def __init__(self, *, agent_name: str, project_id: str | None = None) -> None:
        self._agent_name = agent_name
        self._project_id = project_id or os.environ.get("PUBSUB_PROJECT_ID", "accessorflow")

        if "PUBSUB_EMULATOR_HOST" in os.environ:
            del os.environ["PUBSUB_EMULATOR_HOST"]

        from google.cloud import pubsub_v1
        self._publisher = pubsub_v1.PublisherClient()
        self._subscriber = pubsub_v1.SubscriberClient()
        self._poll_tasks: list[asyncio.Task[None]] = []

    @staticmethod
    def _build_envelope(event_type: str, payload: dict[str, Any], source_agent: str) -> dict[str, Any]:
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
        parts = topic.split(".")
        return ".".join(parts[1:]) if len(parts) > 1 else topic

    async def publish(self, topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        topic_path = self._publisher.topic_path(self._project_id, topic)
        event_type = self._derive_event_type(topic)
        envelope = self._build_envelope(event_type, payload, self._agent_name)
        message_data = json.dumps(envelope).encode("utf-8")
        workflow_id = envelope["workflow_id"]

        logger.info("pubsub_publish", topic=topic, workflow_id=workflow_id, event_type=event_type)
        future = self._publisher.publish(
            topic_path, data=message_data,
            workflow_id=str(workflow_id), source_agent=self._agent_name,
        )
        message_id = await asyncio.to_thread(future.result, timeout=10)
        logger.info("pubsub_published", topic=topic, message_id=message_id)

    async def subscribe_and_process(self, subscription: str, handler: Any) -> None:
        subscription_path = self._subscriber.subscription_path(self._project_id, subscription)
        topic_name = subscription.removesuffix(".sub")
        topic_path = self._publisher.topic_path(self._project_id, topic_name)

        try:
            await asyncio.to_thread(self._publisher.create_topic, request={"name": topic_path})
        except Exception:
            pass
        try:
            await asyncio.to_thread(
                self._subscriber.create_subscription,
                request={"name": subscription_path, "topic": topic_path},
            )
        except Exception:
            pass

        logger.info("pubsub_subscribed", subscription=subscription, agent=self._agent_name)

        async def _poll_loop() -> None:
            from google.cloud.pubsub_v1 import types as pubsub_types
            while True:
                try:
                    response = await asyncio.to_thread(
                        self._subscriber.pull,
                        request=pubsub_types.PullRequest(subscription=subscription_path, max_messages=1),
                        timeout=5.0,
                    )
                    for msg in response.received_messages:
                        publish_time = msg.message.publish_time
                        if publish_time:
                            age_s = (datetime.now(timezone.utc) - publish_time).total_seconds()
                            if age_s > 300:
                                logger.warning("pubsub_stale_message_skipped", age_seconds=round(age_s))
                                self._subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": [msg.ack_id]})
                                continue

                        raw = json.loads(msg.message.data.decode("utf-8"))
                        payload = raw.get("payload", raw)
                        logger.info("pubsub_received", subscription=subscription, workflow_id=payload.get("workflow_id", "?"))
                        try:
                            await handler(payload)
                            self._subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": [msg.ack_id]})
                        except Exception:
                            logger.exception("pubsub_handler_error")
                            try:
                                self._subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": [msg.ack_id]})
                            except Exception:
                                pass
                except Exception as exc:
                    exc_name = type(exc).__name__
                    if exc_name not in ("DeadlineExceeded", "_InactiveRpcError"):
                        logger.warning("pubsub_poll_error", error=str(exc), error_type=exc_name)

        task = asyncio.create_task(_poll_loop())
        self._poll_tasks.append(task)

    async def stop(self) -> None:
        for task in self._poll_tasks:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_tasks.clear()
