"""Decision Audit Pub/Sub adapter — publishes audit events to Decision Audit Service.

Publishes to assessorflow.audit.decision topic. The Decision Audit Service
(L-11) subscribes and persists to PostgreSQL (append-only agent_decision_log).

Fire-and-forget: exceptions are logged but never propagate to the pipeline.

Environment Variables:
    PUBSUB_PROJECT_ID: GCP project ID (default: accessorflow)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from validator_agent.ports.decision_audit_port import DecisionAuditPort

logger = structlog.get_logger(__name__)

_AUDIT_DECISION_TOPIC = "assessorflow.audit.decision"


class PubSubDecisionAuditAdapter(DecisionAuditPort):
    """Publishes audit decisions to Pub/Sub for the Decision Audit Service."""

    def __init__(self, project_id: str | None = None) -> None:
        self._project_id = project_id or os.environ.get(
            "PUBSUB_PROJECT_ID", "accessorflow"
        )
        self._publisher = None

    def _ensure_publisher(self):
        if self._publisher is None:
            from google.cloud import pubsub_v1
            self._publisher = pubsub_v1.PublisherClient()
        return self._publisher

    async def log_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish decision to assessorflow.audit.decision — fire-and-forget."""
        try:
            publisher = self._ensure_publisher()
            topic_path = publisher.topic_path(self._project_id, _AUDIT_DECISION_TOPIC)

            envelope = {
                "event_id": str(uuid.uuid4()),
                "event_type": "audit.decision",
                "workflow_id": workflow_id,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "source_agent": agent_name,
                "correlation_id": workflow_id,
                "payload": {
                    "workflow_id": workflow_id,
                    "agent_name": agent_name,
                    "decision_type": decision_type,
                    **payload,
                },
            }

            message_data = json.dumps(envelope).encode("utf-8")
            future = publisher.publish(
                topic_path,
                data=message_data,
                workflow_id=workflow_id,
                source_agent=agent_name,
            )
            await asyncio.to_thread(future.result, timeout=10)

            logger.info(
                "audit_decision_published",
                workflow_id=workflow_id,
                agent_name=agent_name,
                decision_type=decision_type,
                topic=_AUDIT_DECISION_TOPIC,
            )

        except Exception:
            logger.warning(
                "audit_decision_publish_failed",
                workflow_id=workflow_id,
                agent_name=agent_name,
                decision_type=decision_type,
                exc_info=True,
            )
