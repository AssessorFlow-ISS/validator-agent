"""StubDecisionAuditAdapter — records audit log calls for test assertion.

Stores all log_decision calls so tests can verify the immutable audit
trail is being populated correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validator_agent.ports.decision_audit_port import DecisionAuditPort


@dataclass
class AuditLogEntry:
    """Record of a single log_decision invocation."""

    workflow_id: str
    agent_name: str
    decision_type: str
    payload: dict[str, Any]


class StubDecisionAuditAdapter(DecisionAuditPort):
    """Stub adapter that records audit log entries for test inspection."""

    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    async def log_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Record the audit log entry without persisting."""
        self.entries.append(AuditLogEntry(
            workflow_id=workflow_id,
            agent_name=agent_name,
            decision_type=decision_type,
            payload=payload,
        ))
