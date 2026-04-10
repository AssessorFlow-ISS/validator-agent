"""DecisionAuditPort — abstract interface for the Decision Audit Service.

Every validation decision is logged to the immutable audit trail
(Invariant #5) via the Decision Audit Service (L-11, gRPC fire-and-forget).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DecisionAuditPort(ABC):
    """Port for logging decisions to the Decision Audit Service (L-11)."""

    @abstractmethod
    async def log_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Log a validation decision to the audit trail.

        This is a fire-and-forget call — failures are logged but do not
        block the validation response.

        Args:
            workflow_id: The workflow identifier.
            agent_name: Name of the agent making the decision.
            decision_type: Type of decision (e.g. "content_validation").
            payload: Full decision payload including reasoning_steps, terminal_signal, etc.
        """
