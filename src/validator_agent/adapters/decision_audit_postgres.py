"""PostgresDecisionAuditAdapter — asyncpg implementation of DecisionAuditPort.

Writes validation decisions directly to the af_decision_audit.agent_decision_log
table. Used for local dev and integration testing against the local Cloud SQL
proxy (localhost:15432).

Fire-and-forget: exceptions are logged as warnings but never propagate to the
validation pipeline (Invariant #5 — audit failures must not break the agent).
"""
from __future__ import annotations

import json
import os
from typing import Any

import asyncpg
import structlog

from validator_agent.ports.decision_audit_port import DecisionAuditPort

logger = structlog.get_logger(__name__)


class PostgresDecisionAuditAdapter(DecisionAuditPort):
    """Async PostgreSQL adapter for the Decision Audit Service audit trail."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]

    def _dsn(self) -> str:
        """Build the PostgreSQL DSN from environment variables."""
        host = os.getenv("AUDIT_DB_HOST", "localhost")
        port = os.getenv("AUDIT_DB_PORT", "15432")
        name = os.getenv("AUDIT_DB_NAME", "af_decision_audit")
        user = os.getenv("AUDIT_DB_USER", "assessorflow")
        password = os.getenv("AUDIT_DB_PASSWORD", "dev_password")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        """Lazy-initialise the connection pool on first use."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn(),
                min_size=1,
                max_size=5,
            )
            logger.info("audit_pg_pool_created", dsn=self._dsn().replace(
                os.getenv("AUDIT_DB_PASSWORD", "dev_password"), "***",
            ))
        return self._pool

    async def close(self) -> None:
        """Close the connection pool (call on shutdown)."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("audit_pg_pool_closed")

    async def log_decision(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        decision_type: str,
        payload: dict[str, Any],
    ) -> None:
        """INSERT a decision row — fire-and-forget, never raises."""
        try:
            pool = await self._ensure_pool()

            sql = """
                INSERT INTO agent_decision_log (
                    workflow_id, agent_name, decision_type,
                    input_summary, output_summary, reasoning_steps,
                    confidence_score, prompt_version, model_id,
                    grounding_sources
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """

            # Extract structured fields from the payload, mirroring the
            # column layout in the agent_decision_log table.
            input_summary = payload.get("input_summary")
            output_summary = payload.get("output_summary")
            reasoning_steps = payload.get("reasoning_steps")
            confidence_score = payload.get("confidence_score")
            prompt_version = payload.get("prompt_version")
            model_id = payload.get("model_id")
            grounding_sources = payload.get("grounding_sources")

            await pool.execute(
                sql,
                workflow_id,
                agent_name,
                decision_type,
                json.dumps(input_summary) if input_summary is not None else None,
                json.dumps(output_summary) if output_summary is not None else None,
                json.dumps(reasoning_steps) if reasoning_steps is not None else None,
                float(confidence_score) if confidence_score is not None else None,
                prompt_version,
                model_id,
                json.dumps(grounding_sources) if grounding_sources is not None else None,
            )

            logger.info(
                "audit_decision_logged",
                workflow_id=workflow_id,
                agent_name=agent_name,
                decision_type=decision_type,
            )

        except Exception:
            # Fire-and-forget — log the failure but never break the pipeline.
            logger.warning(
                "audit_decision_log_failed",
                workflow_id=workflow_id,
                agent_name=agent_name,
                decision_type=decision_type,
                exc_info=True,
            )
