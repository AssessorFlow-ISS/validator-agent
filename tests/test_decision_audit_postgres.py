"""Unit tests for PostgresDecisionAuditAdapter.

Mocks asyncpg so we never touch a real database. Covers DSN construction,
lazy pool init, log_decision happy path, fire-and-forget exception swallow,
and close().
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from validator_agent.adapters.decision_audit_postgres import (
    PostgresDecisionAuditAdapter,
)


class TestDsnBuilding:
    """DSN construction from env vars."""

    def test_prefers_full_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIT_DATABASE_URL", "postgresql://u:p@h:1/d")
        adapter = PostgresDecisionAuditAdapter()
        assert adapter._dsn() == "postgresql://u:p@h:1/d"

    def test_builds_from_parts_when_no_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUDIT_DATABASE_URL", raising=False)
        monkeypatch.setenv("AUDIT_DB_HOST", "pgdb")
        monkeypatch.setenv("AUDIT_DB_PORT", "6000")
        monkeypatch.setenv("AUDIT_DB_NAME", "audit")
        monkeypatch.setenv("AUDIT_DB_USER", "user")
        monkeypatch.setenv("AUDIT_DB_PASSWORD", "pw")
        adapter = PostgresDecisionAuditAdapter()
        assert adapter._dsn() == "postgresql://user:pw@pgdb:6000/audit"

    def test_part_defaults_apply_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "AUDIT_DATABASE_URL",
            "AUDIT_DB_HOST",
            "AUDIT_DB_PORT",
            "AUDIT_DB_NAME",
            "AUDIT_DB_USER",
            "AUDIT_DB_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)
        adapter = PostgresDecisionAuditAdapter()
        dsn = adapter._dsn()
        assert "localhost" in dsn
        assert "15432" in dsn
        assert "af_decision_audit" in dsn


class TestPoolLifecycle:
    """Lazy pool initialization + close."""

    async def test_ensure_pool_creates_on_first_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = PostgresDecisionAuditAdapter()
        fake_pool = MagicMock()
        create_pool_mock = AsyncMock(return_value=fake_pool)

        with patch(
            "validator_agent.adapters.decision_audit_postgres.asyncpg.create_pool",
            create_pool_mock,
        ):
            pool1 = await adapter._ensure_pool()
            pool2 = await adapter._ensure_pool()

        # Pool is cached — only created once
        assert pool1 is fake_pool
        assert pool2 is fake_pool
        create_pool_mock.assert_awaited_once()

    async def test_close_releases_pool(self) -> None:
        adapter = PostgresDecisionAuditAdapter()
        fake_pool = MagicMock()
        fake_pool.close = AsyncMock()
        adapter._pool = fake_pool

        await adapter.close()

        fake_pool.close.assert_awaited_once()
        assert adapter._pool is None

    async def test_close_is_noop_when_no_pool(self) -> None:
        adapter = PostgresDecisionAuditAdapter()
        assert adapter._pool is None
        # Should not raise
        await adapter.close()


class TestLogDecision:
    """log_decision INSERT path + fire-and-forget swallow."""

    async def test_writes_row_with_serialized_json(self) -> None:
        adapter = PostgresDecisionAuditAdapter()
        fake_pool = MagicMock()
        fake_pool.execute = AsyncMock()
        adapter._pool = fake_pool

        payload = {
            "input_summary": {"files": ["a.pdf"]},
            "output_summary": {"status": "PROCEED"},
            "reasoning_steps": [{"step": 1, "action": "ok"}],
            "confidence_score": 0.9,
            "prompt_version": "validator@v1",
            "model_id": "gpt-4o",
            "grounding_sources": ["chunk-1"],
            "assessor_id": "user-1",
        }

        await adapter.log_decision(
            workflow_id="wf-1",
            agent_name="validator-agent",
            decision_type="content_validation",
            payload=payload,
        )

        fake_pool.execute.assert_awaited_once()
        args = fake_pool.execute.call_args[0]
        # SQL + 11 bound params
        assert "INSERT INTO agent_decision_log" in args[0]
        assert args[1] == "wf-1"
        assert args[2] == "validator-agent"
        assert args[3] == "content_validation"
        # JSON fields serialized as strings
        assert '"files"' in args[4]
        assert '"status"' in args[5]
        assert args[7] == 0.9  # confidence_score
        assert args[11] == "user-1"  # assessor_id

    async def test_none_payload_fields_passed_as_null(self) -> None:
        adapter = PostgresDecisionAuditAdapter()
        fake_pool = MagicMock()
        fake_pool.execute = AsyncMock()
        adapter._pool = fake_pool

        await adapter.log_decision(
            workflow_id="wf-2",
            agent_name="validator-agent",
            decision_type="content_validation",
            payload={},  # everything None
        )

        args = fake_pool.execute.call_args[0]
        # All JSON fields should be None
        assert args[4] is None  # input_summary
        assert args[5] is None  # output_summary
        assert args[6] is None  # reasoning_steps
        assert args[7] is None  # confidence_score
        assert args[10] is None  # grounding_sources

    async def test_swallows_pool_exception(self) -> None:
        """Fire-and-forget: pool failures must never propagate."""
        adapter = PostgresDecisionAuditAdapter()
        fake_pool = MagicMock()
        fake_pool.execute = AsyncMock(side_effect=RuntimeError("db down"))
        adapter._pool = fake_pool

        # Should not raise
        await adapter.log_decision(
            workflow_id="wf-3",
            agent_name="validator-agent",
            decision_type="content_validation",
            payload={"confidence_score": 0.5},
        )

    async def test_swallows_pool_init_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = PostgresDecisionAuditAdapter()

        create_pool_mock = AsyncMock(side_effect=RuntimeError("cannot connect"))
        with patch(
            "validator_agent.adapters.decision_audit_postgres.asyncpg.create_pool",
            create_pool_mock,
        ):
            # Should not raise despite pool init failure
            await adapter.log_decision(
                workflow_id="wf-4",
                agent_name="validator-agent",
                decision_type="content_validation",
                payload={},
            )
