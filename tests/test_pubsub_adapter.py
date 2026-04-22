"""Tests for the PubSubEventPublisherAdapter.

Uses mocked google.cloud.pubsub_v1 clients to test publish, subscribe,
and error handling without a real Pub/Sub emulator.

After the refactor to inherit from AgentPubSubSubscriber, the __new__
pattern must also set _agent_name (set by the base __init__).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch



def _make_adapter():
    """Create a PubSubEventPublisherAdapter bypassing __init__ for unit tests."""
    from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

    adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
    adapter._agent_name = "validator-agent"
    adapter._project_id = "test-project"
    adapter._emulator_host = "localhost:18085"
    adapter._poll_tasks = []

    mock_publisher = MagicMock()
    mock_publisher.topic_path.return_value = "projects/test/topics/t"
    mock_future = MagicMock()
    mock_future.result.return_value = "msg-1"
    mock_publisher.publish.return_value = mock_future
    adapter._publisher = mock_publisher

    adapter._subscriber = MagicMock()
    adapter._subscriber.subscription_path.return_value = "projects/test/subscriptions/test.sub"
    return adapter


class TestPubSubPublish:
    """Tests for the publish() method."""

    async def test_publish_sends_json_payload(self) -> None:
        adapter = _make_adapter()

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value="msg-123"):
            await adapter.publish(
                topic="test.topic",
                payload={"workflow_id": "wf-1", "data": "hello"},
            )

        adapter._publisher.publish.assert_called_once()
        call_args = adapter._publisher.publish.call_args
        envelope = json.loads(call_args[1]["data"].decode("utf-8"))
        # Envelope-level fields
        assert envelope["workflow_id"] == "wf-1"
        assert envelope["event_type"] == "topic"
        assert envelope["source_agent"] == "validator-agent"
        assert "event_id" in envelope
        assert "timestamp" in envelope
        assert "correlation_id" in envelope
        # Inner payload preserved
        assert envelope["payload"]["workflow_id"] == "wf-1"
        assert envelope["payload"]["data"] == "hello"
        # Message attributes still present
        assert call_args[1]["workflow_id"] == "wf-1"
        assert call_args[1]["source_agent"] == "validator-agent"

    async def test_publish_accepts_keyword_args(self) -> None:
        """Verify publish works with keyword-only args (port interface)."""
        adapter = _make_adapter()

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value="msg-456"):
            # Keyword call -- matches the port interface signature
            await adapter.publish(topic="my.topic", payload={"workflow_id": "wf-kw"})
        assert adapter._publisher.publish.called


class TestPubSubSubscribe:
    """Tests for subscribe_and_process()."""

    async def test_poll_task_stored_to_prevent_gc(self) -> None:
        adapter = _make_adapter()

        # Make pull raise immediately so the loop doesn't block
        adapter._subscriber.pull.side_effect = Exception("test stop")

        handler = AsyncMock()
        await adapter.subscribe_and_process("test.sub", handler)

        assert len(adapter._poll_tasks) == 1
        assert isinstance(adapter._poll_tasks[0], asyncio.Task)

        # Clean up the task
        adapter._poll_tasks[0].cancel()
        try:
            await adapter._poll_tasks[0]
        except (asyncio.CancelledError, Exception):
            pass

    async def test_handler_called_on_message(self) -> None:
        """Verify the handler is called when a message arrives."""
        adapter = _make_adapter()

        payload = {"workflow_id": "wf-handler-test", "assessment_id": "a-1"}
        msg = MagicMock()
        msg.message.data = json.dumps(payload).encode("utf-8")
        msg.ack_id = "ack-1"

        pull_response = MagicMock()
        pull_response.received_messages = [msg]

        # First pull returns message, second raises to stop loop
        call_count = 0
        def pull_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pull_response
            raise asyncio.CancelledError()

        adapter._subscriber.pull.side_effect = pull_side_effect

        handler = AsyncMock()
        await adapter.subscribe_and_process("test.sub", handler)

        # Give the poll loop time to process
        await asyncio.sleep(0.2)

        handler.assert_called_once_with(payload)
        adapter._subscriber.acknowledge.assert_called_once()

        # Clean up
        for t in adapter._poll_tasks:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def test_nack_on_handler_error(self) -> None:
        """Verify message is nacked when handler raises."""
        adapter = _make_adapter()

        payload = {"workflow_id": "wf-nack"}
        msg = MagicMock()
        msg.message.data = json.dumps(payload).encode("utf-8")
        msg.ack_id = "ack-nack"

        pull_response = MagicMock()
        pull_response.received_messages = [msg]

        call_count = 0
        def pull_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pull_response
            raise asyncio.CancelledError()

        adapter._subscriber.pull.side_effect = pull_side_effect

        handler = AsyncMock(side_effect=ValueError("handler failed"))
        await adapter.subscribe_and_process("test.sub", handler)

        await asyncio.sleep(0.2)

        # Handler was called but failed
        handler.assert_called_once()
        # Message should NOT be acked
        adapter._subscriber.acknowledge.assert_not_called()
        # Message should be nacked (modify_ack_deadline to 0)
        adapter._subscriber.modify_ack_deadline.assert_called_once()
        nack_args = adapter._subscriber.modify_ack_deadline.call_args[1]["request"]
        assert nack_args["ack_deadline_seconds"] == 0

        for t in adapter._poll_tasks:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
