"""Tests for the PubSubEventPublisherAdapter.

Uses mocked google.cloud.pubsub_v1 clients to test publish, subscribe,
and error handling without a real Pub/Sub emulator.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestPubSubPublish:
    """Tests for the publish() method."""

    async def test_publish_sends_json_payload(self) -> None:
        with patch.dict("os.environ", {"PUBSUB_EMULATOR_HOST": "localhost:18085"}):
            with patch("validator_agent.adapters.pubsub_event_publisher.pubsub_v1", create=True) as mock_pubsub:
                # Mock the clients
                mock_publisher = MagicMock()
                mock_subscriber = MagicMock()
                mock_pubsub.PublisherClient.return_value = mock_publisher
                mock_pubsub.SubscriberClient.return_value = mock_subscriber

                mock_publisher.topic_path.return_value = "projects/test/topics/test.topic"
                mock_future = MagicMock()
                mock_future.result.return_value = "msg-123"
                mock_publisher.publish.return_value = mock_future

                from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

                # Build adapter with mocked internals
                adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
                adapter._project_id = "test-project"
                adapter._emulator_host = "localhost:18085"
                adapter._publisher = mock_publisher
                adapter._subscriber = mock_subscriber
                adapter._poll_tasks = []

                # Patch asyncio.to_thread since publish uses it for future.result
                with patch("asyncio.to_thread", new_callable=AsyncMock, return_value="msg-123"):
                    await adapter.publish(
                        topic="test.topic",
                        payload={"workflow_id": "wf-1", "data": "hello"},
                    )

                mock_publisher.publish.assert_called_once()
                call_args = mock_publisher.publish.call_args
                data = json.loads(call_args[1]["data"].decode("utf-8"))
                assert data["workflow_id"] == "wf-1"
                assert data["data"] == "hello"
                assert call_args[1]["workflow_id"] == "wf-1"
                assert call_args[1]["source_agent"] == "validator-agent"

    async def test_publish_accepts_keyword_args(self) -> None:
        """Verify publish works with keyword-only args (port interface)."""
        with patch.dict("os.environ", {"PUBSUB_EMULATOR_HOST": "localhost:18085"}):
            from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

            adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
            adapter._project_id = "test-project"
            adapter._emulator_host = "localhost:18085"
            adapter._poll_tasks = []

            mock_publisher = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "msg-456"
            mock_publisher.publish.return_value = mock_future
            mock_publisher.topic_path.return_value = "projects/test/topics/t"
            adapter._publisher = mock_publisher
            adapter._subscriber = MagicMock()

            # Keyword call — matches the port interface signature
            with patch("asyncio.to_thread", new_callable=AsyncMock, return_value="msg-456"):
                await adapter.publish(topic="my.topic", payload={"workflow_id": "wf-kw"})
            assert mock_publisher.publish.called


class TestPubSubSubscribe:
    """Tests for subscribe_and_process()."""

    async def test_poll_task_stored_to_prevent_gc(self) -> None:
        from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

        adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
        adapter._project_id = "test-project"
        adapter._emulator_host = "localhost:18085"
        adapter._poll_tasks = []

        mock_subscriber = MagicMock()
        mock_subscriber.subscription_path.return_value = "projects/test/subscriptions/test.sub"
        # Make pull raise immediately so the loop doesn't block
        mock_subscriber.pull.side_effect = Exception("test stop")
        adapter._subscriber = mock_subscriber

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
        from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

        adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
        adapter._project_id = "test-project"
        adapter._emulator_host = "localhost:18085"
        adapter._poll_tasks = []

        payload = {"workflow_id": "wf-handler-test", "assessment_id": "a-1"}
        msg = MagicMock()
        msg.message.data = json.dumps(payload).encode("utf-8")
        msg.ack_id = "ack-1"

        pull_response = MagicMock()
        pull_response.received_messages = [msg]

        mock_subscriber = MagicMock()
        mock_subscriber.subscription_path.return_value = "projects/test/subscriptions/test.sub"

        # First pull returns message, second raises to stop loop
        call_count = 0
        def pull_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pull_response
            raise asyncio.CancelledError()

        mock_subscriber.pull.side_effect = pull_side_effect
        adapter._subscriber = mock_subscriber

        handler = AsyncMock()

        await adapter.subscribe_and_process("test.sub", handler)

        # Give the poll loop time to process
        await asyncio.sleep(0.2)

        handler.assert_called_once_with(payload)
        mock_subscriber.acknowledge.assert_called_once()

        # Clean up
        for t in adapter._poll_tasks:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def test_nack_on_handler_error(self) -> None:
        """Verify message is nacked when handler raises."""
        from validator_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter

        adapter = PubSubEventPublisherAdapter.__new__(PubSubEventPublisherAdapter)
        adapter._project_id = "test-project"
        adapter._emulator_host = "localhost:18085"
        adapter._poll_tasks = []

        payload = {"workflow_id": "wf-nack"}
        msg = MagicMock()
        msg.message.data = json.dumps(payload).encode("utf-8")
        msg.ack_id = "ack-nack"

        pull_response = MagicMock()
        pull_response.received_messages = [msg]

        mock_subscriber = MagicMock()
        mock_subscriber.subscription_path.return_value = "projects/test/subscriptions/test.sub"

        call_count = 0
        def pull_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pull_response
            raise asyncio.CancelledError()

        mock_subscriber.pull.side_effect = pull_side_effect
        adapter._subscriber = mock_subscriber

        handler = AsyncMock(side_effect=ValueError("handler failed"))

        await adapter.subscribe_and_process("test.sub", handler)

        await asyncio.sleep(0.2)

        # Handler was called but failed
        handler.assert_called_once()
        # Message should NOT be acked
        mock_subscriber.acknowledge.assert_not_called()
        # Message should be nacked (modify_ack_deadline to 0)
        mock_subscriber.modify_ack_deadline.assert_called_once()
        nack_args = mock_subscriber.modify_ack_deadline.call_args[1]["request"]
        assert nack_args["ack_deadline_seconds"] == 0

        for t in adapter._poll_tasks:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
