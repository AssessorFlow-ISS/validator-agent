"""Real Model Broker HTTP client for the Validator Agent.

Bridges the Validator's ModelBrokerPort (generate(prompt, model_tier))
to the real Model Broker FastAPI service (POST /api/v1/generate).

Architecture Invariant #6: All LLM calls go through Model Broker.
"""

from __future__ import annotations

import os

import httpx
import structlog

from validator_agent.ports.model_broker_port import ModelBrokerPort

logger = structlog.get_logger(__name__)


class ModelBrokerHttpAdapter(ModelBrokerPort):
    """HTTP client adapter calling the real Model Broker service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url or os.environ.get(
            "MODEL_BROKER_URL", "http://localhost:8010"
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def generate(self, *, prompt: str, model_tier: str) -> str:
        """Call Model Broker and return the LLM response content.

        Maps the Validator's (prompt, model_tier) interface to the
        Model Broker's full request schema.
        """
        # Map model_tier to a task_key the Model Broker understands
        task_key = (
            "validator.content_safety_reasoning"
            if model_tier == "HIGH"
            else "validator.mrc_interpretation"
        )

        request_body = {
            "task_key": task_key,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.2,
            "session_id": os.environ.get("CURRENT_WORKFLOW_ID", "unknown"),
            "agent_id": "validator-agent",
            "prompt_version": "validator/content_safety@v1",
        }

        logger.info(
            "model_broker_request",
            task_key=task_key,
            model_tier=model_tier,
            prompt_length=len(prompt),
        )

        response = await self._client.post("/api/v1/generate", json=request_body)
        response.raise_for_status()
        data = response.json()

        logger.info(
            "model_broker_response",
            model=data.get("model_used"),
            tier=data.get("model_tier"),
            latency_ms=data.get("latency_ms"),
        )

        return data["content"]

    async def close(self) -> None:
        await self._client.aclose()
