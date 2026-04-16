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
        # Cumulative stats for test reporting
        self.request_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._per_model: dict[str, dict] = {}
        self.last_model_used: str = "unknown"
        self.last_model_tier: str = "unknown"

    async def generate(
        self,
        *,
        prompt: str,
        model_tier: str,
        response_format: str | None = None,
        response_schema: dict | None = None,
    ) -> str:
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
        if response_format:
            request_body["response_format"] = response_format
        if response_schema:
            request_body["response_schema"] = response_schema

        logger.info(
            "model_broker_request",
            task_key=task_key,
            model_tier=model_tier,
            prompt_length=len(prompt),
        )

        response = await self._client.post("/api/v1/generate", json=request_body)
        response.raise_for_status()
        data = response.json()

        # Accumulate stats
        usage = data.get("token_usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        cost = data.get("cost_usd", 0.0)
        self.request_count += 1
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok
        self.total_tokens += p_tok + c_tok
        self.total_cost_usd += cost
        self.last_model_used = data.get("model_used", "unknown")
        self.last_model_tier = data.get("model_tier", "unknown")

        pm = self._per_model.setdefault(self.last_model_used, {
            "request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0, "tests_passed": 0, "tests_failed": 0,
        })
        pm["request_count"] += 1
        pm["prompt_tokens"] += p_tok
        pm["completion_tokens"] += c_tok
        pm["total_tokens"] += p_tok + c_tok
        pm["cost_usd"] += cost

        logger.info(
            "model_broker_response",
            model=self.last_model_used,
            tier=self.last_model_tier,
            latency_ms=data.get("latency_ms"),
        )

        return data["content"]

    async def close(self) -> None:
        await self._client.aclose()
