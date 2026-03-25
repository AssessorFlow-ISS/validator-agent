"""ModelBrokerPort — abstract interface for LLM inference.

All LLM calls go through the Model Broker (Invariant #6).  The port
abstracts away the provider (Vertex AI, Google AI Studio, stub) and
routes by model tier (HIGH, CHEAP) per ADR-38.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ModelBrokerPort(ABC):
    """Port for LLM inference via the Model Broker (L-09)."""

    @abstractmethod
    async def generate(self, *, prompt: str, model_tier: str) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: The fully rendered prompt string.
            model_tier: Tier hint for model routing (e.g. "HIGH", "CHEAP").

        Returns:
            Raw LLM response string.
        """
