"""Configuration for adapter selection and runtime settings.

Environment variables control which adapter implementation is used for
each port.  This enables config-driven swap between stub and real
adapters (ADR-42) with zero code changes.

Environment Variables:
    MRC_ADAPTER:       "stub" (default) | "vertex_ai"
    OCR_ADAPTER:       "stub" (default) | "vertex_ai"
    LLM_PROVIDER:      "stub" (default) | "google_ai_studio" | "vertex_ai"
    KNOWLEDGE_ADAPTER: "stub" (default) | "grpc"
    AUDIT_ADAPTER:     "stub" (default) | "grpc"
    EVENT_ADAPTER:     "stub" (default) | "pubsub"
    STORAGE_ADAPTER:   "stub" (default) | "gcs"
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatorConfig:
    """Immutable runtime configuration for the Validator Agent."""

    mrc_adapter: str
    ocr_adapter: str
    llm_provider: str
    knowledge_adapter: str
    audit_adapter: str
    event_adapter: str
    storage_adapter: str

    @classmethod
    def from_env(cls) -> ValidatorConfig:
        """Load configuration from environment variables."""
        return cls(
            mrc_adapter=os.getenv("MRC_ADAPTER", "stub"),
            ocr_adapter=os.getenv("OCR_ADAPTER", "stub"),
            llm_provider=os.getenv("LLM_PROVIDER", "stub"),
            knowledge_adapter=os.getenv("KNOWLEDGE_ADAPTER", "stub"),
            audit_adapter=os.getenv("AUDIT_ADAPTER", "stub"),
            event_adapter=os.getenv("EVENT_ADAPTER", "stub"),
            storage_adapter=os.getenv("STORAGE_ADAPTER", "stub"),
        )
