"""MrcPort — abstract interface for the Material Readiness Checker.

MRC (#11) is Thet's Vertex AI endpoint that classifies documents as
readable/unreadable with a confidence score.  The Validator Agent wraps it
as a tool via this port (Invariant #21, ADR-42).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MrcResult:
    """Result from the Material Readiness Checker."""

    readiness: bool
    confidence: float


class MrcPort(ABC):
    """Port for Material Readiness Checker predictions."""

    @abstractmethod
    async def predict(self, storage_path: str) -> MrcResult:
        """Predict whether a document is readable.

        Args:
            storage_path: Cloud Storage path to the document.

        Returns:
            MrcResult with readiness flag and confidence score.
        """
