"""OcrPort — abstract interface for OCR text extraction.

In production this calls Vertex AI Vision.  Stub adapter returns
pre-extracted text from test fixtures for local development.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrResult:
    """Result from OCR text extraction."""

    text: str
    source_type: str  # e.g. "ocr_extracted", "native_text"


class OcrPort(ABC):
    """Port for OCR text extraction from documents."""

    @abstractmethod
    async def extract_text(self, storage_path: str) -> OcrResult:
        """Extract text content from a document.

        Args:
            storage_path: Cloud Storage path to the document.

        Returns:
            OcrResult with extracted text and source type.
        """
