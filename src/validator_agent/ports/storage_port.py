"""StoragePort — abstract interface for Cloud Storage file access.

The Validator Agent downloads files from Cloud Storage before running
MRC and OCR checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class StoragePort(ABC):
    """Port for downloading files from Cloud Storage."""

    @abstractmethod
    async def download_file(self, storage_path: str) -> bytes:
        """Download a file from Cloud Storage.

        Args:
            storage_path: The storage path to the file.

        Returns:
            Raw file bytes.
        """
