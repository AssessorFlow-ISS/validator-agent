"""LocalStorageAdapter — reads files from local filesystem.

For E2E testing, the Mock API saves uploaded files to a local directory
(UPLOAD_DIR). The Validator reads from the same directory using the
storage_path as a relative path within UPLOAD_DIR.

In production, this would be replaced by a GCS adapter that downloads
from Cloud Storage using the storage_path as a GCS URI.
"""
from __future__ import annotations

import os
from pathlib import Path

import structlog

from validator_agent.ports.storage_port import StoragePort

logger = structlog.get_logger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/assessorflow-uploads")


class LocalStorageAdapter(StoragePort):
    """Reads files from a local directory for E2E testing."""

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or UPLOAD_DIR)

    async def download_file(self, storage_path: str) -> bytes:
        """Read file bytes from local filesystem.

        The storage_path is treated as either:
        - An absolute path (if it starts with /)
        - A relative path within UPLOAD_DIR
        - A filename to search for in UPLOAD_DIR
        """
        # Try absolute path first
        abs_path = Path(storage_path)
        if abs_path.is_file():
            logger.info("storage_read", path=str(abs_path))
            return abs_path.read_bytes()

        # Try relative to base_dir
        rel_path = self._base_dir / storage_path
        if rel_path.is_file():
            logger.info("storage_read", path=str(rel_path))
            return rel_path.read_bytes()

        # Try matching by filename only (search recursively)
        filename = Path(storage_path).name
        for p in self._base_dir.rglob(filename):
            if p.is_file():
                logger.info("storage_read_by_name", path=str(p), query=storage_path)
                return p.read_bytes()

        raise FileNotFoundError(
            f"File not found: {storage_path} (searched {self._base_dir})"
        )
