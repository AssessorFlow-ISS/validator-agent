"""StubStorageAdapter — returns dummy file bytes for local dev/testing.

The stub tracks which files were downloaded so tests can verify the
download step occurred.
"""
from __future__ import annotations

from validator_agent.ports.storage_port import StoragePort


class StubStorageAdapter(StoragePort):
    """Stub adapter that returns dummy bytes and records download calls."""

    def __init__(
        self,
        *,
        default_bytes: bytes = b"%PDF-1.4 fake-pdf-content",
        overrides: dict[str, bytes] | None = None,
    ) -> None:
        self._default_bytes = default_bytes
        self._overrides = overrides or {}
        self.downloaded: list[str] = []

    async def download_file(self, storage_path: str) -> bytes:
        """Return canned file bytes and record the download."""
        self.downloaded.append(storage_path)
        return self._overrides.get(storage_path, self._default_bytes)
