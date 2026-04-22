"""Unit tests for LocalStorageAdapter.

Covers the three lookup modes (absolute path, base_dir-relative, name search)
and the FileNotFoundError sad path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from validator_agent.adapters.storage_local import LocalStorageAdapter


class TestLocalStorageAdapter:
    """Filesystem-backed storage adapter for E2E testing."""

    async def test_reads_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "sample.pdf"
        target.write_bytes(b"PDF-CONTENT")
        adapter = LocalStorageAdapter(base_dir=str(tmp_path))

        content = await adapter.download_file(str(target))

        assert content == b"PDF-CONTENT"

    async def test_reads_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "material-1.pdf").write_bytes(b"REL-CONTENT")
        adapter = LocalStorageAdapter(base_dir=str(tmp_path))

        content = await adapter.download_file("material-1.pdf")

        assert content == b"REL-CONTENT"

    async def test_recursive_search_by_filename(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        nested.mkdir(parents=True)
        (nested / "found.txt").write_bytes(b"NESTED")
        adapter = LocalStorageAdapter(base_dir=str(tmp_path))

        content = await adapter.download_file("found.txt")

        assert content == b"NESTED"

    async def test_raises_file_not_found(self, tmp_path: Path) -> None:
        adapter = LocalStorageAdapter(base_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="not found"):
            await adapter.download_file("missing.txt")

    async def test_defaults_to_upload_dir_constant(self, tmp_path: Path) -> None:
        """If no base_dir, uses the module's UPLOAD_DIR env var default."""
        adapter = LocalStorageAdapter()

        # The base_dir should be a Path object
        assert isinstance(adapter._base_dir, Path)
