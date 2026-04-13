"""Tests for GcsStorageAdapter — GCS path parsing, download, error handling,
and config-driven adapter selection.

All GCS SDK interactions are mocked; no network calls required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import Forbidden, NotFound

from validator_agent.adapters.storage_gcs import GcsStorageAdapter, _parse_gcs_uri


# ---------------------------------------------------------------------------
# GCS URI parsing
# ---------------------------------------------------------------------------

class TestParseGcsUri:
    def test_standard_uri(self) -> None:
        bucket, path = _parse_gcs_uri("gs://my-bucket/path/to/file.pdf")
        assert bucket == "my-bucket"
        assert path == "path/to/file.pdf"

    def test_single_level_path(self) -> None:
        bucket, path = _parse_gcs_uri("gs://bucket/file.pdf")
        assert bucket == "bucket"
        assert path == "file.pdf"

    def test_deeply_nested_path(self) -> None:
        bucket, path = _parse_gcs_uri("gs://b/a/b/c/d/e/f.txt")
        assert bucket == "b"
        assert path == "a/b/c/d/e/f.txt"

    def test_not_a_gcs_uri(self) -> None:
        with pytest.raises(ValueError, match="Not a GCS URI"):
            _parse_gcs_uri("s3://wrong-scheme/file.pdf")

    def test_no_path_component(self) -> None:
        with pytest.raises(ValueError, match="no object path"):
            _parse_gcs_uri("gs://bucket-only")

    def test_empty_path_after_slash(self) -> None:
        with pytest.raises(ValueError, match="no object path"):
            _parse_gcs_uri("gs://bucket/")

    def test_http_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="Not a GCS URI"):
            _parse_gcs_uri("https://storage.googleapis.com/bucket/file.pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(content: bytes = b"file-content") -> MagicMock:
    """Build a mock google.cloud.storage.Client with a downloadable blob."""
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = content

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    return mock_client


# ---------------------------------------------------------------------------
# Download — full GCS URI
# ---------------------------------------------------------------------------

class TestGcsDownloadFullUri:
    async def test_download_with_gs_uri(self) -> None:
        mock_client = _make_mock_client(b"pdf-bytes")
        adapter = GcsStorageAdapter(client=mock_client)

        data = await adapter.download_file("gs://my-bucket/path/to/file.pdf")

        assert data == b"pdf-bytes"
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_client.bucket.return_value.blob.assert_called_once_with("path/to/file.pdf")

    async def test_download_returns_raw_bytes(self) -> None:
        binary = bytes(range(256))
        mock_client = _make_mock_client(binary)
        adapter = GcsStorageAdapter(client=mock_client)

        data = await adapter.download_file("gs://bucket/binary.bin")
        assert data == binary


# ---------------------------------------------------------------------------
# Download — relative path with default bucket
# ---------------------------------------------------------------------------

class TestGcsDownloadRelativePath:
    async def test_relative_path_uses_default_bucket(self) -> None:
        mock_client = _make_mock_client(b"content")
        adapter = GcsStorageAdapter(bucket_name="default-bucket", client=mock_client)

        data = await adapter.download_file("materials/abc123/file.pdf")

        assert data == b"content"
        mock_client.bucket.assert_called_once_with("default-bucket")
        mock_client.bucket.return_value.blob.assert_called_once_with(
            "materials/abc123/file.pdf"
        )

    async def test_relative_path_without_bucket_raises(self) -> None:
        mock_client = _make_mock_client()
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(ValueError, match="GCS_BUCKET_NAME"):
            await adapter.download_file("materials/file.pdf")

    async def test_bucket_from_env(self) -> None:
        mock_client = _make_mock_client(b"env-content")
        with patch.dict("os.environ", {"GCS_BUCKET_NAME": "env-bucket"}):
            adapter = GcsStorageAdapter(client=mock_client)

        data = await adapter.download_file("path/file.pdf")

        assert data == b"env-content"
        mock_client.bucket.assert_called_once_with("env-bucket")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGcsErrorHandling:
    async def test_not_found_raises_file_not_found(self) -> None:
        mock_client = _make_mock_client()
        mock_client.bucket.return_value.blob.return_value.download_as_bytes.side_effect = (
            NotFound("Object not found")
        )
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(FileNotFoundError, match="GCS object not found"):
            await adapter.download_file("gs://bucket/missing.pdf")

    async def test_forbidden_raises_permission_error(self) -> None:
        mock_client = _make_mock_client()
        mock_client.bucket.return_value.blob.return_value.download_as_bytes.side_effect = (
            Forbidden("403 Access Denied")
        )
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(PermissionError, match="Permission denied"):
            await adapter.download_file("gs://bucket/secret.pdf")

    async def test_not_found_on_relative_path(self) -> None:
        mock_client = _make_mock_client()
        mock_client.bucket.return_value.blob.return_value.download_as_bytes.side_effect = (
            NotFound("404")
        )
        adapter = GcsStorageAdapter(bucket_name="b", client=mock_client)

        with pytest.raises(FileNotFoundError, match="gs://b/rel.pdf"):
            await adapter.download_file("rel.pdf")


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

class TestGcsClientInit:
    async def test_client_created_lazily(self) -> None:
        """Client is not created until first download_file call."""
        with patch("validator_agent.adapters.storage_gcs.gcs.Client") as mock_cls:
            mock_instance = _make_mock_client(b"lazy")
            mock_cls.return_value = mock_instance

            adapter = GcsStorageAdapter(bucket_name="b")
            # Client should NOT be created yet
            mock_cls.assert_not_called()

            data = await adapter.download_file("file.pdf")
            # Now it should be created
            mock_cls.assert_called_once()
            assert data == b"lazy"

    async def test_project_id_passed_to_client(self) -> None:
        with patch("validator_agent.adapters.storage_gcs.gcs.Client") as mock_cls:
            mock_cls.return_value = _make_mock_client(b"ok")
            adapter = GcsStorageAdapter(
                bucket_name="b", project_id="my-project"
            )

            await adapter.download_file("file.pdf")
            mock_cls.assert_called_once_with(project="my-project")


# ---------------------------------------------------------------------------
# Config-driven adapter selection (integration with main.py wiring)
# ---------------------------------------------------------------------------

class TestConfigDrivenSelection:
    def test_stub_is_default(self) -> None:
        from validator_agent.config import ValidatorConfig
        config = ValidatorConfig.from_env()
        assert config.storage_adapter == "stub"

    def test_gcs_config_value(self) -> None:
        from validator_agent.config import ValidatorConfig
        with patch.dict("os.environ", {"STORAGE_ADAPTER": "gcs"}):
            config = ValidatorConfig.from_env()
        assert config.storage_adapter == "gcs"

    def test_local_config_value(self) -> None:
        from validator_agent.config import ValidatorConfig
        with patch.dict("os.environ", {"STORAGE_ADAPTER": "local"}):
            config = ValidatorConfig.from_env()
        assert config.storage_adapter == "local"

    def test_gcs_adapter_is_storage_port(self) -> None:
        """GcsStorageAdapter satisfies the StoragePort interface."""
        from validator_agent.ports.storage_port import StoragePort
        mock_client = _make_mock_client()
        adapter = GcsStorageAdapter(bucket_name="b", client=mock_client)
        assert isinstance(adapter, StoragePort)
