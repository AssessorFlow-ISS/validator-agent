"""Tests for GCS storage adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from validator_agent.adapters.storage_gcs import GcsStorageAdapter


class TestGcsStorageAdapter:
    """Test suite for GcsStorageAdapter."""

    async def test_download_with_gs_uri(self) -> None:
        """Test downloading a file using full gs:// URI."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"file content from gcs"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)
        data = await adapter.download_file("gs://my-bucket/path/to/file.pdf")

        assert data == b"file content from gcs"
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("path/to/file.pdf")

    async def test_download_with_relative_path_and_env_bucket(self) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"relative path content"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch.dict("os.environ", {"GCS_BUCKET_NAME": "env-bucket"}):
            adapter = GcsStorageAdapter(client=mock_client)
            data = await adapter.download_file("uploads/document.pdf")

        assert data == b"relative path content"
        mock_client.bucket.assert_called_once_with("env-bucket")
        mock_bucket.blob.assert_called_once_with("uploads/document.pdf")

    async def test_download_with_relative_path_and_ctor_bucket(self) -> None:
        """Test downloading with relative path using constructor bucket_name."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"ctor bucket content"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(bucket_name="ctor-bucket", client=mock_client)
        data = await adapter.download_file("materials/file.txt")

        assert data == b"ctor bucket content"
        mock_client.bucket.assert_called_once_with("ctor-bucket")

    async def test_download_blob_not_found_raises_file_not_found(self) -> None:
        class NotFoundError(Exception):
            pass

        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = NotFoundError("404 Not Found")

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(FileNotFoundError) as exc_info:
            await adapter.download_file("gs://my-bucket/missing.pdf")

        assert "Blob not found" in str(exc_info.value)
        assert "gs://my-bucket/missing.pdf" in str(exc_info.value)

    async def test_relative_path_without_bucket_raises_value_error(self) -> None:
        mock_client = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import validator_agent.adapters.storage_gcs as gcs_module

            importlib.reload(gcs_module)
            from validator_agent.adapters.storage_gcs import GcsStorageAdapter as ReloadedAdapter

            adapter = ReloadedAdapter(client=mock_client)

            with pytest.raises(ValueError) as exc_info:
                await adapter.download_file("relative/path.pdf")

        assert "Cannot resolve relative storage path" in str(exc_info.value)
        assert "GCS_BUCKET_NAME" in str(exc_info.value)

    async def test_uri_overrides_env_bucket(self) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"uri wins"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch.dict("os.environ", {"GCS_BUCKET_NAME": "env-bucket"}):
            adapter = GcsStorageAdapter(client=mock_client)
            data = await adapter.download_file("gs://uri-bucket/file.pdf")

        assert data == b"uri wins"
        mock_client.bucket.assert_called_once_with("uri-bucket")

    async def test_ctor_bucket_overrides_env_bucket(self) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"ctor wins"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch.dict("os.environ", {"GCS_BUCKET_NAME": "env-bucket"}):
            adapter = GcsStorageAdapter(bucket_name="ctor-bucket", client=mock_client)
            data = await adapter.download_file("path.pdf")

        assert data == b"ctor wins"
        mock_client.bucket.assert_called_once_with("ctor-bucket")

    async def test_gs_uri_with_nested_path(self) -> None:
        """Test gs:// URI with deeply nested path."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"nested"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)
        data = await adapter.download_file("gs://bucket/a/b/c/d/e/deeply/nested/file.pdf")

        assert data == b"nested"
        mock_bucket.blob.assert_called_once_with("a/b/c/d/e/deeply/nested/file.pdf")

    async def test_gs_uri_with_special_characters(self) -> None:
        """Test gs:// URI with special characters in path."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"special"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)
        data = await adapter.download_file("gs://my-bucket/path/with spaces/and-dashes_file.pdf")

        assert data == b"special"

    async def test_adapter_creates_client_on_demand(self) -> None:
        """Test that adapter creates storage.Client when none provided."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"lazy client"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter()
        assert adapter._client is None

        with patch("google.cloud.storage.Client", return_value=mock_client):
            data = await adapter.download_file("gs://bucket/file.pdf")

        assert data == b"lazy client"
        assert adapter._client is mock_client

    async def test_reuses_existing_client(self) -> None:
        """Test that adapter reuses client after first creation."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"reuse"

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)

        await adapter.download_file("gs://bucket/file1.pdf")
        await adapter.download_file("gs://bucket/file2.pdf")

        assert adapter._client is mock_client
        assert mock_client.bucket.call_count == 2

    async def test_malformed_gs_uri_trailing_slash(self) -> None:
        mock_client = MagicMock()
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(ValueError) as exc_info:
            await adapter.download_file("gs://bucket/")

        assert "Invalid GCS URI" in str(exc_info.value)

    async def test_malformed_gs_uri_no_blob(self) -> None:
        mock_client = MagicMock()
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(ValueError) as exc_info:
            await adapter.download_file("gs://bucket")

        assert "Invalid GCS URI" in str(exc_info.value)

    async def test_empty_storage_path(self) -> None:
        mock_client = MagicMock()
        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(ValueError) as exc_info:
            await adapter.download_file("")

        assert "Storage path cannot be empty" in str(exc_info.value)

    async def test_blob_deleted_between_check_and_download(self) -> None:
        class NotFoundError(Exception):
            pass

        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = NotFoundError("404 Not Found")

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        adapter = GcsStorageAdapter(client=mock_client)

        with pytest.raises(FileNotFoundError) as exc_info:
            await adapter.download_file("gs://bucket/file.pdf")

        assert "Blob not found" in str(exc_info.value)
