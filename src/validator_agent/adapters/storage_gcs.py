"""GcsStorageAdapter — downloads files from Google Cloud Storage.

Production adapter that reads files from GCS buckets using Application Default
Credentials (ADC). Supports both full GCS URIs (gs://bucket/path) and
bucket-relative paths when GCS_BUCKET_NAME is configured.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING

import structlog

from validator_agent.ports.storage_port import StoragePort

if TYPE_CHECKING:
    from google.cloud.storage import Blob, Bucket, Client

logger = structlog.get_logger(__name__)

_GCS_URI_PATTERN = re.compile(r"^gs://([^/]+)/(.+)$")


class GcsStorageAdapter(StoragePort):
    """Downloads files from Google Cloud Storage.

    Supports two storage_path formats:
    1. Full GCS URI: gs://bucket-name/path/to/file.pdf
    2. Bucket-relative path: path/to/file.pdf (requires GCS_BUCKET_NAME env var)

    Uses Application Default Credentials (ADC) for authentication.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        client: Client | None = None,
    ) -> None:
        """Initialize the GCS storage adapter."""
        self._default_bucket = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self._client = client

    def _get_client(self) -> Client:
        """Get or create the GCS client."""
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def _parse_storage_path(self, storage_path: str) -> tuple[str, str]:
        """Parse storage path into bucket name and blob path.

        Args:
            storage_path: Either a gs:// URI or bucket-relative path.

        Returns:
            Tuple of (bucket_name, blob_path).

        Raises:
            ValueError: If storage_path is a relative path but no bucket is configured.
        """
        # Check if it's a gs:// URI
        if not storage_path:
            raise ValueError("Storage path cannot be empty.")

        if storage_path.startswith("gs://"):
            match = _GCS_URI_PATTERN.match(storage_path)
            if not match:
                raise ValueError(f"Invalid GCS URI: '{storage_path}'. Expected format: gs://bucket-name/path/to/blob")
            return match.group(1), match.group(2)

        if self._default_bucket is None:
            raise ValueError(
                f"Cannot resolve relative storage path '{storage_path}'. "
                "Either provide a gs:// URI or set GCS_BUCKET_NAME environment variable."
            )

        return self._default_bucket, storage_path

    async def download_file(self, storage_path: str) -> bytes:
        """Download a file from Google Cloud Storage.

        Args:
            storage_path: GCS URI (gs://bucket/path) or bucket-relative path.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the blob does not exist in the bucket.
            ValueError: If storage_path is invalid or bucket cannot be determined.
        """
        bucket_name, blob_path = self._parse_storage_path(storage_path)

        logger.info(
            "gcs_download_start",
            bucket=bucket_name,
            blob=blob_path,
            storage_path=storage_path,
        )

        def _download() -> bytes:
            client = self._get_client()
            bucket: Bucket = client.bucket(bucket_name)
            blob: Blob = bucket.blob(blob_path)

            try:
                return blob.download_as_bytes()
            except Exception as e:
                if "NotFound" in type(e).__name__ or "404" in str(e):
                    logger.warning(
                        "gcs_blob_not_found",
                        bucket=bucket_name,
                        blob=blob_path,
                    )
                    raise FileNotFoundError(f"Blob not found: gs://{bucket_name}/{blob_path}") from e
                raise

        try:
            # Run synchronous GCS operation in thread pool
            data = await asyncio.to_thread(_download)

            logger.info(
                "gcs_download_complete",
                bucket=bucket_name,
                blob=blob_path,
                size_bytes=len(data),
            )

            return data

        except FileNotFoundError:
            # Re-raise FileNotFoundError for consistency with StoragePort contract
            raise

        except Exception as e:
            # Log and re-raise other exceptions (auth errors, network issues, etc.)
            logger.error(
                "gcs_download_failed",
                bucket=bucket_name,
                blob=blob_path,
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise
