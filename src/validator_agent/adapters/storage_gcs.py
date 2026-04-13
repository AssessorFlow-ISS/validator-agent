"""GcsStorageAdapter — downloads files from Google Cloud Storage.

For GCP deployment, the Validator Agent reads assessment materials from
Cloud Storage using the google-cloud-storage SDK.  Uses Application
Default Credentials (ADC), which works with both ``gcloud auth`` locally
and GKE Workload Identity in production.

Environment Variables:
    GCS_BUCKET_NAME:  Default bucket when storage_path is a relative key
                      (not a full ``gs://`` URI).  No default — must be set
                      if relative paths are used.
    GCS_PROJECT_ID:   (Optional) Explicit project ID.  When omitted the
                      SDK infers it from ADC / metadata server.
"""
from __future__ import annotations

import os

import structlog
from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage as gcs

from validator_agent.ports.storage_port import StoragePort

logger = structlog.get_logger(__name__)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a ``gs://bucket/path/to/file`` URI into (bucket, blob_path).

    Raises:
        ValueError: If the URI does not start with ``gs://`` or has no path.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    without_scheme = uri[len("gs://"):]
    parts = without_scheme.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"GCS URI has no object path: {uri}")
    return parts[0], parts[1]


class GcsStorageAdapter(StoragePort):
    """Downloads files from Google Cloud Storage."""

    def __init__(
        self,
        *,
        bucket_name: str | None = None,
        project_id: str | None = None,
        client: gcs.Client | None = None,
    ) -> None:
        self._default_bucket = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self._project_id = project_id or os.getenv("GCS_PROJECT_ID") or None
        # Accept an injected client for testing; create lazily otherwise.
        self._client = client

    def _get_client(self) -> gcs.Client:
        """Return the GCS client, creating it on first use."""
        if self._client is None:
            self._client = gcs.Client(project=self._project_id)
        return self._client

    async def download_file(self, storage_path: str) -> bytes:
        """Download a file from GCS and return raw bytes.

        ``storage_path`` may be either:
        * A full GCS URI: ``gs://bucket-name/path/to/file.pdf``
        * A relative object key: ``materials/abc123/file.pdf``
          (uses ``GCS_BUCKET_NAME`` as the bucket)

        Raises:
            FileNotFoundError: Blob or bucket does not exist.
            PermissionError:   Caller lacks ``storage.objects.get``.
            ValueError:        Relative path used without ``GCS_BUCKET_NAME``.
        """
        if storage_path.startswith("gs://"):
            bucket_name, blob_path = _parse_gcs_uri(storage_path)
        else:
            if not self._default_bucket:
                raise ValueError(
                    f"Relative storage path '{storage_path}' requires "
                    "GCS_BUCKET_NAME to be set."
                )
            bucket_name = self._default_bucket
            blob_path = storage_path

        logger.info(
            "gcs_download_start",
            bucket=bucket_name,
            blob=blob_path,
        )

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            data = blob.download_as_bytes()
            logger.info(
                "gcs_download_complete",
                bucket=bucket_name,
                blob=blob_path,
                size_bytes=len(data),
            )
            return data
        except NotFound:
            raise FileNotFoundError(
                f"GCS object not found: gs://{bucket_name}/{blob_path}"
            )
        except Forbidden as exc:
            raise PermissionError(
                f"Permission denied for gs://{bucket_name}/{blob_path}: {exc}"
            )
