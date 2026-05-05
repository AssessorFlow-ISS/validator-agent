"""SubmissionClient -- gRPC client for the Submission Service.

Copied from the canonical Q&A Generation Agent helper (Phase 6C). The
Validator Agent uses only a subset of the RPCs -- reading materials
pending validation and writing a validation decision per material.

Design notes
------------
* **Single shared channel.** Instantiate one :class:`SubmissionClient`
  at startup and close it on shutdown. The gRPC channel multiplexes
  concurrent RPCs over a single HTTP/2 connection.
* **Plaintext in-cluster.** Production addresses Submission Service via
  ``submission-service.af-submission.svc.cluster.local:9001`` over the
  pod network, which is already protected by NetworkPolicy / mTLS
  sidecar where enabled.
* **Snake-case fields.** Mirror the proto field names verbatim.
* **Retry policy.** Only ``UNAVAILABLE`` is retried -- the canonical
  "pod I was talking to went away" code for rolling deploys. Domain
  errors (``INVALID_ARGUMENT`` / ``NOT_FOUND`` / ``ALREADY_EXISTS``)
  bubble up immediately.
* **Deadlines.** ``default_timeout_seconds`` applies to every RPC
  unless the helper explicitly overrides it.

Environment variables
---------------------
``SUBMISSION_SERVICE_GRPC_URL``
    ``host:port`` of the Submission Service gRPC endpoint. Defaults to
    ``localhost:9001`` for local dev.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

import grpc
import structlog

from validator_agent._grpc import submission_pb2, submission_pb2_grpc

logger = structlog.get_logger(__name__)

_DEFAULT_GRPC_URL = "localhost:9001"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_BACKOFF_SECONDS = 0.25

# Codes we retry on -- UNAVAILABLE covers pod restarts / rolling deploys.
# Domain errors must propagate so the caller can decide on messaging.
_RETRIABLE_CODES = frozenset({grpc.StatusCode.UNAVAILABLE})

T = TypeVar("T")


class SubmissionClient:
    """Async gRPC client for the Submission Service (Validator subset).

    Callers should treat this as a long-lived singleton: construct once
    per process, call :meth:`close` on shutdown. All helpers return the
    raw proto response message so callers can map it to domain types in
    the adapter layer.
    """

    def __init__(
        self,
        *,
        grpc_url: str | None = None,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._grpc_url = grpc_url or os.environ.get("SUBMISSION_SERVICE_GRPC_URL", _DEFAULT_GRPC_URL)
        self._default_timeout_seconds = default_timeout_seconds
        self._channel: grpc.aio.Channel | None = None
        self._stub: submission_pb2_grpc.SubmissionServiceStub | None = None
        logger.info(
            "submission_client_configured",
            grpc_url=self._grpc_url,
            default_timeout_seconds=default_timeout_seconds,
        )

    # -- Lifecycle -----------------------------------------------------------

    def _ensure_stub(self) -> submission_pb2_grpc.SubmissionServiceStub:
        if self._stub is None:
            # Insecure is intentional: in-cluster traffic to
            # submission-service.af-submission.svc.cluster.local:9001
            # rides the pod network and is already protected by the
            # cluster's NetworkPolicy / mTLS sidecar (if enabled).
            self._channel = grpc.aio.insecure_channel(self._grpc_url)
            self._stub = submission_pb2_grpc.SubmissionServiceStub(self._channel)
        return self._stub

    async def close(self) -> None:
        """Close the underlying gRPC channel. Safe to call multiple times."""
        if self._channel is not None:
            try:
                await self._channel.close()
            except Exception:  # pragma: no cover -- best-effort shutdown
                logger.warning("submission_client_close_error", exc_info=True)
            self._channel = None
            self._stub = None

    # -- Retry helper --------------------------------------------------------

    async def _call_with_retry(
        self,
        rpc_name: str,
        call: Callable[[], Awaitable[T]],
    ) -> T:
        """Invoke ``call`` with a bounded retry on transient codes only.

        ``UNAVAILABLE`` is the only retriable code -- everything else is
        treated as a domain error and re-raised on the first attempt.
        """
        last_exc: grpc.aio.AioRpcError | None = None
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                return await call()
            except grpc.aio.AioRpcError as exc:
                code = exc.code()
                if code not in _RETRIABLE_CODES or attempt == _RETRY_MAX_ATTEMPTS:
                    logger.error(
                        "submission_rpc_failed",
                        rpc=rpc_name,
                        grpc_code=code.name if code else "UNKNOWN",
                        grpc_detail=exc.details(),
                        attempt=attempt,
                    )
                    raise
                last_exc = exc
                backoff = _RETRY_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "submission_rpc_retry",
                    rpc=rpc_name,
                    grpc_code=code.name if code else "UNKNOWN",
                    attempt=attempt,
                    backoff_seconds=backoff,
                )
                await asyncio.sleep(backoff)
        # Unreachable -- the final attempt either returns or re-raises.
        assert last_exc is not None
        raise last_exc

    # -- Typed helpers (Validator subset) ------------------------------------

    async def get_materials(
        self,
        *,
        assessment_id: str,
        unvalidated_only: bool = False,
        timeout_seconds: float | None = None,
    ) -> submission_pb2.GetMaterialsResponse:
        """Read materials for an assessment (optionally filtered to unvalidated)."""
        stub = self._ensure_stub()
        request = submission_pb2.GetMaterialsRequest(
            assessment_id=assessment_id,
            unvalidated_only=unvalidated_only,
        )
        return await self._call_with_retry(
            "GetMaterials",
            lambda: asyncio.wait_for(
                stub.GetMaterials(request),
                timeout=timeout_seconds or self._default_timeout_seconds,
            ),
        )

    async def update_material_validation(
        self,
        *,
        assessment_id: str,
        material_id: str,
        readiness_status: str,
        validation_reason_code: str = "",
        validation_message: str = "",
        timeout_seconds: float | None = None,
    ) -> submission_pb2.UpdateMaterialValidationResponse:
        """Write a validation decision (readiness + reason) for a single material.

        Added in proto v2 (Phase 6C.1). Replaces the Validator's previous
        direct-DB / HTTP write to ``assessment_materials.readiness_status``.
        """
        stub = self._ensure_stub()
        request = submission_pb2.UpdateMaterialValidationRequest(
            assessment_id=assessment_id,
            material_id=material_id,
            readiness_status=readiness_status,
            validation_reason_code=validation_reason_code,
            validation_message=validation_message,
        )
        return await self._call_with_retry(
            "UpdateMaterialValidation",
            lambda: asyncio.wait_for(
                stub.UpdateMaterialValidation(request),
                timeout=timeout_seconds or self._default_timeout_seconds,
            ),
        )


__all__ = ["SubmissionClient"]
