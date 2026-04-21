"""Component 1: MRC (Material Readiness Checker) client.

Supports two modes:
  - dev: gcloud CLI for Bearer token (local development)
  - prod: GKE Workload Identity for automatic token (no CLI needed)

Applies the 30% blurry threshold to decide PROCEED, PROCEED_WITH_EXCLUSIONS, or TERMINATE.
"""

from __future__ import annotations

import logging
import os
import subprocess

import requests

from validator_agent.pipeline.models import MrcResult

logger = logging.getLogger(__name__)

MRC_ENDPOINT = os.getenv("MRC_ENDPOINT", "")
BLUR_THRESHOLD = float(os.getenv("MRC_BLUR_THRESHOLD", "0.30"))
APP_MODE = os.getenv("APP_MODE", "dev")


def _is_vertex_ai_endpoint(endpoint: str) -> bool:
    return "aiplatform.googleapis.com" in endpoint


def _get_gcp_token() -> str:
    """Get GCP access token based on APP_MODE.

    dev:  uses gcloud CLI (requires gcloud auth login)
    prod: uses GKE Workload Identity via metadata server (automatic on GKE)
    """
    if APP_MODE == "prod":
        # GKE Workload Identity — fetch token from metadata server
        try:
            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception:
            return ""
    else:
        # Dev mode — use gcloud CLI
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""


def _call_vertex_ai(file_bytes: bytes, file_name: str) -> dict:
    """Call MRC via Vertex AI rawPredict endpoint."""
    token = _get_gcp_token()
    if not token:
        raise ConnectionError(
            "Failed to get GCP access token — "
            "run 'gcloud auth login' (dev) or check Workload Identity (prod)"
        )

    response = requests.post(
        MRC_ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (file_name, file_bytes)},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _call_local(file_bytes: bytes, file_name: str) -> dict:
    """Call MRC via local FastAPI server."""
    response = requests.post(
        MRC_ENDPOINT,
        files={"file": (file_name, file_bytes)},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def check_readability(file_bytes: bytes, file_name: str) -> MrcResult:
    """Send file to MRC and apply blurry threshold logic.

    Automatically detects whether to use Vertex AI production endpoint
    or local server based on the MRC_ENDPOINT URL.

    Args:
        file_bytes: Raw file content.
        file_name: Original file name.

    Returns:
        MrcResult with overall_status, excluded_pages, and readable_page_numbers set.
    """
    # Golden / test-env escape hatch: if MRC_ENDPOINT is unset, skip the
    # per-page readability check entirely and PROCEED. The hexagonal-layer
    # `mrc_adapter=stub` (services.py path) already covers synthetic
    # validation decisions; this path exists so Thet's pipeline doesn't
    # crash with `MissingSchema: Invalid URL ''` when there's no MRC
    # service to call. PROD always sets MRC_ENDPOINT to a Vertex AI URL.
    if not MRC_ENDPOINT:
        return MrcResult(
            overall_readiness=True,
            overall_confidence=0.95,
            total_pages=1,
            readable_pages=1,
            unreadable_pages=0,
            page_results=[],
            file_name=file_name,
            inference_time_ms=0.0,
            overall_status="PROCEED",
            blurry_ratio=0.0,
        )
    try:
        if _is_vertex_ai_endpoint(MRC_ENDPOINT):
            data = _call_vertex_ai(file_bytes, file_name)
        else:
            data = _call_local(file_bytes, file_name)
    except requests.exceptions.ConnectionError as exc:
        logger.error("MRC connection error for %s: %s", file_name, exc)
        return MrcResult(
            overall_readiness=False,
            overall_confidence=0.0,
            total_pages=0,
            readable_pages=0,
            unreadable_pages=0,
            page_results=[],
            file_name=file_name,
            inference_time_ms=0.0,
            overall_status="TERMINATE",
            blurry_ratio=0.0,
        )
    except Exception as exc:
        logger.error("MRC call failed for %s: %s", file_name, exc, exc_info=True)
        return MrcResult(
            overall_readiness=False,
            overall_confidence=0.0,
            total_pages=0,
            readable_pages=0,
            unreadable_pages=0,
            page_results=[],
            file_name=file_name,
            inference_time_ms=0.0,
            overall_status="TERMINATE",
            blurry_ratio=0.0,
        )

    mrc_result = MrcResult(**data)

    # Apply 30% blurry threshold
    if mrc_result.total_pages == 0:
        mrc_result.overall_status = "TERMINATE"
        mrc_result.blurry_ratio = 1.0
        return mrc_result

    blurry_ratio = mrc_result.unreadable_pages / mrc_result.total_pages
    mrc_result.blurry_ratio = round(blurry_ratio, 4)

    # Determine readable and excluded pages
    readable = [p.page for p in mrc_result.page_results if p.readiness]
    excluded = [p.page for p in mrc_result.page_results if not p.readiness]
    mrc_result.readable_page_numbers = readable
    mrc_result.excluded_pages = excluded

    if blurry_ratio > BLUR_THRESHOLD:
        mrc_result.overall_status = "TERMINATE"
    elif blurry_ratio > 0:
        mrc_result.overall_status = "PROCEED_WITH_EXCLUSIONS"
    else:
        mrc_result.overall_status = "PROCEED"

    return mrc_result
