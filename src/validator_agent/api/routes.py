"""FastAPI routes for the Validator Agent.

Endpoints:
  POST /invoke  — Main validation endpoint
  GET  /health  — Liveness probe
  GET  /ready   — Readiness probe
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from validator_agent.api.schemas import ValidationRequest, ValidationResponse

router = APIRouter()


@router.post("/invoke", response_model=ValidationResponse)
async def invoke(
    body: ValidationRequest,
    request: Request,
) -> ValidationResponse:
    """Run the full validation pipeline for the submitted files.

    The ValidatorService is injected via app.state at startup.
    """
    service = request.app.state.validator_service
    return await service.validate(body)


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 if the process is running."""
    return {"status": "healthy"}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe — returns 200 if the agent is ready to accept traffic."""
    return {"status": "ready"}
