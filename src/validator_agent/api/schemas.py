"""API request/response schemas for the Validator Agent.

These models define the contract for POST /invoke and are used by both
the routes and the domain service layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from validator_agent.domain.terminal_signal import TerminalSignal


class FileInfo(BaseModel):
    """Information about a single file to validate."""

    file_name: str = Field(..., description="Original filename")
    storage_path: str = Field(..., description="Cloud Storage path")
    file_type: str = Field(..., description="File type (pdf, png, jpg, etc.)")


class ValidationRequest(BaseModel):
    """Request body for POST /invoke."""

    workflow_id: str = Field(..., description="Workflow identifier")
    assessment_id: str = Field(..., description="Assessment identifier")
    assessor_id: str | None = Field(default=None, description="Assessor identity for trace filtering")
    validation_type: str = Field(
        default="material_validation",
        description="Type of validation (material_validation, web_research_validation)",
    )
    files: list[FileInfo] = Field(
        ...,
        min_length=1,
        description="List of files to validate",
    )


class FileResult(BaseModel):
    """Validation result for a single file including pipeline output."""

    file_name: str
    terminal_signal: TerminalSignal
    cleaned_text: str = ""
    assessor_warnings: list[dict] = Field(default_factory=list)
    total_time_ms: float = 0.0
    terminated_at_component: str | None = None


class ValidationResponse(BaseModel):
    """Response body for POST /invoke."""

    workflow_id: str
    terminal_signal: TerminalSignal
    file_results: list[FileResult]
    # Aggregated from all files
    cleaned_text: str = ""
    assessor_warnings: list[dict] = Field(default_factory=list)
