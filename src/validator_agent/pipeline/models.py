from __future__ import annotations

from pydantic import BaseModel, Field


# ── Component 1: MRC Models ────

class MrcPageResult(BaseModel):
    page: int
    readiness: bool
    confidence: float


class MrcResult(BaseModel):
    overall_readiness: bool
    overall_confidence: float
    total_pages: int
    readable_pages: int
    unreadable_pages: int
    page_results: list[MrcPageResult]
    file_name: str
    inference_time_ms: float

    # Validator Agent decision fields (set after processing MRC response)
    overall_status: str = "PENDING"  # "PROCEED", "PROCEED_WITH_EXCLUSIONS", "TERMINATE"
    blurry_ratio: float = 0.0
    excluded_pages: list[int] = Field(default_factory=list)  # page numbers excluded
    readable_page_numbers: list[int] = Field(default_factory=list)  # page numbers that passed


# ── Component 2: OCR Models ────

class PageOcrResult(BaseModel):
    page_number: int = Field(description="1-indexed, matches MRC page numbering")
    extracted_text: str = ""
    word_count: int = 0
    confidence: float | None = Field(default=None, description="0.0-1.0")
    classification: str | None = Field(default=None, description="TEXT or VISUAL")
    source: str | None = Field(default=None, description="ocr, llm, or ocr_fallback")
    visual_attempts: int = 0
    visual_evaluation: list | None = None


class OcrResult(BaseModel):
    file_name: str
    total_pages: int = 0
    pages: list[PageOcrResult] = Field(default_factory=list)
    total_word_count: int = 0
    ocr_time_ms: float = 0.0
    overall_status: str = Field(default="PENDING", description="PROCEED or TERMINATE")
    success: bool = False
    error_message: str | None = None
    processing_mode: str | None = None
    visual_pages_processed: int = 0
    visual_processing_error: str | None = None
    harmful_image_detected: bool = False
    harmful_image_detail: str | None = None
    harmful_image_page: int | None = None


# ── Component 3: Content Safety Models ────

class Finding(BaseModel):
    page: int
    type: str
    detail: str
    original: str | None = None
    redacted_to: str | None = None
    source: str | None = None
    confidence: str | None = None


class ContentSafetyResult(BaseModel):
    overall_status: str = Field(description="PROCEED, TERMINATE, or PROCEED_WITH_WARNINGS")
    termination_reason: str | None = None
    termination_detail: str | None = None
    error_message: str | None = None

    harmful_detected: bool = False
    harmful_findings: list[Finding] = Field(default_factory=list)
    pii_detected: bool = False
    pii_findings: list[Finding] = Field(default_factory=list)
    religious_political_detected: bool = False
    religious_political_findings: list[Finding] = Field(default_factory=list)
    copyright_detected: bool = False
    copyright_findings: list[Finding] = Field(default_factory=list)
    misinformation_detected: bool = False
    misinformation_findings: list[Finding] = Field(default_factory=list)

    cleaned_text: str = ""
    assessor_action_required: bool = False
    assessor_warnings: list[dict] = Field(default_factory=list)


# ── Validator Agent Overall Result ────

class ValidatorResult(BaseModel):
    """End-to-end Validator Agent result — all 3 components combined."""

    file_name: str
    overall_status: str = Field(description="PROCEED, PROCEED_WITH_WARNINGS, or TERMINATE")
    termination_reason: str | None = None
    termination_detail: str | None = None
    terminated_at_component: str | None = Field(default=None, description="Which component terminated: mrc, ocr, or content_safety")

    # Component results
    mrc: MrcResult | None = None
    ocr: OcrResult | None = None
    content_safety: ContentSafetyResult | None = None

    # Final output (only when PROCEED or PROCEED_WITH_WARNINGS)
    cleaned_text: str = ""
    assessor_warnings: list[dict] = Field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0
