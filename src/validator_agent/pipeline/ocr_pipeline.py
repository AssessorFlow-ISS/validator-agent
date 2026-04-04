"""Component 2: OCR Pipeline — text extraction + visual understanding.

Takes file bytes + list of readable page numbers (from MRC).
Only processes readable pages. Blurry pages are excluded.
"""

from __future__ import annotations

import os
import time

from validator_agent.pipeline.documentai_client import extract_with_documentai_bytes
from validator_agent.pipeline.models import OcrResult, PageOcrResult
from validator_agent.pipeline.page_classifier import classify_page
from validator_agent.pipeline.pdf_to_images import pdf_pages_to_images
from validator_agent.pipeline.visual_understanding import process_visual_page

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "accessorflow")
LOCATION = os.getenv("GCP_LOCATION", "asia-southeast1")
PROCESSOR_ID = os.getenv("DOCUMENTAI_PROCESSOR_ID", "dabf82e23a09dead")
ENABLE_VISUAL = os.getenv("ENABLE_VISUAL", "true").lower() == "true"


def extract_text(
    file_bytes: bytes,
    file_name: str,
    readable_pages: list[int] | None = None,
) -> OcrResult:
    """Extract text from file. Processes only readable pages if specified.

    Args:
        file_bytes: Raw file content.
        file_name: Original file name.
        readable_pages: Page numbers to process (from MRC). None = all pages.

    Returns:
        OcrResult with per-page text.
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext not in ("pdf", "png", "jpg", "jpeg"):
        return OcrResult(
            file_name=file_name,
            total_pages=0,
            error_message=f"Unsupported file type: .{ext}. Supported: pdf, png, jpg",
        )

    # Pass 1: Document AI OCR
    result = extract_with_documentai_bytes(
        file_bytes=file_bytes,
        file_name=file_name,
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID,
    )

    if not result.success:
        result.overall_status = "TERMINATE"
        return result

    # Filter out blurry pages if MRC provided readable_pages list
    if readable_pages is not None:
        result.pages = [p for p in result.pages if p.page_number in readable_pages]
        result.total_pages = len(result.pages)
        result.total_word_count = sum(p.word_count for p in result.pages)

    if not ENABLE_VISUAL:
        result.overall_status = "PROCEED"
        return result

    # Convert pages to images for GPT-4o
    if ext == "pdf":
        try:
            all_page_images = pdf_pages_to_images(file_bytes)
        except Exception as e:
            result.visual_processing_error = f"Failed to convert PDF to images: {e}"
            result.overall_status = "PROCEED"
            return result
    elif ext in ("png", "jpg", "jpeg"):
        all_page_images = [file_bytes]
    else:
        result.overall_status = "PROCEED"
        return result

    # Build page_number → image mapping
    page_image_map = {}
    for idx, img in enumerate(all_page_images):
        page_num = idx + 1  # 1-indexed
        if readable_pages is None or page_num in readable_pages:
            page_image_map[page_num] = img

    # Pass 2 + 3: Classify and enhance visual pages
    _enhance_visual_pages(result, page_image_map)

    if not result.harmful_image_detected:
        result.overall_status = "PROCEED"

    return result


def _enhance_visual_pages(result: OcrResult, page_image_map: dict[int, bytes]) -> None:
    """Classify pages and enhance visual pages with GPT-4o.

    Includes image moderation — if harmful imagery detected, terminates early.
    """
    visual_pages_processed = 0

    for page in result.pages:
        # Pass 2: Classify
        classification = classify_page(page.extracted_text)
        page.classification = classification

        if classification != "VISUAL":
            page.source = "ocr"
            continue

        # Get image for this page
        page_image = page_image_map.get(page.page_number)
        if not page_image:
            page.source = "ocr"
            continue

        # Pass 3: Visual understanding (includes image moderation)
        visual_result = process_visual_page(
            page_image_bytes=page_image,
            ocr_text=page.extracted_text,
        )

        # Check harmful image detection
        if visual_result.harmful_image_detected:
            result.harmful_image_detected = True
            result.harmful_image_detail = (
                visual_result.image_moderation.detail
                if visual_result.image_moderation
                else "Harmful image detected"
            )
            result.harmful_image_page = page.page_number
            result.overall_status = "TERMINATE"

            # Clear this page and all remaining pages
            for p in result.pages:
                if p.page_number >= page.page_number:
                    p.extracted_text = ""
                    p.word_count = 0

            result.total_word_count = sum(p.word_count for p in result.pages)
            return

        page.source = visual_result.source
        page.visual_evaluation = visual_result.evaluations
        page.visual_attempts = visual_result.attempts

        if visual_result.source == "llm":
            page.extracted_text = visual_result.text
            page.word_count = len(visual_result.text.split())
            visual_pages_processed += 1

    result.total_word_count = sum(p.word_count for p in result.pages)
    result.visual_pages_processed = visual_pages_processed
