"""Component 2: OCR Pipeline — text extraction + visual understanding.

Takes file bytes + list of readable page numbers (from MRC).
Only processes readable pages. Blurry pages are excluded.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from validator_agent.pipeline.documentai_client import extract_with_documentai_bytes
from validator_agent.pipeline.models import OcrResult
from validator_agent.pipeline.page_classifier import classify_page
from validator_agent.pipeline.pdf_to_images import pdf_pages_to_images
from validator_agent.pipeline.visual_understanding import process_visual_batch, process_visual_page

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "accessorflow")
VISUAL_BATCH_SIZE = int(os.getenv("VISUAL_BATCH_SIZE", "3"))
VISUAL_BATCH_STAGGER_SECONDS = float(os.getenv("VISUAL_BATCH_STAGGER_SECONDS", "2.0"))
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

    Uses hybrid batch + parallel processing (AF-184):
    - Phase 1: Classify all pages (sequential, cheap GPT-4o-mini)
    - Phase 2: Chunk VISUAL pages into batches of VISUAL_BATCH_SIZE
    - Phase 3: Run all batches in parallel via ThreadPoolExecutor
    - Phase 4: Map results back to original page objects

    Includes image moderation — if harmful imagery detected, terminates early.
    """
    # Phase 1: Classify ALL pages (sequential — cheap GPT-4o-mini calls)
    visual_pages = []
    for page in result.pages:
        classification = classify_page(page.extracted_text)
        page.classification = classification

        if classification == "VISUAL" and page.page_number in page_image_map:
            visual_pages.append(page)
        else:
            page.source = "ocr"

    if not visual_pages:
        return

    # Phase 2: Chunk visual pages into batches
    batches: list[list[tuple[int, bytes, str]]] = []
    for i in range(0, len(visual_pages), VISUAL_BATCH_SIZE):
        batch_pages = visual_pages[i : i + VISUAL_BATCH_SIZE]
        batch_items = [
            (p.page_number, page_image_map[p.page_number], p.extracted_text)
            for p in batch_pages
        ]
        batches.append(batch_items)

    # Phase 3: Run batches in parallel with staggered submission
    # Stagger by VISUAL_BATCH_STAGGER_SECONDS to avoid hitting provider TPM rate limits.
    # With 5 concurrent batches of 3 high-res images each (~6K tokens/batch),
    # simultaneous submission can exhaust a 30K TPM limit instantly.
    all_results: dict[int, object] = {}

    with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
        future_to_batch: dict = {}
        for i, batch in enumerate(batches):
            if i > 0:
                time.sleep(VISUAL_BATCH_STAGGER_SECONDS)
            future_to_batch[executor.submit(process_visual_batch, batch)] = batch

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.update(batch_results)
            except Exception as e:
                # Batch failed — fall back to single-page processing
                logger.warning("Batch failed: %s — falling back to single-page", e)
                for pn, img_bytes, ocr_text in batch:
                    all_results[pn] = process_visual_page(
                        page_image_bytes=img_bytes,
                        ocr_text=ocr_text,
                    )

    # Phase 4: Map results back to pages
    visual_pages_processed = 0
    for page in visual_pages:
        vr = all_results.get(page.page_number)
        if vr is None:
            page.source = "ocr"
            continue

        # Check harmful image detection
        if vr.harmful_image_detected:
            result.harmful_image_detected = True
            result.harmful_image_detail = (
                vr.image_moderation.detail
                if vr.image_moderation
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

        page.source = vr.source
        page.visual_evaluation = vr.evaluations
        page.visual_attempts = vr.attempts

        if vr.source == "llm":
            page.extracted_text = vr.text
            page.word_count = len(vr.text.split())
            visual_pages_processed += 1

    result.total_word_count = sum(p.word_count for p in result.pages)
    result.visual_pages_processed = visual_pages_processed
