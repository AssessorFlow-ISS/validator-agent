"""Document AI OCR client — smart routing between online and chunked processing.

- PNG/JPG → Online processing (single page, fast)
- PDF ≤ 15 pages → Online processing (fast, immediate response)
- PDF > 15 pages → Split into 15-page chunks, process each chunk online, combine results

No batch processing needed. Online processing is always fast (~3-6s per chunk).
"""

from __future__ import annotations

import io
import re
import time

from google.cloud import documentai_v1 as documentai
from pypdf import PdfReader, PdfWriter

from validator_agent.pipeline.models import OcrResult, PageOcrResult

ONLINE_PAGE_LIMIT = 15


def _calculate_page_confidence(page: documentai.Document.Page) -> float | None:
    """Average confidence across all blocks on a page."""
    confidences = []
    for block in page.blocks:
        if block.layout and block.layout.confidence:
            confidences.append(block.layout.confidence)
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _extract_page_text(full_text: str, page: documentai.Document.Page) -> str:
    """Extract the text segment belonging to a specific page."""
    segments = []
    if page.layout and page.layout.text_anchor and page.layout.text_anchor.text_segments:
        for segment in page.layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index)
            segments.append(full_text[start:end])
    return "".join(segments)


def _get_mime_type(file_name: str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower()
    mime_map = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    mime_type = mime_map.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: pdf, png, jpg")
    return mime_type


def _get_pdf_page_count(file_bytes: bytes) -> int:
    """Get page count from PDF bytes. Fast — only reads the header."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def _split_pdf(file_bytes: bytes) -> list[bytes]:
    """Split a PDF into chunks of up to ONLINE_PAGE_LIMIT pages each."""
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)

    if total_pages <= ONLINE_PAGE_LIMIT:
        return [file_bytes]

    chunks = []
    for start in range(0, total_pages, ONLINE_PAGE_LIMIT):
        writer = PdfWriter()
        end = min(start + ONLINE_PAGE_LIMIT, total_pages)
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())

    return chunks


def _parse_document_to_pages(document: documentai.Document) -> list[PageOcrResult]:
    """Parse a Document AI Document into PageOcrResult list."""
    pages: list[PageOcrResult] = []

    for i, page in enumerate(document.pages):
        page_text = _extract_page_text(document.text, page)
        word_count = len(page_text.split()) if page_text.strip() else 0
        confidence = _calculate_page_confidence(page)

        pages.append(
            PageOcrResult(
                page_number=i + 1,
                extracted_text=page_text,
                word_count=word_count,
                confidence=round(confidence, 4) if confidence is not None else None,
            )
        )

    # For single-page images, Document AI may not populate pages list
    # but still returns text in document.text
    if not pages and document.text:
        word_count = len(document.text.split()) if document.text.strip() else 0
        pages.append(
            PageOcrResult(
                page_number=1,
                extracted_text=document.text,
                word_count=word_count,
                confidence=None,
            )
        )

    return pages


def _process_online(
    file_bytes: bytes,
    mime_type: str,
    resource_name: str,
    client: documentai.DocumentProcessorServiceClient,
) -> list[PageOcrResult]:
    """Send file bytes directly to Document AI online processing (≤ 15 pages)."""
    raw_document = documentai.RawDocument(content=file_bytes, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
    response = client.process_document(request=request)
    return _parse_document_to_pages(response.document)


def _build_result(
    file_name: str,
    all_pages: list[PageOcrResult],
    start_ms: float,
    processing_mode: str,
) -> OcrResult:
    """Build OcrResult from parsed pages."""
    total_word_count = sum(p.word_count for p in all_pages)
    elapsed_ms = time.time() * 1000 - start_ms

    return OcrResult(
        file_name=file_name,
        total_pages=len(all_pages),
        pages=all_pages,
        total_word_count=total_word_count,
        ocr_time_ms=round(elapsed_ms, 2),
        success=total_word_count > 0,
        error_message=None if total_word_count > 0 else "No text extracted from any page",
        processing_mode=processing_mode,
    )


# ── Public API ────


def extract_with_documentai_bytes(
    file_bytes: bytes,
    file_name: str,
    project_id: str,
    location: str,
    processor_id: str,
) -> OcrResult:
    """Extract text from file bytes. Smart routing:

    - Images (PNG/JPG) → online (single page)
    - PDF ≤ 15 pages → online (fast)
    - PDF > 15 pages → split into 15-page chunks, each processed online
    """
    start_ms = time.time() * 1000

    try:
        mime_type = _get_mime_type(file_name)
    except ValueError as e:
        return OcrResult(
            file_name=file_name, total_pages=0,
            ocr_time_ms=time.time() * 1000 - start_ms, error_message=str(e),
        )

    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    resource_name = client.processor_path(project_id, location, processor_id)

    ext = file_name.rsplit(".", 1)[-1].lower()

    try:
        if ext == "pdf":
            page_count = _get_pdf_page_count(file_bytes)

            if page_count <= ONLINE_PAGE_LIMIT:
                # Small PDF — single online request
                all_pages = _process_online(file_bytes, mime_type, resource_name, client)
                return _build_result(file_name, all_pages, start_ms, "online")
            else:
                # Large PDF — split into chunks, process each online
                chunks = _split_pdf(file_bytes)
                all_pages: list[PageOcrResult] = []
                page_offset = 0

                for chunk in chunks:
                    pages = _process_online(chunk, mime_type, resource_name, client)
                    for page in pages:
                        page.page_number = page_offset + page.page_number
                    all_pages.extend(pages)
                    page_offset += len(pages)

                return _build_result(file_name, all_pages, start_ms, f"chunked ({len(chunks)} chunks)")
        else:
            # Images — single online request
            all_pages = _process_online(file_bytes, mime_type, resource_name, client)
            return _build_result(file_name, all_pages, start_ms, "online")
    except Exception as e:
        return OcrResult(
            file_name=file_name, total_pages=0,
            ocr_time_ms=time.time() * 1000 - start_ms,
            error_message=f"Document AI processing error: {e}",
        )


def extract_with_documentai_gcs(
    gcs_input_uri: str,
    file_name: str,
    project_id: str,
    location: str,
    processor_id: str,
) -> OcrResult:
    """Extract text from a file in Cloud Storage. Downloads first, then smart routes."""
    start_ms = time.time() * 1000

    try:
        _get_mime_type(file_name)
    except ValueError as e:
        return OcrResult(
            file_name=file_name, total_pages=0,
            ocr_time_ms=time.time() * 1000 - start_ms, error_message=str(e),
        )

    try:
        file_bytes = _download_from_gcs(gcs_input_uri)
    except Exception as e:
        return OcrResult(
            file_name=file_name, total_pages=0,
            ocr_time_ms=time.time() * 1000 - start_ms,
            error_message=f"Failed to download from GCS: {e}",
        )

    # Reuse the bytes-based function (handles all routing logic)
    result = extract_with_documentai_bytes(
        file_bytes=file_bytes,
        file_name=file_name,
        project_id=project_id,
        location=location,
        processor_id=processor_id,
    )
    # Adjust timing to include download
    result.ocr_time_ms = round(time.time() * 1000 - start_ms, 2)
    return result


def _download_from_gcs(gcs_uri: str) -> bytes:
    """Download file bytes from GCS."""
    from google.cloud import storage

    match = re.match(r"gs://([^/]+)/(.+)", gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_path = match.group(1), match.group(2)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()
