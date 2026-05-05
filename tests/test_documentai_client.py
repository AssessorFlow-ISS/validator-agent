"""Unit tests for validator_agent.pipeline.documentai_client.

Covers:
  - _get_mime_type for pdf/png/jpg/jpeg + unsupported extension.
  - _get_pdf_page_count good path and exception path.
  - _split_pdf: short PDF returns single chunk; long PDF is sliced.
  - _calculate_page_confidence: avg of block confidences / None when empty.
  - _extract_page_text: reads text_anchor segments from a page.
  - _parse_document_to_pages for multi-page + single-page (text-only).
  - _build_result: word count aggregation + success flag.
  - extract_with_documentai_bytes: unsupported ext, online path, chunked
    path, exception path.
  - _download_from_gcs: bad URI raises ValueError.
  - extract_with_documentai_gcs: unsupported ext + GCS failure.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pypdf import PdfWriter

from validator_agent.pipeline import documentai_client as dai


def _build_page(confidences: list[float], text_start: int = 0, text_end: int = 10):
    """Build a mock documentai.Document.Page with given block confidences."""
    blocks = []
    for c in confidences:
        blk = SimpleNamespace(layout=SimpleNamespace(confidence=c))
        blocks.append(blk)

    text_segment = SimpleNamespace(
        start_index=text_start,
        end_index=text_end,
    )
    layout = SimpleNamespace(text_anchor=SimpleNamespace(text_segments=[text_segment]))
    return SimpleNamespace(blocks=blocks, layout=layout)


def _build_document(full_text: str, pages_conf: list[list[float]] | None = None):
    """Build a mock documentai.Document with the given text + page list."""
    pages_conf = pages_conf or []
    pages = []
    offset = 0
    page_size = max(1, len(full_text) // max(1, len(pages_conf)))
    for conf_list in pages_conf:
        start = offset
        end = min(offset + page_size, len(full_text))
        pages.append(_build_page(conf_list, start, end))
        offset = end
    return SimpleNamespace(text=full_text, pages=pages)


def _one_page_pdf_bytes() -> bytes:
    """Minimal valid PDF with a single blank page."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _n_page_pdf_bytes(n: int) -> bytes:
    writer = PdfWriter()
    for _ in range(n):
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestMimeType:
    def test_pdf(self) -> None:
        assert dai._get_mime_type("file.pdf") == "application/pdf"

    def test_png(self) -> None:
        assert dai._get_mime_type("IMG.PNG") == "image/png"

    def test_jpg_and_jpeg(self) -> None:
        assert dai._get_mime_type("x.jpg") == "image/jpeg"
        assert dai._get_mime_type("y.jpeg") == "image/jpeg"

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            dai._get_mime_type("file.docx")


class TestPdfPageCount:
    def test_returns_page_count(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(3)
        assert dai._get_pdf_page_count(pdf_bytes) == 3

    def test_invalid_pdf_returns_zero(self) -> None:
        assert dai._get_pdf_page_count(b"not-a-pdf") == 0


class TestSplitPdf:
    def test_short_pdf_returned_unchanged(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(3)
        chunks = dai._split_pdf(pdf_bytes)
        assert chunks == [pdf_bytes]

    def test_long_pdf_splits_into_chunks_of_15(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(32)
        chunks = dai._split_pdf(pdf_bytes)
        # 32 pages / 15 per chunk = 3 chunks (15 + 15 + 2)
        assert len(chunks) == 3
        assert all(isinstance(c, (bytes, bytearray)) for c in chunks)


class TestPageConfidence:
    def test_avg_of_block_confidences(self) -> None:
        page = _build_page([0.8, 0.6])
        assert dai._calculate_page_confidence(page) == pytest.approx(0.7)

    def test_no_blocks_returns_none(self) -> None:
        page = SimpleNamespace(blocks=[], layout=None)
        assert dai._calculate_page_confidence(page) is None


class TestExtractPageText:
    def test_extracts_segments(self) -> None:
        full = "The quick brown fox"
        page = _build_page([0.9], text_start=4, text_end=15)
        assert dai._extract_page_text(full, page) == "quick brown"

    def test_no_layout_returns_empty(self) -> None:
        page = SimpleNamespace(blocks=[], layout=None)
        assert dai._extract_page_text("abc", page) == ""


class TestParseDocumentToPages:
    def test_multipage_document(self) -> None:
        doc = _build_document(
            "hello world another page here and more",
            pages_conf=[[0.8, 0.9], [0.7]],
        )
        pages = dai._parse_document_to_pages(doc)
        assert len(pages) == 2
        assert all(p.word_count > 0 for p in pages)
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    def test_single_page_text_only(self) -> None:
        """When Document.pages is empty but text exists, fabricate one page."""
        doc = SimpleNamespace(text="just text, no layout pages", pages=[])
        pages = dai._parse_document_to_pages(doc)
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].word_count > 0


class TestBuildResult:
    def test_zero_word_count_sets_failure(self) -> None:
        from validator_agent.pipeline.models import PageOcrResult

        pages = [PageOcrResult(page_number=1, extracted_text="", word_count=0)]
        out = dai._build_result("doc.pdf", pages, start_ms=0.0, processing_mode="online")
        assert out.success is False
        assert out.error_message == "No text extracted from any page"

    def test_populated_words_marks_success(self) -> None:
        from validator_agent.pipeline.models import PageOcrResult

        pages = [PageOcrResult(page_number=1, extracted_text="hi", word_count=1)]
        out = dai._build_result("doc.pdf", pages, start_ms=0.0, processing_mode="online")
        assert out.success is True
        assert out.total_pages == 1
        assert out.total_word_count == 1


class TestExtractWithDocumentAIBytes:
    def test_unsupported_extension_returns_error_result(self) -> None:
        out = dai.extract_with_documentai_bytes(
            file_bytes=b"raw",
            file_name="file.docx",
            project_id="p",
            location="asia-southeast1",
            processor_id="pid",
        )
        assert out.total_pages == 0
        assert out.error_message and "Unsupported" in out.error_message

    def test_short_pdf_uses_online_path(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(2)
        # Mock the processor client + online processing entirely.
        with (
            patch.object(dai.documentai, "DocumentProcessorServiceClient"),
            patch.object(dai, "_process_online") as online,
        ):
            online.return_value = [
                SimpleNamespace(
                    page_number=1,
                    extracted_text="hello",
                    word_count=1,
                    confidence=0.9,
                ),
            ]
            # Make the returned pages look like real PageOcrResult
            from validator_agent.pipeline.models import PageOcrResult

            online.return_value = [
                PageOcrResult(page_number=1, extracted_text="hello", word_count=1),
            ]
            out = dai.extract_with_documentai_bytes(
                file_bytes=pdf_bytes,
                file_name="a.pdf",
                project_id="p",
                location="loc",
                processor_id="pid",
            )

        online.assert_called_once()
        assert out.success is True
        assert out.processing_mode == "online"

    def test_long_pdf_uses_chunked_path(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(32)
        from validator_agent.pipeline.models import PageOcrResult

        def _online_side_effect(chunk_bytes, mime, resource_name, client):
            # Return pages numbered 1..N from this chunk
            return [PageOcrResult(page_number=i, extracted_text="t", word_count=1) for i in (1, 2)]

        with (
            patch.object(dai.documentai, "DocumentProcessorServiceClient"),
            patch.object(dai, "_process_online", side_effect=_online_side_effect) as online,
        ):
            out = dai.extract_with_documentai_bytes(
                file_bytes=pdf_bytes,
                file_name="big.pdf",
                project_id="p",
                location="loc",
                processor_id="pid",
            )

        # 32 pages / 15 per chunk → 3 chunks
        assert online.call_count == 3
        assert out.processing_mode.startswith("chunked")

    def test_image_uses_online_path(self) -> None:
        from validator_agent.pipeline.models import PageOcrResult

        with (
            patch.object(dai.documentai, "DocumentProcessorServiceClient"),
            patch.object(
                dai,
                "_process_online",
                return_value=[PageOcrResult(page_number=1, extracted_text="a", word_count=1)],
            ) as online,
        ):
            out = dai.extract_with_documentai_bytes(
                file_bytes=b"raw-png",
                file_name="img.png",
                project_id="p",
                location="loc",
                processor_id="pid",
            )
        online.assert_called_once()
        assert out.processing_mode == "online"

    def test_processing_exception_returns_error_result(self) -> None:
        pdf_bytes = _n_page_pdf_bytes(1)
        with (
            patch.object(dai.documentai, "DocumentProcessorServiceClient"),
            patch.object(dai, "_process_online", side_effect=RuntimeError("network")),
        ):
            out = dai.extract_with_documentai_bytes(
                file_bytes=pdf_bytes,
                file_name="a.pdf",
                project_id="p",
                location="loc",
                processor_id="pid",
            )
        assert out.total_pages == 0
        assert out.error_message and "network" in out.error_message


class TestDownloadFromGcs:
    def test_invalid_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid GCS URI"):
            dai._download_from_gcs("not-a-uri")

    def test_download_reads_blob_bytes(self) -> None:
        fake_blob = MagicMock()
        fake_blob.download_as_bytes.return_value = b"DATA"
        fake_bucket = MagicMock()
        fake_bucket.blob.return_value = fake_blob
        fake_client = MagicMock()
        fake_client.bucket.return_value = fake_bucket

        with patch("google.cloud.storage.Client", return_value=fake_client):
            out = dai._download_from_gcs("gs://bkt/path/to/file.pdf")

        assert out == b"DATA"


class TestExtractWithDocumentAiGcs:
    def test_unsupported_ext_short_circuits(self) -> None:
        out = dai.extract_with_documentai_gcs(
            gcs_input_uri="gs://b/k.docx",
            file_name="k.docx",
            project_id="p",
            location="loc",
            processor_id="pid",
        )
        assert out.total_pages == 0
        assert out.error_message and "Unsupported" in out.error_message

    def test_gcs_failure_captured_in_result(self) -> None:
        with patch.object(dai, "_download_from_gcs", side_effect=RuntimeError("gcs 403")):
            out = dai.extract_with_documentai_gcs(
                gcs_input_uri="gs://b/a.pdf",
                file_name="a.pdf",
                project_id="p",
                location="loc",
                processor_id="pid",
            )
        assert out.total_pages == 0
        assert "Failed to download from GCS" in (out.error_message or "")

    def test_delegates_to_bytes_path_on_success(self) -> None:
        from validator_agent.pipeline.models import OcrResult

        fake_result = OcrResult(file_name="a.pdf", total_pages=1, success=True)
        with (
            patch.object(dai, "_download_from_gcs", return_value=_n_page_pdf_bytes(1)),
            patch.object(dai, "extract_with_documentai_bytes", return_value=fake_result) as bp,
        ):
            out = dai.extract_with_documentai_gcs(
                gcs_input_uri="gs://b/a.pdf",
                file_name="a.pdf",
                project_id="p",
                location="loc",
                processor_id="pid",
            )
        bp.assert_called_once()
        assert out.success is True
