"""Unit tests for validator_agent.pipeline.pdf_to_images.

``convert_from_bytes`` is patched so tests do not require Poppler.
"""

from __future__ import annotations

from unittest.mock import patch

from validator_agent.pipeline.pdf_to_images import pdf_pages_to_images


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, buf, format: str) -> None:  # noqa: A002 - matches PIL API
        assert format == "PNG"
        buf.write(self._payload)


class TestPdfPagesToImages:
    def test_returns_png_bytes_per_page(self) -> None:
        fake_pages = [_FakeImage(b"PAGE1"), _FakeImage(b"PAGE2")]
        with patch(
            "validator_agent.pipeline.pdf_to_images.convert_from_bytes",
            return_value=fake_pages,
        ) as mock_convert:
            out = pdf_pages_to_images(b"%PDF", dpi=150)
        mock_convert.assert_called_once_with(b"%PDF", dpi=150, fmt="png")
        assert out == [b"PAGE1", b"PAGE2"]

    def test_empty_pdf_returns_empty_list(self) -> None:
        with patch(
            "validator_agent.pipeline.pdf_to_images.convert_from_bytes",
            return_value=[],
        ):
            out = pdf_pages_to_images(b"%PDF")
        assert out == []
