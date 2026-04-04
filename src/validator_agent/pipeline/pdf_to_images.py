"""Convert PDF pages to PNG images for GPT-4o visual processing."""

from __future__ import annotations

import io

from pdf2image import convert_from_bytes


def pdf_pages_to_images(file_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Convert each PDF page to a PNG image.

    Args:
        file_bytes: Raw PDF content.
        dpi: Resolution for rendering. 200 is good balance of quality vs size.

    Returns:
        List of PNG bytes, one per page.
    """
    images = convert_from_bytes(file_bytes, dpi=dpi, fmt="png")
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result
