"""StubOcrAdapter — returns pre-extracted text from fixtures or default text.

For testing, the adapter can be configured with per-file text overrides
or a default text string.  When a fixture_dir is provided, it looks for
a .txt file matching the storage_path filename.
"""

from __future__ import annotations

from pathlib import Path

from validator_agent.ports.ocr_port import OcrPort, OcrResult

_DEFAULT_TEXT = (
    "This is a sample educational document about software engineering principles. "
    "It covers topics such as design patterns, testing strategies, and agile methodologies. "
    "Students should review chapters 1 through 5 before the assessment."
)


class StubOcrAdapter(OcrPort):
    """Stub adapter for OCR text extraction.

    Args:
        default_text: Default extracted text when no override or fixture matches.
        overrides: Per-file text overrides keyed by storage_path.
        fixture_dir: Optional directory to look up .txt fixture files.
    """

    def __init__(
        self,
        *,
        default_text: str = _DEFAULT_TEXT,
        overrides: dict[str, str] | None = None,
        fixture_dir: Path | None = None,
    ) -> None:
        self._default_text = default_text
        self._overrides = overrides or {}
        self._fixture_dir = fixture_dir

    async def extract_text(self, storage_path: str) -> OcrResult:
        """Return pre-extracted text, checking overrides then fixtures then default."""
        # Check explicit per-file overrides first
        if storage_path in self._overrides:
            return OcrResult(
                text=self._overrides[storage_path],
                source_type="ocr_extracted",
            )

        # Check fixture directory for a matching .txt file
        if self._fixture_dir is not None:
            filename = Path(storage_path).stem + ".txt"
            fixture_path = self._fixture_dir / filename
            if fixture_path.exists():
                text = fixture_path.read_text(encoding="utf-8")
                return OcrResult(text=text, source_type="ocr_extracted")

        return OcrResult(
            text=self._default_text,
            source_type="ocr_extracted",
        )
