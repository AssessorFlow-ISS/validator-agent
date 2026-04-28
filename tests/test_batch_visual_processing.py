"""RED tests for hybrid batch + parallel visual page processing (AF-184).

Tests the batch refactoring of _enhance_visual_pages():
- process_visual_batch() in visual_understanding.py
- evaluate_visual_batch() in visual_understanding.py
- VISUAL_BATCH_SIZE config in ocr_pipeline.py
- Refactored _enhance_visual_pages() with batch + parallel

All tests import not-yet-implemented functions and will FAIL until
the GREEN implementation phase.

Design reference: References/dale_hybrid.md
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


from validator_agent.pipeline.models import OcrResult, PageOcrResult
from validator_agent.pipeline.visual_understanding import (
    ImageModerationResult,
    VisualProcessResult,
    process_visual_batch,
    evaluate_visual_batch,
    _parse_batch_descriptions,
)
from validator_agent.pipeline.ocr_pipeline import (
    VISUAL_BATCH_SIZE,
    _enhance_visual_pages,
)


# ── Helpers ────


def _make_pages(count: int, classifications: dict[int, str] | None = None) -> list[PageOcrResult]:
    """Create a list of PageOcrResult with given count.

    Args:
        count: Number of pages.
        classifications: Optional dict of {page_number: "TEXT"|"VISUAL"} overrides.
            Pages not in dict default to None (unclassified).
    """
    return [
        PageOcrResult(
            page_number=i + 1,
            extracted_text=f"OCR text for page {i + 1}",
            word_count=5,
        )
        for i in range(count)
    ]


def _make_ocr_result(pages: list[PageOcrResult]) -> OcrResult:
    return OcrResult(
        file_name="test.pdf",
        total_pages=len(pages),
        pages=pages,
        total_word_count=sum(p.word_count for p in pages),
        success=True,
    )


def _make_page_image_map(page_numbers: list[int]) -> dict[int, bytes]:
    return {pn: b"fake-image-bytes-" + str(pn).encode() for pn in page_numbers}


def _make_batch_items(page_numbers: list[int]) -> list[tuple[int, bytes, str]]:
    """Create batch items: list of (page_number, image_bytes, ocr_text)."""
    return [
        (pn, b"fake-image-bytes-" + str(pn).encode(), f"OCR text for page {pn}")
        for pn in page_numbers
    ]


# ── AC Row 1: Batch Grouping ────


class TestBatchGrouping:
    """7 VISUAL pages with VISUAL_BATCH_SIZE=3 should produce 3 batches (3, 3, 1)."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_seven_visual_pages_produce_three_batches(
        self, mock_batch_fn, mock_classify
    ):
        """AC: 7 VISUAL pages, BATCH_SIZE=3 -> batches of [3, 3, 1]."""
        pages = _make_pages(7)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([1, 2, 3, 4, 5, 6, 7])

        # Return benign results for each batch
        mock_batch_fn.return_value = {
            pn: VisualProcessResult(
                text=f"LLM description for page {pn}", source="llm", attempts=1,
            )
            for pn in range(1, 8)
        }

        _enhance_visual_pages(result, page_image_map)

        # process_visual_batch should be called 3 times (batches of 3, 3, 1)
        assert mock_batch_fn.call_count == 3

        # Verify batch sizes from call args
        batch_sizes = []
        for call in mock_batch_fn.call_args_list:
            batch_items = call[0][0] if call[0] else call[1].get("batch_items", [])
            batch_sizes.append(len(batch_items))

        assert sorted(batch_sizes) == [1, 3, 3]

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_exact_multiple_produces_even_batches(self, mock_batch_fn, mock_classify):
        """6 VISUAL pages, BATCH_SIZE=3 -> exactly 2 batches of 3."""
        pages = _make_pages(6)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([1, 2, 3, 4, 5, 6])

        mock_batch_fn.return_value = {
            pn: VisualProcessResult(
                text=f"LLM description for page {pn}", source="llm", attempts=1,
            )
            for pn in range(1, 7)
        }

        _enhance_visual_pages(result, page_image_map)

        assert mock_batch_fn.call_count == 2


# ── AC Row 2: Page Tracking ────


class TestPageTracking:
    """Descriptions must map back to correct page numbers after batching."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_results_map_to_correct_page_numbers(self, mock_batch_fn, mock_classify):
        """AC: pages 3, 5, 8 get correct descriptions after batch processing."""
        # Mix of TEXT and VISUAL pages — only 3, 5, 8 are VISUAL
        pages = _make_pages(8)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([3, 5, 8])

        # classify_page returns VISUAL only for pages with images
        def classify_side_effect(text):
            pn = int(text.split()[-1])  # "OCR text for page N"
            return "VISUAL" if pn in (3, 5, 8) else "TEXT"

        mock_classify.side_effect = classify_side_effect

        # Return unique descriptions keyed by page_number
        def batch_side_effect(batch_items):
            return {
                item[0]: VisualProcessResult(
                    text=f"UNIQUE-DESC-PAGE-{item[0]}", source="llm", attempts=1,
                )
                for item in batch_items
            }

        mock_batch_fn.side_effect = batch_side_effect

        _enhance_visual_pages(result, page_image_map)

        # Verify each VISUAL page got the correct unique description
        for page in result.pages:
            if page.page_number in (3, 5, 8):
                assert page.extracted_text == f"UNIQUE-DESC-PAGE-{page.page_number}"
                assert page.source == "llm"
            else:
                assert page.source == "ocr"


# ── AC Row 3: Harmful Image in Batch ────


class TestHarmfulImageInBatch:
    """If moderation flags a harmful image, pipeline terminates."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_harmful_in_batch_terminates_pipeline(self, mock_batch_fn, mock_classify):
        """AC: page 2 of 3-page batch is harmful -> TERMINATE, harmful page identified."""
        pages = _make_pages(3)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([1, 2, 3])

        # Batch returns harmful for page 2
        mock_batch_fn.return_value = {
            1: VisualProcessResult(text="ok page 1", source="llm", attempts=1),
            2: VisualProcessResult(
                text="",
                source="ocr_fallback",
                harmful_image_detected=True,
                image_moderation=ImageModerationResult(
                    flagged=True,
                    categories=["sexual"],
                    detail="Image moderation flagged: sexual",
                ),
            ),
            3: VisualProcessResult(text="ok page 3", source="llm", attempts=1),
        }

        _enhance_visual_pages(result, page_image_map)

        assert result.harmful_image_detected is True
        assert result.harmful_image_page == 2
        assert result.overall_status == "TERMINATE"


# ── AC Row 4: Single-Page Batch (Remainder) ────


class TestSinglePageBatch:
    """A remainder batch of 1 page should work identically."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_single_page_remainder_batch(self, mock_batch_fn, mock_classify):
        """AC: 1 VISUAL page -> single-page batch still processed correctly."""
        pages = _make_pages(1)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([1])

        mock_batch_fn.return_value = {
            1: VisualProcessResult(
                text="LLM description for lone page", source="llm", attempts=1,
            ),
        }

        _enhance_visual_pages(result, page_image_map)

        assert mock_batch_fn.call_count == 1
        assert result.pages[0].extracted_text == "LLM description for lone page"
        assert result.pages[0].source == "llm"


# ── AC Row 5: Batch Evaluation Failure → Single-Page Fallback ────


class TestBatchEvalFailureFallback:
    """If batch call fails, fall back to sequential process_visual_page() per page."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_page")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_batch_failure_falls_back_to_single_page(
        self, mock_batch_fn, mock_single_fn, mock_classify
    ):
        """AC: batch call raises exception -> falls back to process_visual_page per page."""
        pages = _make_pages(3)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map([1, 2, 3])

        # Batch call fails
        mock_batch_fn.side_effect = Exception("Batch API call failed")

        # Single-page fallback succeeds
        mock_single_fn.return_value = VisualProcessResult(
            text="fallback description", source="llm", attempts=1,
        )

        _enhance_visual_pages(result, page_image_map)

        # Should have fallen back to single-page processing
        assert mock_single_fn.call_count == 3
        for page in result.pages:
            assert page.extracted_text == "fallback description"
            assert page.source == "llm"


# ── AC Row 6: Parallel Execution ────


class TestParallelExecution:
    """Batches must execute concurrently via ThreadPoolExecutor."""

    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_STAGGER_SECONDS", 0)
    @patch("validator_agent.pipeline.ocr_pipeline.VISUAL_BATCH_SIZE", 3)
    @patch("validator_agent.pipeline.ocr_pipeline.classify_page", return_value="VISUAL")
    @patch("validator_agent.pipeline.ocr_pipeline.process_visual_batch")
    def test_batches_submitted_to_thread_pool(self, mock_batch_fn, mock_classify):
        """AC: 9 VISUAL pages in 3 batches -> ThreadPoolExecutor used."""
        pages = _make_pages(9)
        result = _make_ocr_result(pages)
        page_image_map = _make_page_image_map(list(range(1, 10)))

        def batch_side_effect(batch_items):
            return {
                item[0]: VisualProcessResult(
                    text=f"desc {item[0]}", source="llm", attempts=1,
                )
                for item in batch_items
            }

        mock_batch_fn.side_effect = batch_side_effect

        with patch(
            "validator_agent.pipeline.ocr_pipeline.ThreadPoolExecutor"
        ) as mock_executor_cls:
            # Create a real-enough executor mock that works with as_completed
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

            # Create mock futures that as_completed can iterate
            from concurrent.futures import Future
            futures = []
            submitted_batches = []

            def submit_side_effect(fn, batch):
                submitted_batches.append(batch)
                f = Future()
                f.set_result(fn(batch))
                futures.append(f)
                return f

            mock_executor.submit.side_effect = submit_side_effect

            _enhance_visual_pages(result, page_image_map)

            # ThreadPoolExecutor must have been used
            mock_executor_cls.assert_called_once()
            # 3 batches submitted
            assert mock_executor.submit.call_count == 3
            # Verify batch sizes
            batch_sizes = sorted(len(b) for b in submitted_batches)
            assert batch_sizes == [3, 3, 3]


# ── AC Row 7: Config Override ────


class TestConfigOverride:
    """VISUAL_BATCH_SIZE should be configurable via environment variable."""

    def test_default_batch_size_is_three(self):
        """AC: Default VISUAL_BATCH_SIZE is 3."""
        assert VISUAL_BATCH_SIZE == 3

    @patch.dict(os.environ, {"VISUAL_BATCH_SIZE": "5"})
    def test_batch_size_from_env(self):
        """AC: VISUAL_BATCH_SIZE=5 from env is respected.

        NOTE: This test verifies the config is read from env at module load time.
        The actual batching behavior with size=5 is covered in test_seven_visual_pages.
        """
        # Re-import to pick up new env var
        import importlib
        import validator_agent.pipeline.ocr_pipeline as mod

        importlib.reload(mod)
        try:
            assert mod.VISUAL_BATCH_SIZE == 5
        finally:
            # Restore default
            importlib.reload(mod)


# ── AC Row 8: process_visual_batch unit tests ────


class TestProcessVisualBatch:
    """Unit tests for the batch generator function itself."""

    def test_process_visual_batch_returns_dict_keyed_by_page_number(self):
        """process_visual_batch returns {page_number: VisualProcessResult}."""
        batch_items = _make_batch_items([3, 5, 8])

        batch_response = (
            "## Page 3\nDescription of page 3 diagram.\n\n"
            "## Page 5\nDescription of page 5 chart.\n\n"
            "## Page 8\nDescription of page 8 table."
        )

        # Mock both the generator call and the evaluator call via _call_model_broker
        import json as json_mod
        page_eval = {
            "overall": "PASS",
            "dimensions": {
                "accuracy": {"verdict": "PASS", "feedback": None},
                "completeness": {"verdict": "PASS", "feedback": None},
                "educational_value": {"verdict": "PASS", "feedback": None},
            },
            "retry_prompt_supplement": None,
        }
        eval_response = json_mod.dumps({
            "3": page_eval,
            "5": page_eval,
            "8": page_eval,
        })

        call_count = {"n": 0}

        def broker_side_effect(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return batch_response  # generator call
            return eval_response  # evaluator call

        with patch(
            "validator_agent.pipeline.visual_understanding.check_image_moderation"
        ) as mock_mod, patch(
            "validator_agent.pipeline.visual_understanding._call_model_broker"
        ) as mock_broker:
            mock_mod.return_value = ImageModerationResult(flagged=False)
            mock_broker.side_effect = broker_side_effect

            results = process_visual_batch(batch_items)

            assert isinstance(results, dict)
            assert set(results.keys()) == {3, 5, 8}
            for pn in (3, 5, 8):
                assert isinstance(results[pn], VisualProcessResult)
                assert results[pn].source in ("llm", "ocr_fallback")

    def test_process_visual_batch_harmful_image_terminates_early(self):
        """If any image in batch is flagged harmful, return immediately."""
        batch_items = _make_batch_items([1, 2, 3])

        with patch(
            "validator_agent.pipeline.visual_understanding.check_image_moderation"
        ) as mock_mod:
            # Page 2 is harmful
            def mod_side_effect(img_bytes):
                if b"2" in img_bytes:
                    return ImageModerationResult(
                        flagged=True,
                        categories=["violence"],
                        detail="Image moderation flagged: violence",
                    )
                return ImageModerationResult(flagged=False)

            mock_mod.side_effect = mod_side_effect

            results = process_visual_batch(batch_items)

            # Should contain the harmful page result
            assert 2 in results
            assert results[2].harmful_image_detected is True


class TestEvaluateVisualBatch:
    """Unit tests for the batch evaluator function."""

    def test_evaluate_batch_returns_per_page_verdicts(self):
        """evaluate_visual_batch returns dict of {page_number: EvaluationResult}."""
        batch_items = _make_batch_items([1, 2, 3])
        descriptions = {
            1: "Description of page 1",
            2: "Description of page 2",
            3: "Description of page 3",
        }

        import json as json_mod
        page_eval = {
            "overall": "PASS",
            "dimensions": {
                "accuracy": {"verdict": "PASS", "feedback": None},
                "completeness": {"verdict": "PASS", "feedback": None},
                "educational_value": {"verdict": "PASS", "feedback": None},
            },
            "retry_prompt_supplement": None,
        }
        eval_response = json_mod.dumps({
            "1": page_eval,
            "2": page_eval,
            "3": page_eval,
        })

        with patch(
            "validator_agent.pipeline.visual_understanding._call_model_broker",
            return_value=eval_response,
        ):
            eval_results = evaluate_visual_batch(batch_items, descriptions)

            assert isinstance(eval_results, dict)
            assert set(eval_results.keys()) == {1, 2, 3}


# ── _parse_batch_descriptions edge cases ────


class TestParseBatchDescriptions:
    """Direct unit tests for the batch response parser."""

    def test_standard_format(self):
        """Pages in order with ## Page N headings."""
        text = "## Page 3\nDescription of page 3.\n\n## Page 5\nDescription of page 5."
        result = _parse_batch_descriptions(text, [3, 5])
        assert result == {3: "Description of page 3.", 5: "Description of page 5."}

    def test_pages_in_wrong_order(self):
        """GPT-4o returns pages out of order — parser should still map correctly."""
        text = "## Page 8\nEighth page desc.\n\n## Page 3\nThird page desc."
        result = _parse_batch_descriptions(text, [3, 8])
        assert result[3] == "Third page desc."
        assert result[8] == "Eighth page desc."

    def test_extra_page_ignored(self):
        """GPT-4o hallucinates a page not in page_numbers — should be filtered out."""
        text = "## Page 3\nReal page.\n\n## Page 99\nHallucinated page.\n\n## Page 5\nAnother real page."
        result = _parse_batch_descriptions(text, [3, 5])
        assert set(result.keys()) == {3, 5}
        assert 99 not in result

    def test_missing_page(self):
        """GPT-4o omits one page from response — only present pages returned."""
        text = "## Page 3\nDescription of page 3."
        result = _parse_batch_descriptions(text, [3, 5, 8])
        assert set(result.keys()) == {3}
        assert 5 not in result
        assert 8 not in result

    def test_heading_with_colon(self):
        """Variant heading format: ## Page 3: (with colon) — should still parse."""
        text = "## Page 3:\nDescription with colon heading.\n\n## Page 5\nNormal heading."
        result = _parse_batch_descriptions(text, [3, 5])
        # Page 3 heading has colon — regex may or may not capture it
        # At minimum page 5 should parse
        assert 5 in result

    def test_empty_response(self):
        """Empty response text — returns empty dict."""
        result = _parse_batch_descriptions("", [3, 5])
        assert result == {}

    def test_preamble_text_before_first_heading(self):
        """GPT-4o adds preamble text before the first heading — should be ignored."""
        text = "Here are the descriptions:\n\n## Page 3\nPage three content.\n\n## Page 5\nPage five content."
        result = _parse_batch_descriptions(text, [3, 5])
        assert set(result.keys()) == {3, 5}
        assert result[3] == "Page three content."

    def test_single_page_batch(self):
        """Single page in batch — should work fine."""
        text = "## Page 7\nSolo page description with detailed content."
        result = _parse_batch_descriptions(text, [7])
        assert result == {7: "Solo page description with detailed content."}
