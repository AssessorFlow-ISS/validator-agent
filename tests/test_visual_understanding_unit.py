"""Unit tests for validator_agent.pipeline.visual_understanding.

Focuses on the single-page Generator+Evaluator loop and helpers that are
not covered by test_batch_visual_processing.py. All network calls are
mocked via ``httpx.post`` so these are pure unit tests.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from validator_agent.pipeline.visual_understanding import (
    DimensionResult,
    EvaluationResult,
    ImageModerationResult,
    Verdict,
    _call_model_broker,
    _extract_text_from_messages,
    _format_evaluation_feedback,
    check_image_moderation,
    evaluate_description,
    generate_visual_description,
    process_visual_page,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


def _eval_pass_json() -> str:
    return json.dumps(
        {
            "overall": "PASS",
            "dimensions": {
                "accuracy": {"verdict": "PASS", "feedback": None},
                "completeness": {"verdict": "PASS", "feedback": None},
                "educational_value": {"verdict": "PASS", "feedback": None},
            },
            "retry_prompt_supplement": None,
        }
    )


def _eval_fail_json() -> str:
    return json.dumps(
        {
            "overall": "FAIL",
            "dimensions": {
                "accuracy": {"verdict": "FAIL", "feedback": "wrong"},
                "completeness": {"verdict": "PASS", "feedback": None},
                "educational_value": {"verdict": "PASS", "feedback": None},
            },
            "retry_prompt_supplement": "add numbers",
        }
    )


def _mock_response(json_payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    return resp


# ── helper functions ──────────────────────────────────────────────────


class TestExtractTextFromMessages:
    def test_string_content(self) -> None:
        out = _extract_text_from_messages([{"role": "user", "content": "hello"}])
        assert out == "hello"

    def test_list_content_with_text_parts(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        out = _extract_text_from_messages(msgs)
        assert "first" in out and "second" in out
        assert "..." not in out

    def test_truncates_at_4000(self) -> None:
        msg_text = "x" * 8000
        out = _extract_text_from_messages([{"role": "user", "content": msg_text}])
        assert len(out) == 4000


class TestFormatEvaluationFeedback:
    def test_failed_dimension_renders_feedback_and_supplement(self) -> None:
        ev = EvaluationResult(
            overall=Verdict.FAIL,
            dimensions={
                "accuracy": DimensionResult(verdict=Verdict.FAIL, feedback="bad"),
                "completeness": DimensionResult(verdict=Verdict.PASS, feedback=None),
            },
            retry_prompt_supplement="try harder",
        )
        text = _format_evaluation_feedback(ev)
        assert "ACCURACY" in text
        assert "FAIL" in text
        assert "bad" in text
        assert "COMPLETENESS" in text
        assert "try harder" in text


# ── _call_model_broker ──────────────────────────────────────────────


class TestCallModelBroker:
    def test_posts_expected_payload(self) -> None:
        with patch("validator_agent.pipeline.visual_understanding.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"content": "out"})
            out = _call_model_broker(
                [{"role": "user", "content": "hi"}],
                task_key="t",
                max_tokens=5,
                temperature=0.1,
                workflow_id="wf-1",
                response_format="json",
                response_schema={"type": "object"},
            )
            assert out == "out"
            body = mock_post.call_args[1]["json"]
            assert body["task_key"] == "t"
            assert body["session_id"] == "wf-1"
            assert body["max_tokens"] == 5
            assert body["temperature"] == 0.1
            assert body["response_format"] == "json"
            assert body["response_schema"] == {"type": "object"}
            assert body["agent_id"] == "validator-agent"

    def test_raises_on_error_status(self) -> None:
        with patch("validator_agent.pipeline.visual_understanding.httpx.post") as mock_post:
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            mock_post.return_value = resp
            with pytest.raises(httpx.HTTPStatusError):
                _call_model_broker([{"role": "user", "content": "hi"}], task_key="t")


# ── check_image_moderation ────────────────────────────────────────


class TestCheckImageModeration:
    def test_not_flagged(self) -> None:
        with patch("validator_agent.pipeline.visual_understanding.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"flagged": False, "categories": []})
            result = check_image_moderation(PNG_BYTES)
            assert isinstance(result, ImageModerationResult)
            assert result.flagged is False
            assert result.detail is None

    def test_flagged_populates_detail(self) -> None:
        with patch("validator_agent.pipeline.visual_understanding.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(
                {"flagged": True, "categories": ["sexual", "violent"]}
            )
            result = check_image_moderation(PNG_BYTES)
            assert result.flagged is True
            assert "sexual" in (result.detail or "")
            assert "violent" in (result.detail or "")

    def test_network_error_returns_error_result(self) -> None:
        with patch("validator_agent.pipeline.visual_understanding.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("down")
            result = check_image_moderation(PNG_BYTES)
            assert result.flagged is False
            assert result.error is not None


# ── generate_visual_description ─────────────────────────────────────


class TestGenerateVisualDescription:
    def test_initial_call_builds_user_content_default(self) -> None:
        with patch(
            "validator_agent.pipeline.visual_understanding._call_model_broker",
            return_value=" described ",
        ) as mock_broker:
            out = generate_visual_description(PNG_BYTES, "ocr text")
            assert out == "described"
            messages = mock_broker.call_args[0][0]
            assert messages[0]["role"] == "system"
            # user content contains the default prompt
            assert isinstance(messages[1]["content"], list)

    def test_retry_call_injects_feedback(self) -> None:
        with patch(
            "validator_agent.pipeline.visual_understanding._call_model_broker",
            return_value="retried",
        ) as mock_broker:
            out = generate_visual_description(
                PNG_BYTES,
                "ocr",
                previous_output="prev",
                retry_feedback="the evaluator said fix X",
            )
            assert out == "retried"
            messages = mock_broker.call_args[0][0]
            user_text = messages[1]["content"][0]["text"]
            assert "prev" in user_text


# ── evaluate_description ────────────────────────────────────────────


class TestEvaluateDescription:
    def test_returns_parsed_evaluation_result(self) -> None:
        with patch(
            "validator_agent.pipeline.visual_understanding._call_model_broker",
            return_value=_eval_pass_json(),
        ):
            ev = evaluate_description(PNG_BYTES, "desc")
            assert ev.overall == Verdict.PASS
            assert set(ev.dimensions.keys()) == {
                "accuracy",
                "completeness",
                "educational_value",
            }


# ── process_visual_page ─────────────────────────────────────────────


class TestProcessVisualPage:
    def test_harmful_image_short_circuits(self) -> None:
        with patch(
            "validator_agent.pipeline.visual_understanding.check_image_moderation",
            return_value=ImageModerationResult(flagged=True, categories=["x"], detail="nude"),
        ):
            result = process_visual_page(PNG_BYTES, "ocr")
            assert result.harmful_image_detected is True
            assert result.source == "ocr_fallback"
            assert result.text == "ocr"

    def test_generator_exception_returns_ocr_fallback(self) -> None:
        with (
            patch(
                "validator_agent.pipeline.visual_understanding.check_image_moderation",
                return_value=ImageModerationResult(flagged=False),
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.generate_visual_description",
                side_effect=RuntimeError("broker down"),
            ),
        ):
            result = process_visual_page(PNG_BYTES, "ocr-text")
            assert result.source == "ocr_fallback"
            assert result.attempts == 1
            assert "broker down" in (result.error or "")

    def test_pass_on_first_attempt(self) -> None:
        with (
            patch(
                "validator_agent.pipeline.visual_understanding.check_image_moderation",
                return_value=ImageModerationResult(flagged=False),
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.generate_visual_description",
                return_value="good output",
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.evaluate_description",
                return_value=EvaluationResult(
                    overall=Verdict.PASS,
                    dimensions={
                        "accuracy": DimensionResult(verdict=Verdict.PASS, feedback=None),
                        "completeness": DimensionResult(verdict=Verdict.PASS, feedback=None),
                        "educational_value": DimensionResult(verdict=Verdict.PASS, feedback=None),
                    },
                ),
            ),
        ):
            result = process_visual_page(PNG_BYTES, "ocr")
            assert result.source == "llm"
            assert result.text == "good output"
            assert result.attempts == 1

    def test_evaluator_exception_breaks_loop_and_falls_back(self) -> None:
        with (
            patch(
                "validator_agent.pipeline.visual_understanding.check_image_moderation",
                return_value=ImageModerationResult(flagged=False),
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.generate_visual_description",
                return_value="gen",
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.evaluate_description",
                side_effect=RuntimeError("eval crash"),
            ),
        ):
            result = process_visual_page(PNG_BYTES, "ocr")
            assert result.source == "ocr_fallback"
            assert "max retries" in (result.error or "")

    def test_exhaust_retries_falls_back_to_ocr(self) -> None:
        fail_eval = EvaluationResult(
            overall=Verdict.FAIL,
            dimensions={
                "accuracy": DimensionResult(verdict=Verdict.FAIL, feedback="nope"),
                "completeness": DimensionResult(verdict=Verdict.PASS, feedback=None),
                "educational_value": DimensionResult(verdict=Verdict.PASS, feedback=None),
            },
            retry_prompt_supplement="try again",
        )
        with (
            patch(
                "validator_agent.pipeline.visual_understanding.check_image_moderation",
                return_value=ImageModerationResult(flagged=False),
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.generate_visual_description",
                return_value="gen",
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.evaluate_description",
                return_value=fail_eval,
            ),
        ):
            result = process_visual_page(PNG_BYTES, "ocr")
            assert result.source == "ocr_fallback"
            assert len(result.evaluations) >= 1
            assert "max retries" in (result.error or "")

    def test_retry_generator_exception_breaks(self) -> None:
        fail_eval = EvaluationResult(
            overall=Verdict.FAIL,
            dimensions={
                "accuracy": DimensionResult(verdict=Verdict.FAIL, feedback="nope"),
                "completeness": DimensionResult(verdict=Verdict.PASS, feedback=None),
                "educational_value": DimensionResult(verdict=Verdict.PASS, feedback=None),
            },
        )
        # First generator call succeeds, first eval FAILs, retry generator raises
        gen_calls = [
            lambda *a, **kw: "first-gen",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("retry boom")),
        ]
        call_iter = iter(gen_calls)

        def gen_side_effect(*a, **kw):
            return next(call_iter)(*a, **kw)

        with (
            patch(
                "validator_agent.pipeline.visual_understanding.check_image_moderation",
                return_value=ImageModerationResult(flagged=False),
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.generate_visual_description",
                side_effect=gen_side_effect,
            ),
            patch(
                "validator_agent.pipeline.visual_understanding.evaluate_description",
                return_value=fail_eval,
            ),
        ):
            result = process_visual_page(PNG_BYTES, "ocr")
            assert result.source == "ocr_fallback"
            # Last eval entry should carry the retry-failure record
            assert any("Retry failed" in str(e.get("error", "")) for e in result.evaluations)
