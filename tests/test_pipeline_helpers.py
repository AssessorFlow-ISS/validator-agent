"""Unit tests for pipeline helper modules: moderation_prefilter,
page_classifier, content_analyzers, content_synthesizer, content_safety.

Each module is a thin layer over LLM calls; we mock the httpx / generate_structured
boundary and verify branching, parsing, and error-fallback paths.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from validator_agent.pipeline import (
    content_analyzers,
    content_safety as cs_mod,
    content_synthesizer,
    moderation_prefilter,
    page_classifier,
)
from validator_agent.pipeline.content_analyzers import (
    AllAnalyzerResults,
    AnalyzerResponse,
    AnalyzerResult,
)
from validator_agent.pipeline.content_synthesizer import (
    DiscardedFinding,
    SynthesizedFinding,
    SynthesizerResponse,
)
from validator_agent.pipeline.llm_client import LlmResponse
from validator_agent.pipeline.models import PageOcrResult


def _mk_page(text: str = "hello world", page_number: int = 1) -> PageOcrResult:
    return PageOcrResult(
        page_number=page_number,
        extracted_text=text,
        word_count=len(text.split()),
    )


# ---- moderation_prefilter ------------------------------------------------


class TestModerationPrefilter:
    def test_flagged_returns_categories(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "flagged": True,
            "categories": ["violence"],
        }
        with patch.object(moderation_prefilter.httpx, "post", return_value=mock_resp):
            res = moderation_prefilter.check_moderation("bad stuff")
        assert res.flagged is True
        assert res.categories == ["violence"]
        assert res.error is None

    def test_not_flagged(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"flagged": False}
        with patch.object(moderation_prefilter.httpx, "post", return_value=mock_resp):
            res = moderation_prefilter.check_moderation("benign")
        assert res.flagged is False
        assert res.categories == []

    def test_exception_is_captured_into_error_field(self) -> None:
        with patch.object(
            moderation_prefilter.httpx, "post", side_effect=RuntimeError("boom")
        ):
            res = moderation_prefilter.check_moderation("x")
        assert res.flagged is False
        assert res.error and "boom" in res.error


# ---- page_classifier -----------------------------------------------------


class TestPageClassifier:
    def test_empty_text_returns_visual(self) -> None:
        assert page_classifier.classify_page("") == "VISUAL"
        assert page_classifier.classify_page("   \n  ") == "VISUAL"

    def test_structured_llm_success_returns_value(self) -> None:
        cls = page_classifier.ClassificationResult(
            classification=page_classifier.PageType.TEXT
        )
        with patch.object(
            page_classifier, "generate_structured", return_value=(cls, LlmResponse(content=""))
        ):
            assert page_classifier.classify_page("some coherent prose") == "TEXT"

    def test_llm_error_falls_back_to_heuristic(self) -> None:
        long_prose = "The quick brown fox jumps over the lazy dog. " * 10
        with patch.object(
            page_classifier, "generate_structured", side_effect=RuntimeError("llm down")
        ):
            # Heuristic: long coherent sentences → TEXT
            assert page_classifier.classify_page(long_prose) == "TEXT"

    def test_heuristic_short_text_returns_visual(self) -> None:
        # _heuristic_classify directly (< 10 words)
        assert page_classifier._heuristic_classify("one two three") == "VISUAL"

    def test_heuristic_only_periods_returns_visual(self) -> None:
        """Purely-punctuation input has no non-empty sentences after filter.

        Twenty+ empty segments after splitting on '.' but each is filtered
        by ``.strip()`` so the sentence list is empty → VISUAL.
        """
        # 10+ "words" (after split() on whitespace) but every sentence
        # collapses to "" after strip()
        punctuation_only = ". . . . . . . . . . . . ."
        assert page_classifier._heuristic_classify(punctuation_only) == "VISUAL"

    def test_heuristic_short_sentences_classify_as_visual(self) -> None:
        # Short sentences < 5 avg words/sentence → VISUAL
        short = ". ".join(["x y"] * 10)
        assert page_classifier._heuristic_classify(short) == "VISUAL"

    def test_heuristic_long_sentences_classify_as_text(self) -> None:
        long = (
            "This is a properly structured paragraph with sensible "
            "sentences of reasonable length. Each sentence conveys "
            "meaningful educational content with several words."
        )
        assert page_classifier._heuristic_classify(long) == "TEXT"


# ---- content_analyzers ---------------------------------------------------


class TestContentAnalyzers:
    def test_format_pages_inserts_page_markers(self) -> None:
        p1 = _mk_page("first", 1)
        p2 = _mk_page("second", 2)
        formatted = content_analyzers._format_pages_for_prompt([p1, p2])
        assert "=== PAGE 1 ===" in formatted
        assert "=== PAGE 2 ===" in formatted
        assert "first" in formatted
        assert "second" in formatted

    def test_run_analyzer_success_tags_source(self) -> None:
        parsed = AnalyzerResponse(
            findings=[
                content_analyzers.AnalyzerFinding(page=1, type="harmful", detail="x"),
            ]
        )
        with patch.object(
            content_analyzers,
            "generate_structured",
            return_value=(parsed, LlmResponse(content="")),
        ):
            result = content_analyzers._run_analyzer(
                "system", "pages", "analyzer_a"
            )

        assert result.error is None
        assert len(result.findings) == 1
        assert result.findings[0]["source"] == "analyzer_a"

    def test_run_analyzer_error_propagates_to_result(self) -> None:
        with patch.object(
            content_analyzers,
            "generate_structured",
            side_effect=RuntimeError("llm fail"),
        ):
            result = content_analyzers._run_analyzer(
                "system", "pages", "analyzer_b"
            )
        assert result.error is not None
        assert "analyzer_b" in result.error

    def test_run_all_analyzers_collects_errors(self) -> None:
        def _mock(sys_prompt: str, pages_text: str, name: str):
            if name == "analyzer_b":
                return AnalyzerResult(error=f"{name}: failed")
            return AnalyzerResult(
                findings=[{"page": 1, "type": "x", "detail": "ok", "source": name}]
            )

        with patch.object(content_analyzers, "_run_analyzer", side_effect=_mock):
            out = content_analyzers.run_all_analyzers([_mk_page()])

        assert isinstance(out, AllAnalyzerResults)
        assert len(out.errors) == 1
        assert out.analyzer_b.error is not None
        assert out.analyzer_a.findings and out.analyzer_c.findings and out.analyzer_d.findings


# ---- content_synthesizer -------------------------------------------------


class TestContentSynthesizer:
    def test_synthesize_success_returns_findings(self) -> None:
        parsed = SynthesizerResponse(
            findings=[
                SynthesizedFinding(page=1, type="pii", detail="x", original="John", redacted_to="[NAME]"),
            ],
            discarded=[
                DiscardedFinding(page=1, type="x", detail="y", reason="bad"),
            ],
        )

        empty = AllAnalyzerResults()
        with patch.object(
            content_synthesizer,
            "generate_structured",
            return_value=(parsed, LlmResponse(content="")),
        ):
            result = content_synthesizer.synthesize_findings(empty, [_mk_page()])

        assert result.error is None
        assert len(result.findings) == 1
        assert len(result.discarded) == 1
        assert result.findings[0]["original"] == "John"

    def test_synthesize_error_falls_back_to_merge(self) -> None:
        # Each analyzer has one raw finding — fallback should concatenate
        analyzer_a = AnalyzerResult(findings=[{"page": 1, "type": "x", "detail": "a"}])
        analyzer_b = AnalyzerResult(findings=[{"page": 1, "type": "y", "detail": "b"}])
        ar = AllAnalyzerResults(
            analyzer_a=analyzer_a,
            analyzer_b=analyzer_b,
            analyzer_c=AnalyzerResult(),
            analyzer_d=AnalyzerResult(),
        )
        with patch.object(
            content_synthesizer,
            "generate_structured",
            side_effect=RuntimeError("down"),
        ):
            result = content_synthesizer.synthesize_findings(ar, [_mk_page()])

        assert result.error is not None
        assert "Synthesizer failed" in result.error
        assert len(result.findings) == 2
        for f in result.findings:
            assert f["confidence"] == "unverified"

    def test_findings_to_models_casts_to_finding(self) -> None:
        dicts = [{"page": 1, "type": "pii", "detail": "SSN"}]
        out = content_synthesizer.findings_to_models(dicts)
        assert len(out) == 1
        assert out[0].page == 1
        assert out[0].type == "pii"


# ---- pipeline/content_safety ---------------------------------------------


class TestPipelineContentSafety:
    def _mod_ok(self):
        return moderation_prefilter.ModerationCheckResult(flagged=False)

    def _mod_flagged(self):
        return moderation_prefilter.ModerationCheckResult(
            flagged=True, categories=["violence"]
        )

    def _empty_analyzers(self) -> AllAnalyzerResults:
        return AllAnalyzerResults()

    def test_empty_pages_returns_proceed(self) -> None:
        out = cs_mod.check_content_safety([])
        assert out.overall_status == "PROCEED"

    def test_moderation_flag_terminates_early(self) -> None:
        with patch.object(cs_mod, "check_moderation", return_value=self._mod_flagged()):
            out = cs_mod.check_content_safety([_mk_page()])
        assert out.overall_status == "TERMINATE"
        assert out.termination_reason == "HARMFUL_CONTENT"
        assert out.harmful_detected is True

    def test_all_analyzers_failed_terminates_with_unavailable(self) -> None:
        """When all 4 analyzers error → CONTENT_SAFETY_UNAVAILABLE."""
        failed = AllAnalyzerResults(
            analyzer_a=AnalyzerResult(error="a failed"),
            analyzer_b=AnalyzerResult(error="b failed"),
            analyzer_c=AnalyzerResult(error="c failed"),
            analyzer_d=AnalyzerResult(error="d failed"),
        )
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=failed),
        ):
            out = cs_mod.check_content_safety([_mk_page()])
        assert out.overall_status == "TERMINATE"
        assert out.termination_reason == "CONTENT_SAFETY_UNAVAILABLE"

    def test_harmful_finding_terminates(self) -> None:
        analyzers = self._empty_analyzers()
        synth_findings = [
            {"page": 3, "type": "harmful", "detail": "violence described"},
        ]
        synth = content_synthesizer.SynthesisResult(findings=synth_findings)
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=analyzers),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([_mk_page()])

        assert out.overall_status == "TERMINATE"
        assert out.termination_reason == "HARMFUL_CONTENT"
        assert out.harmful_detected is True
        assert "Page 3" in (out.termination_detail or "")

    def test_religious_political_terminates(self) -> None:
        analyzers = self._empty_analyzers()
        synth = content_synthesizer.SynthesisResult(
            findings=[{"page": 1, "type": "religious_political", "detail": "preach"}]
        )
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=analyzers),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([_mk_page()])
        assert out.overall_status == "TERMINATE"
        assert out.termination_reason == "RELIGIOUS_POLITICAL_VIOLATION"

    def test_pii_triggers_proceed_with_redactions_and_warnings(self) -> None:
        """PII alone → PROCEED with redactions applied, no soft warnings."""
        page = PageOcrResult(
            page_number=1,
            extracted_text="Hi John at john@x.com",
            word_count=5,
        )
        synth = content_synthesizer.SynthesisResult(
            findings=[{
                "page": 1,
                "type": "name",
                "detail": "name",
                "original": "John",
                "redacted_to": "[NAME]",
            }]
        )
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=self._empty_analyzers()),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([page])

        assert out.overall_status == "PROCEED"
        assert out.pii_detected is True
        assert "[NAME]" in out.cleaned_text
        assert any(w["type"] == "pii_redacted" for w in out.assessor_warnings)

    def test_copyright_triggers_proceed_with_warnings(self) -> None:
        synth = content_synthesizer.SynthesisResult(
            findings=[
                {"page": 2, "type": "copyright", "detail": "© ACME"},
            ]
        )
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=self._empty_analyzers()),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([_mk_page()])
        assert out.overall_status == "PROCEED_WITH_WARNINGS"
        assert out.copyright_detected is True

    def test_misinformation_triggers_proceed_with_warnings(self) -> None:
        synth = content_synthesizer.SynthesisResult(
            findings=[{"page": 1, "type": "misinformation", "detail": "fake fact"}]
        )
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=self._empty_analyzers()),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([_mk_page()])
        assert out.overall_status == "PROCEED_WITH_WARNINGS"
        assert out.misinformation_detected is True

    def test_clean_pages_return_proceed_empty_cleaned(self) -> None:
        synth = content_synthesizer.SynthesisResult(findings=[])
        with (
            patch.object(cs_mod, "check_moderation", return_value=self._mod_ok()),
            patch.object(cs_mod, "run_all_analyzers", return_value=self._empty_analyzers()),
            patch.object(cs_mod, "synthesize_findings", return_value=synth),
        ):
            out = cs_mod.check_content_safety([_mk_page("just some plain text")])
        assert out.overall_status == "PROCEED"

    def test_apply_redactions_across_multiple_pages(self) -> None:
        p1 = PageOcrResult(page_number=1, extracted_text="Alice called Bob", word_count=3)
        p2 = PageOcrResult(page_number=2, extracted_text="Carol met Dave", word_count=3)
        from validator_agent.pipeline.models import Finding
        pii = [
            Finding(page=1, type="name", detail="x", original="Alice", redacted_to="[NAME1]"),
            Finding(page=2, type="name", detail="x", original="Dave", redacted_to="[NAME2]"),
        ]
        cleaned = cs_mod._apply_redactions([p1, p2], pii)
        assert "Alice" not in cleaned
        assert "Dave" not in cleaned
        assert "[NAME1]" in cleaned
        assert "[NAME2]" in cleaned
