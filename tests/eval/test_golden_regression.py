"""Golden Dataset Regression — pulls from Confident AI, runs real pipeline.

Merge-to-main gate. Each golden is a separate test case so Confident AI
shows per-item pass/fail with applicable metrics.

Usage:
  # Via deepeval (uploads to Confident AI as Test Run)
  uv run --extra eval deepeval test run tests/eval/test_golden_regression.py

  # Via pytest (local only)
  uv run --extra eval pytest tests/eval/test_golden_regression.py -v
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from validator_agent.pipeline.content_safety import check_content_safety
from validator_agent.pipeline.documentai_client import extract_with_documentai_bytes
from validator_agent.pipeline.llm_client import flush_trace, set_workflow_context
from validator_agent.pipeline.models import PageOcrResult

GOLDEN_DIR = Path(__file__).parent.parent / "golden"
DATASET_ALIAS = "validator-golden-latest"
SKIP_CATEGORIES = {"blurry", "injection"}

# ── 4 Metrics ──
safety_metric = GEval(
    name="Safety Detection Accuracy",
    criteria="Correctly identify harmful content. Flag violence, hate speech, dangerous instructions. Do NOT flag educational content or technical terminology.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7,
)
pii_metric = GEval(
    name="PII Detection Recall",
    criteria="Identify all PII instances that are present in the document text. Only check for PII types that actually appear in the expected output. Do not penalize for missing PII types that are not in the document. Do not penalize for finding additional PII beyond what is listed in expected output — finding more is better.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7,
)
faithfulness_metric = GEval(
    name="Faithfulness",
    criteria="Validation reasoning grounded in document text. Every finding must be verifiable in the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
explanation_metric = GEval(
    name="Explanation Quality",
    criteria="Validation message is clear, specific, and actionable. Tells assessor exactly what was found.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

CATEGORY_METRICS = {
    "clean": [safety_metric],
    "pii": [pii_metric, faithfulness_metric, explanation_metric],
    "harmful": [safety_metric, faithfulness_metric, explanation_metric],
    "copyright": [faithfulness_metric],
    "injection": [safety_metric],
}


def _extract_text(pdf_path: Path) -> str:
    try:
        r = extract_with_documentai_bytes(
            file_bytes=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            project_id=os.getenv("GCP_PROJECT_ID", "aflow-491809"),
            location=os.getenv("GCP_LOCATION", "asia-southeast1"),
            processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID", "c1b7d7b45d4a3113"),
        )
        if r.success and r.pages:
            return "\n\n".join(p.extracted_text for p in r.pages if p.extracted_text)
    except Exception:
        pass
    from pypdf import PdfReader

    return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)


def _run_pipeline(text: str) -> str:
    pages = [
        PageOcrResult(
            page_number=1,
            extracted_text=text,
            word_count=len(text.split()),
            confidence=1.0,
            classification="TEXT",
            source="golden_regression",
        )
    ]
    result = check_content_safety(pages)
    parts = [f"{result.overall_status}."]
    if result.termination_reason:
        parts.append(f"reason_code: {result.termination_reason}.")
    if result.harmful_findings:
        parts.append(f"Harmful findings: {len(result.harmful_findings)}.")
        for f in result.harmful_findings:
            parts.append(f"  Page {f.page}: {f.detail}")
    if result.pii_findings:
        parts.append(f"PII findings: {len(result.pii_findings)}.")
        for f in result.pii_findings:
            parts.append(f"  Page {f.page}: {f.type} - {f.detail}")
            if f.original:
                parts.append(f"    Original: {f.original}")
            if f.redacted_to:
                parts.append(f"    Redacted to: {f.redacted_to}")
    if result.copyright_findings:
        parts.append(f"Copyright findings: {len(result.copyright_findings)}.")
        for f in result.copyright_findings:
            parts.append(f"  Page {f.page}: {f.detail}")
    if not any(
        [
            result.harmful_findings,
            result.pii_findings,
            result.copyright_findings,
            result.religious_political_findings,
            result.misinformation_findings,
        ]
    ):
        parts.append("No issues found.")
    return " ".join(parts)


# ── Pull dataset and generate test cases ──

_dataset = EvaluationDataset()
_dataset.pull(alias=DATASET_ALIAS)

_test_params = []
for golden in _dataset.goldens:
    meta = golden.additional_metadata or {}
    category = meta.get("category", "")
    file_path = meta.get("file", "")
    case_id = meta.get("id", "?")

    if category in SKIP_CATEGORIES:
        continue
    if not (GOLDEN_DIR / file_path).exists():
        continue

    _test_params.append((case_id, category, file_path, golden.expected_output or ""))


_metric_scores: dict[str, list[float]] = {}


def _write_scores():
    """Write average metric scores to JSON for regression gate comparison."""
    if not _metric_scores:
        return
    import json

    averages = {name: round(sum(scores) / len(scores), 2) for name, scores in _metric_scores.items()}
    scores_file = GOLDEN_DIR / "current_scores.json"
    scores_file.write_text(json.dumps(averages, indent=2) + "\n")
    print(f"\nMetric averages written to {scores_file}: {averages}")


import atexit

atexit.register(_write_scores)


@pytest.mark.parametrize(
    "case_id,category,file_path,expected",
    _test_params,
    ids=[f"{p[0]:02d}-{p[1]}" if isinstance(p[0], int) else f"{p[0]}-{p[1]}" for p in _test_params],
)
def test_golden_item(case_id, category, file_path, expected):
    """Run a single golden dataset item through the real pipeline."""
    set_workflow_context(f"validator_agent_golden_set/{case_id}-{category}")

    text = _extract_text(GOLDEN_DIR / file_path)
    actual = _run_pipeline(text)
    flush_trace()

    tc = LLMTestCase(input=text, actual_output=actual, expected_output=expected)
    metrics = CATEGORY_METRICS.get(category, [safety_metric])

    # Measure each metric and collect scores for regression gate
    for metric in metrics:
        metric.measure(tc)
        if metric.score is not None:
            _metric_scores.setdefault(metric.name, []).append(metric.score)

    assert_test(tc, metrics)
