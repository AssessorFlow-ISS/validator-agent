"""Pass 2: Page classifier — determines if a page is text-only or contains visual elements.

Uses GPT-4o-mini to analyze OCR text coherence. Cheap (~$0.001/page).
Uses Pydantic structured output to enforce strict TEXT/VISUAL response schema.

Prompt template loaded from ``prompts/page_classifier.yaml`` (ADR-39).
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field

from validator_agent.pipeline.llm_client import generate_structured
from validator_agent.pipeline.prompt_loader import load_prompt

# ── Load prompt template (ADR-39) ────

_CLASSIFIER_PROMPT, _CLASSIFIER_META = load_prompt("page_classifier")
_MAX_INPUT_CHARS = int(_CLASSIFIER_META.get("max_input_chars", 2000))

# Public alias for backward compatibility
CLASSIFICATION_PROMPT = _CLASSIFIER_PROMPT


class PageType(str, Enum):
    TEXT = "TEXT"
    VISUAL = "VISUAL"


class ClassificationResult(BaseModel):
    classification: PageType = Field(description="TEXT if coherent prose, VISUAL if scattered labels or diagram content")


def classify_page(ocr_text: str) -> str:
    """Classify a page as 'TEXT' or 'VISUAL' based on its OCR output.

    Uses OpenAI structured output with Pydantic schema — guaranteed to return
    only TEXT or VISUAL, no parsing needed.

    Args:
        ocr_text: The OCR-extracted text from the page.

    Returns:
        'TEXT' or 'VISUAL'
    """
    if not ocr_text.strip():
        return "VISUAL"

    try:
        result, _resp = generate_structured(
            task_key="validator.page_classification",
            system_prompt=CLASSIFICATION_PROMPT,
            user_prompt=ocr_text[:_MAX_INPUT_CHARS],
            response_model=ClassificationResult,
            temperature=0,
            prompt_version="validator/page_classifier@v1",
        )
        return result.classification.value
    except Exception:
        return _heuristic_classify(ocr_text)


def _heuristic_classify(ocr_text: str) -> str:
    """Simple fallback heuristic when OpenAI is unavailable."""
    words = ocr_text.split()
    if len(words) < 10:
        return "VISUAL"

    sentences = [s.strip() for s in ocr_text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return "VISUAL"

    avg_words_per_sentence = len(words) / len(sentences)
    return "VISUAL" if avg_words_per_sentence < 5 else "TEXT"
