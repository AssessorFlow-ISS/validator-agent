"""Pass 2: Page classifier — determines if a page is text-only or contains visual elements.

Uses GPT-4o-mini to analyze OCR text coherence. Cheap (~$0.001/page).
Uses Pydantic structured output to enforce strict TEXT/VISUAL response schema.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field

from validator_agent.pipeline.llm_client import generate_structured

CLASSIFICATION_PROMPT = """You are analyzing OCR-extracted text from a single page of a learning document.

Your task: Determine if this page is purely text content, or if it likely contains visual elements
(diagrams, charts, flowcharts, architecture diagrams, tables-as-images, formulas, illustrations)
that the OCR could not fully capture.

Signs of a VISUAL page:
- Scattered short labels without connecting sentences (e.g. "User ID", "Policy engine", "Data plane")
- Repeated structural words (e.g. "Subnet", "PEP", "Region:")
- Arrow-like symbols or box-drawing characters
- Text that reads like diagram labels rather than prose
- Very short fragments that don't form coherent paragraphs
- Figure/diagram captions (e.g. "Figure 2.1 - ZT Policy Signals")

Signs of a TEXT page:
- Coherent sentences forming paragraphs
- Logical flow of ideas
- Standard document structure (headings, bullet points with full sentences)"""


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
            user_prompt=ocr_text[:2000],
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
