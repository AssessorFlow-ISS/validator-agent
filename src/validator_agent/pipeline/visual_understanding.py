"""Pass 3: Visual understanding — GPT-4o generator + multi-dimension evaluator.

For pages classified as VISUAL, sends the full page image + OCR text to GPT-4o
to produce a complete, context-aware rewrite that integrates text and visual descriptions.

Uses Evaluator-Optimizer pattern with structured per-dimension feedback.

Also includes image moderation via OpenAI Moderation API — catches harmful/sexual/violent
imagery BEFORE spending on the generator. If flagged, the page is marked as harmful
and the entire file terminates (no need to proceed to Component 3).

NOTE: This file uses OpenAI directly (not Model Broker) because the generator
and evaluator send page IMAGES to GPT-4o. Model Broker does not yet support
multimodal image payloads. Image moderation also stays direct (free API, not LLM).
TODO: Route through Model Broker when multimodal support is added.

Prompt templates loaded from ``prompts/visual_generator.yaml`` and
``prompts/visual_evaluator.yaml`` (ADR-39).
"""

from __future__ import annotations

import base64
import os
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field

from validator_agent.pipeline.prompt_loader import load_prompt

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o")

# ── Load prompt templates (ADR-39) ────

_GENERATOR_PROMPT, _GENERATOR_META = load_prompt("visual_generator")
_EVALUATOR_PROMPT, _EVALUATOR_META = load_prompt("visual_evaluator")

MAX_RETRIES = int(_GENERATOR_META.get("max_retries", 2))

# Public aliases for backward compatibility
GENERATOR_SYSTEM_PROMPT = _GENERATOR_PROMPT
GENERATOR_RETRY_PROMPT = _GENERATOR_META.get("retry_template", "").strip()
EVALUATOR_SYSTEM_PROMPT = _EVALUATOR_PROMPT


# ── Pydantic models for structured evaluator output ────

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class DimensionResult(BaseModel):
    verdict: Verdict
    feedback: str | None = Field(default=None, description="Specific issue if FAIL, null if PASS")


class EvaluationResult(BaseModel):
    overall: Verdict
    dimensions: dict[str, DimensionResult] = Field(
        description="Three dimensions: accuracy, completeness, educational_value"
    )
    retry_prompt_supplement: str | None = Field(
        default=None,
        description="Concise fix instructions for the generator if FAIL"
    )


class ImageModerationResult(BaseModel):
    flagged: bool = False
    categories: list[str] = Field(default_factory=list)
    detail: str | None = None
    error: str | None = None


class VisualProcessResult(BaseModel):
    text: str
    source: str  # "llm" or "ocr_fallback"
    attempts: int = 0
    evaluations: list = Field(default_factory=list)
    image_moderation: ImageModerationResult | None = None
    harmful_image_detected: bool = False
    error: str | None = None


def _encode_image_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def check_image_moderation(page_image_bytes: bytes) -> ImageModerationResult:
    """Run OpenAI Moderation API on a page image.

    Catches harmful, sexual, violent, or hateful imagery before
    spending on the GPT-4o generator.
    """
    if not OPENAI_API_KEY:
        return ImageModerationResult(error="No OPENAI_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)
    image_b64 = _encode_image_b64(page_image_bytes)

    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        )

        result = response.results[0]
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items() if flagged
        ]

        return ImageModerationResult(
            flagged=result.flagged,
            categories=flagged_categories,
            detail=f"Image moderation flagged: {', '.join(flagged_categories)}" if result.flagged else None,
        )
    except Exception as e:
        return ImageModerationResult(error=f"Image moderation error: {e}")


def generate_visual_description(
    page_image_bytes: bytes,
    ocr_text: str,
    previous_output: str | None = None,
    retry_feedback: str | None = None,
) -> str:
    """Generate a description of a visual page using GPT-4o.

    Args:
        page_image_bytes: PNG bytes of the page image.
        ocr_text: OCR-extracted text from this page.
        previous_output: If retrying, the previous generator output.
        retry_feedback: If retrying, the evaluator's structured feedback.

    Returns:
        Generated text description of the page.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    image_b64 = _encode_image_b64(page_image_bytes)

    if previous_output and retry_feedback:
        user_content = GENERATOR_RETRY_PROMPT.format(evaluation_feedback=retry_feedback)
        user_content += f"\n\nYour previous output:\n{previous_output}"
    else:
        user_content = "Please analyze this page and produce the rewrite as instructed."

    system_prompt = GENERATOR_SYSTEM_PROMPT.format(ocr_text=ocr_text[:3000])

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def evaluate_description(
    page_image_bytes: bytes,
    generated_description: str,
) -> dict:
    """Evaluate a generated description against the original page image.

    Args:
        page_image_bytes: PNG bytes of the page image.
        generated_description: The generator's output to evaluate.

    Returns:
        Structured evaluation dict with per-dimension verdicts.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    image_b64 = _encode_image_b64(page_image_bytes)

    response = client.beta.chat.completions.parse(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Description to evaluate:\n\n{generated_description}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format=EvaluationResult,
        temperature=0,
    )

    return response.choices[0].message.parsed


def _format_evaluation_feedback(evaluation: EvaluationResult) -> str:
    """Format evaluation result into readable feedback for the generator."""
    lines = []
    for dim_name, dim_result in evaluation.dimensions.items():
        icon = "✅" if dim_result.verdict == Verdict.PASS else "❌"
        line = f"{icon} {dim_name.upper()} — {dim_result.verdict.value}"
        if dim_result.feedback:
            line += f": {dim_result.feedback}"
        lines.append(line)

    if evaluation.retry_prompt_supplement:
        lines.append(f"\nFocus on: {evaluation.retry_prompt_supplement}")

    return "\n".join(lines)


def process_visual_page(
    page_image_bytes: bytes,
    ocr_text: str,
) -> VisualProcessResult:
    """Process a visual page through the Generator + Evaluator-Optimizer loop."""
    if not OPENAI_API_KEY:
        return VisualProcessResult(
            text=ocr_text,
            source="ocr_fallback",
            error="No OPENAI_API_KEY set",
        )

    # ── Image moderation check (before spending on generator) ────
    moderation = check_image_moderation(page_image_bytes)
    if moderation.flagged:
        return VisualProcessResult(
            text=ocr_text,
            source="ocr_fallback",
            image_moderation=moderation,
            harmful_image_detected=True,
        )

    evaluations = []

    # Initial generation
    try:
        generated = generate_visual_description(page_image_bytes, ocr_text)
    except Exception as e:
        return VisualProcessResult(
            text=ocr_text,
            source="ocr_fallback",
            attempts=1,
            error=f"Generator failed: {e}",
        )

    # Evaluate + retry loop
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            evaluation = evaluate_description(page_image_bytes, generated)
        except Exception as e:
            evaluations.append({"attempt": attempt, "error": str(e)})
            break

        evaluations.append({"attempt": attempt, **evaluation.model_dump()})

        if evaluation.overall == Verdict.PASS:
            return VisualProcessResult(
                text=generated,
                source="llm",
                attempts=attempt,
                evaluations=evaluations,
            )

        # Max retries reached?
        if attempt > MAX_RETRIES:
            break

        # Retry with structured feedback
        feedback = _format_evaluation_feedback(evaluation)
        try:
            generated = generate_visual_description(
                page_image_bytes, ocr_text,
                previous_output=generated,
                retry_feedback=feedback,
            )
        except Exception as e:
            evaluations.append({"attempt": attempt + 1, "error": f"Retry failed: {e}"})
            break

    # Exhausted retries — fall back to OCR
    return VisualProcessResult(
        text=ocr_text,
        source="ocr_fallback",
        attempts=len(evaluations),
        evaluations=evaluations,
        error="Evaluator did not pass after max retries",
    )
