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
import json
import logging
import os
import re
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field

from validator_agent.pipeline.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

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


# ── Batch processing (AF-184) ────


_BATCH_GENERATOR_PREAMBLE = (
    "You are analyzing MULTIPLE pages from an educational learning document.\n"
    "Each page is provided as a separate image. For EACH page, produce a complete "
    "description following the instructions below.\n\n"
    "CRITICAL: Label each page description with a markdown heading: ## Page {N}\n"
    "where {N} is the page number provided. Output descriptions in page order.\n\n"
)

_BATCH_EVALUATOR_PREAMBLE = (
    "You are evaluating descriptions of MULTIPLE pages from a learning document.\n"
    "Each page image is provided alongside its generated description.\n\n"
    "For EACH page, evaluate on 3 dimensions: accuracy, completeness, educational_value.\n"
    "Label each evaluation with: ## Page {N}\n"
    "Then provide a JSON block for that page's evaluation.\n\n"
)


def _parse_batch_descriptions(response_text: str, page_numbers: list[int]) -> dict[int, str]:
    """Parse a batch GPT-4o response into per-page descriptions.

    Expects the response to contain ``## Page N`` headings.
    Returns a dict of {page_number: description_text}.
    """
    results: dict[int, str] = {}

    # Split on ## Page N headings
    pattern = r"##\s*Page\s*(\d+)"
    parts = re.split(pattern, response_text)

    # parts alternates: [preamble, page_num, content, page_num, content, ...]
    i = 1
    while i < len(parts) - 1:
        try:
            pn = int(parts[i])
            content = parts[i + 1].strip()
            if pn in page_numbers:
                results[pn] = content
        except (ValueError, IndexError):
            pass
        i += 2

    return results


def process_visual_batch(
    batch_items: list[tuple[int, bytes, str]],
) -> dict[int, VisualProcessResult]:
    """Process multiple visual pages in a single GPT-4o call.

    Args:
        batch_items: List of (page_number, page_image_bytes, ocr_text) tuples.

    Returns:
        Dict of {page_number: VisualProcessResult}.
    """
    if not OPENAI_API_KEY:
        return {
            pn: VisualProcessResult(text=ocr, source="ocr_fallback", error="No OPENAI_API_KEY set")
            for pn, _, ocr in batch_items
        }

    page_numbers = [item[0] for item in batch_items]
    results: dict[int, VisualProcessResult] = {}

    # Phase 1: Moderation check on ALL images (cheap, sequential)
    for pn, img_bytes, ocr_text in batch_items:
        moderation = check_image_moderation(img_bytes)
        if moderation.flagged:
            results[pn] = VisualProcessResult(
                text=ocr_text,
                source="ocr_fallback",
                image_moderation=moderation,
                harmful_image_detected=True,
            )
            # Return immediately — harmful image terminates the batch
            return results

    # Phase 2: Single GPT-4o call for all pages in batch
    ocr_context = "\n\n".join(
        f"--- Page {pn} OCR ---\n{ocr[:3000]}"
        for pn, _, ocr in batch_items
    )
    system_prompt = _BATCH_GENERATOR_PREAMBLE + GENERATOR_SYSTEM_PROMPT.format(ocr_text=ocr_context)

    user_content: list[dict] = [{"type": "text", "text": "Analyze these pages:"}]
    for pn, img_bytes, _ in batch_items:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_encode_image_b64(img_bytes)}",
                "detail": "high",
            },
        })

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=min(2000 * len(batch_items), 16000),
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Batch generator failed: %s — falling back to single-page", e)
        return {
            pn: VisualProcessResult(text=ocr, source="ocr_fallback", attempts=1, error=f"Batch generator failed: {e}")
            for pn, _, ocr in batch_items
        }

    # Phase 3: Parse per-page descriptions
    descriptions = _parse_batch_descriptions(raw_text, page_numbers)

    # Phase 4: Evaluate all descriptions in one call
    eval_results = evaluate_visual_batch(batch_items, descriptions)

    # Phase 5: Build results — retry failed pages via single-page fallback
    for pn, img_bytes, ocr_text in batch_items:
        if pn not in descriptions:
            # Parser couldn't extract this page — fall back to single-page
            logger.warning("Batch parse missing page %d — falling back to single-page", pn)
            results[pn] = process_visual_page(img_bytes, ocr_text)
            continue

        page_eval = eval_results.get(pn)
        if page_eval and page_eval.overall == Verdict.PASS:
            results[pn] = VisualProcessResult(
                text=descriptions[pn],
                source="llm",
                attempts=1,
                evaluations=[{"attempt": 1, **page_eval.model_dump()}],
            )
        else:
            # Evaluation failed or missing — retry via single-page
            logger.info("Page %d failed batch evaluation — retrying single-page", pn)
            results[pn] = process_visual_page(img_bytes, ocr_text)

    return results


def evaluate_visual_batch(
    batch_items: list[tuple[int, bytes, str]],
    descriptions: dict[int, str],
) -> dict[int, EvaluationResult]:
    """Evaluate multiple page descriptions in a single GPT-4o call.

    Args:
        batch_items: List of (page_number, page_image_bytes, ocr_text) tuples.
        descriptions: Dict of {page_number: generated_description}.

    Returns:
        Dict of {page_number: EvaluationResult}.
    """
    if not OPENAI_API_KEY:
        return {}

    page_numbers = [item[0] for item in batch_items if item[0] in descriptions]
    if not page_numbers:
        return {}

    # Build evaluation prompt with all descriptions
    desc_text = "\n\n".join(
        f"## Page {pn}\n{descriptions[pn]}"
        for pn in page_numbers
    )

    user_content: list[dict] = [
        {"type": "text", "text": f"Descriptions to evaluate:\n\n{desc_text}"},
    ]
    for pn, img_bytes, _ in batch_items:
        if pn in descriptions:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image_b64(img_bytes)}",
                    "detail": "high",
                },
            })

    eval_system = (
        _BATCH_EVALUATOR_PREAMBLE + EVALUATOR_SYSTEM_PROMPT + "\n\n"
        "Output a single JSON object with page numbers as keys. Example:\n"
        '{"3": {"overall": "PASS", "dimensions": {"accuracy": {"verdict": "PASS", "feedback": null}, '
        '"completeness": {"verdict": "PASS", "feedback": null}, '
        '"educational_value": {"verdict": "PASS", "feedback": null}}, '
        '"retry_prompt_supplement": null}, "5": {...}}'
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": eval_system},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=min(1500 * len(page_numbers), 12000),
            temperature=0,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Batch evaluator failed: %s", e)
        return {}

    # Parse JSON response — keyed by page number string
    results: dict[int, EvaluationResult] = {}
    try:
        parsed = json.loads(raw_text)
        for key, value in parsed.items():
            try:
                pn = int(key)
                if pn in page_numbers:
                    results[pn] = EvaluationResult(**value)
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse evaluation for page %s: %s", key, e)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Batch evaluator returned invalid JSON: %s", e)

    return results
