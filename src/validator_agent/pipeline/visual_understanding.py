"""Pass 3: Visual understanding — multimodal generator + multi-dimension evaluator.

For pages classified as VISUAL, sends the full page image + OCR text to a
vision-capable LLM to produce a complete, context-aware rewrite that
integrates text and visual descriptions.

Uses Evaluator-Optimizer pattern with structured per-dimension feedback.

Also includes image moderation via OpenAI Moderation API — catches harmful/sexual/violent
imagery BEFORE spending on the generator. If flagged, the page is marked as harmful
and the entire file terminates (no need to proceed to Component 3).

LLM calls use per-agent model registry (model_registry.py) for direct
provider access. No Model Broker proxy — agents manage their own connections.

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

from pydantic import BaseModel, Field

from validator_agent.pipeline.llm_client import generate_multimodal
from validator_agent.pipeline.model_registry import get_moderation_client
from validator_agent.pipeline.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4.1-mini")

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
        default=None, description="Concise fix instructions for the generator if FAIL"
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


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract text-only content from OpenAI-format messages for the prompt field."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n".join(parts)[:4000]  # Truncate for guardrail scanning


def _call_llm(
    messages: list[dict],
    component: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.2,
    response_format: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """Call LLM via llm_client.generate_multimodal() — single funnel for all LLM calls.

    Routes through llm_client so token tracking, cost estimation, and
    Langfuse tracing are handled in one place for both text and multimodal calls.
    """
    resp = generate_multimodal(
        task_key=f"validator.{component}",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        response_schema=response_schema,
        prompt_version=f"validator/{component}@v1",
    )
    return resp.content


def check_image_moderation(page_image_bytes: bytes) -> ImageModerationResult:
    """Run OpenAI Moderation API on a page image directly.

    Catches harmful, sexual, violent, or hateful imagery before
    spending on the expensive generator. Moderation API is free.
    """
    image_b64 = _encode_image_b64(page_image_bytes)

    multimodal_input = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
            },
        },
    ]

    try:
        client = get_moderation_client()
        response = client.moderations.create(
            input=multimodal_input,
            model="omni-moderation-latest",
        )

        result = response.results[0]
        flagged_categories = [cat for cat, flagged in result.categories.model_dump().items() if flagged]

        return ImageModerationResult(
            flagged=result.flagged,
            categories=flagged_categories,
            detail=f"Image moderation flagged: {', '.join(flagged_categories)}" if result.flagged else None,
        )
    except Exception as e:
        logger.warning("Image moderation failed: %s", e)
        return ImageModerationResult(error=f"Image moderation error: {e}")


def generate_visual_description(
    page_image_bytes: bytes,
    ocr_text: str,
    previous_output: str | None = None,
    retry_feedback: str | None = None,
    workflow_id: str | None = None,
) -> str:
    """Generate a description of a visual page via Model Broker.

    Args:
        page_image_bytes: PNG bytes of the page image.
        ocr_text: OCR-extracted text from this page.
        previous_output: If retrying, the previous generator output.
        retry_feedback: If retrying, the evaluator's structured feedback.
        workflow_id: Session ID for budget tracking.

    Returns:
        Generated text description of the page.
    """
    image_b64 = _encode_image_b64(page_image_bytes)

    if previous_output and retry_feedback:
        user_content = GENERATOR_RETRY_PROMPT.format(evaluation_feedback=retry_feedback)
        user_content += f"\n\nYour previous output:\n{previous_output}"
    else:
        user_content = "Please analyze this page and produce the rewrite as instructed."

    system_prompt = GENERATOR_SYSTEM_PROMPT.format(ocr_text=ocr_text[:3000])

    messages = [
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
    ]

    return _call_llm(
        messages,
        component="visual_generator",
        max_tokens=2000,
        temperature=0.2,
    ).strip()


def evaluate_description(
    page_image_bytes: bytes,
    generated_description: str,
    workflow_id: str | None = None,
) -> EvaluationResult:
    """Evaluate a generated description against the original page image.

    Args:
        page_image_bytes: PNG bytes of the page image.
        generated_description: The generator's output to evaluate.
        workflow_id: Session ID for budget tracking.

    Returns:
        Structured EvaluationResult with per-dimension verdicts.
    """
    image_b64 = _encode_image_b64(page_image_bytes)

    # Build the JSON schema from EvaluationResult for structured output.
    # Use the shared cleaner so $ref / $defs / anyOf-null / etc. are resolved
    # recursively. Pre-popping just $defs at top level orphans nested $ref.
    from validator_agent.pipeline.schema_compat import clean_for_gemini

    eval_schema = clean_for_gemini(EvaluationResult.model_json_schema())

    messages = [
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
    ]

    raw = _call_llm(
        messages,
        component="visual_evaluator",
        max_tokens=1500,
        temperature=0,
        response_format="json",
        response_schema=eval_schema,
    )

    return EvaluationResult.model_validate_json(raw)


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
    workflow_id: str | None = None,
) -> VisualProcessResult:
    """Process a visual page through the Generator + Evaluator-Optimizer loop."""
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
        generated = generate_visual_description(
            page_image_bytes,
            ocr_text,
            workflow_id=workflow_id,
        )
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
            evaluation = evaluate_description(
                page_image_bytes,
                generated,
                workflow_id=workflow_id,
            )
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
                page_image_bytes,
                ocr_text,
                previous_output=generated,
                retry_feedback=feedback,
                workflow_id=workflow_id,
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
    workflow_id: str | None = None,
) -> dict[int, VisualProcessResult]:
    """Process multiple visual pages in a single Model Broker call.

    Args:
        batch_items: List of (page_number, page_image_bytes, ocr_text) tuples.
        workflow_id: Session ID for budget tracking.

    Returns:
        Dict of {page_number: VisualProcessResult}.
    """
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

    # Phase 2: Single Model Broker call for all pages in batch
    ocr_context = "\n\n".join(f"--- Page {pn} OCR ---\n{ocr[:3000]}" for pn, _, ocr in batch_items)
    system_prompt = _BATCH_GENERATOR_PREAMBLE + GENERATOR_SYSTEM_PROMPT.format(ocr_text=ocr_context)

    user_content: list[dict] = [{"type": "text", "text": "Analyze these pages:"}]
    for pn, img_bytes, _ in batch_items:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image_b64(img_bytes)}",
                    "detail": "high",
                },
            }
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        raw_text = _call_llm(
            messages,
            component="visual_generator",
            max_tokens=min(2000 * len(batch_items), 16000),
            temperature=0.2,
        ).strip()
    except Exception as e:
        logger.warning("Batch generator failed: %s — falling back to single-page", e)
        return {
            pn: VisualProcessResult(text=ocr, source="ocr_fallback", attempts=1, error=f"Batch generator failed: {e}")
            for pn, _, ocr in batch_items
        }

    # Phase 3: Parse per-page descriptions
    descriptions = _parse_batch_descriptions(raw_text, page_numbers)

    # Phase 4: Evaluate all descriptions in one call
    eval_results = evaluate_visual_batch(batch_items, descriptions, workflow_id=workflow_id)

    # Phase 5: Build results — retry failed pages via single-page fallback
    for pn, img_bytes, ocr_text in batch_items:
        if pn not in descriptions:
            # Parser couldn't extract this page — fall back to single-page
            logger.warning("Batch parse missing page %d — falling back to single-page", pn)
            try:
                results[pn] = process_visual_page(img_bytes, ocr_text, workflow_id=workflow_id)
            except Exception as e:
                logger.warning("Single-page fallback failed for page %d: %s", pn, e)
                results[pn] = VisualProcessResult(text=ocr_text, source="ocr_fallback", error=str(e))
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
            try:
                results[pn] = process_visual_page(img_bytes, ocr_text, workflow_id=workflow_id)
            except Exception as e:
                logger.warning("Single-page fallback failed for page %d: %s", pn, e)
                results[pn] = VisualProcessResult(text=ocr_text, source="ocr_fallback", error=str(e))

    return results


def evaluate_visual_batch(
    batch_items: list[tuple[int, bytes, str]],
    descriptions: dict[int, str],
    workflow_id: str | None = None,
) -> dict[int, EvaluationResult]:
    """Evaluate multiple page descriptions in a single Model Broker call.

    Args:
        batch_items: List of (page_number, page_image_bytes, ocr_text) tuples.
        descriptions: Dict of {page_number: generated_description}.
        workflow_id: Session ID for budget tracking.

    Returns:
        Dict of {page_number: EvaluationResult}.
    """
    page_numbers = [item[0] for item in batch_items if item[0] in descriptions]
    if not page_numbers:
        return {}

    # Build evaluation prompt with all descriptions
    desc_text = "\n\n".join(f"## Page {pn}\n{descriptions[pn]}" for pn in page_numbers)

    user_content: list[dict] = [
        {"type": "text", "text": f"Descriptions to evaluate:\n\n{desc_text}"},
    ]
    for pn, img_bytes, _ in batch_items:
        if pn in descriptions:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_encode_image_b64(img_bytes)}",
                        "detail": "high",
                    },
                }
            )

    eval_system = (
        _BATCH_EVALUATOR_PREAMBLE + EVALUATOR_SYSTEM_PROMPT + "\n\n"
        "Output a single JSON object with page numbers as keys. Example:\n"
        '{"3": {"overall": "PASS", "dimensions": {"accuracy": {"verdict": "PASS", "feedback": null}, '
        '"completeness": {"verdict": "PASS", "feedback": null}, '
        '"educational_value": {"verdict": "PASS", "feedback": null}}, '
        '"retry_prompt_supplement": null}, "5": {...}}'
    )

    messages = [
        {"role": "system", "content": eval_system},
        {"role": "user", "content": user_content},
    ]

    try:
        raw_text = _call_llm(
            messages,
            component="visual_evaluator",
            max_tokens=min(1500 * len(page_numbers), 12000),
            temperature=0,
            response_format="json",
        ).strip()
    except Exception as e:
        logger.warning("Batch evaluator failed: %s", e)
        return {}

    # Parse JSON response — keyed by page number string
    # gpt-4.1-mini may return a flat format (dimensions at top level) instead
    # of the expected {overall, dimensions, retry_prompt_supplement} wrapper.
    results: dict[int, EvaluationResult] = {}
    try:
        parsed = json.loads(raw_text)
        for key, value in parsed.items():
            try:
                pn = int(key)
                if pn not in page_numbers:
                    continue
                # Try expected format first
                if "overall" in value and "dimensions" in value:
                    results[pn] = EvaluationResult(**value)
                else:
                    # Flat format: dimension dicts at top level, derive overall
                    dims = {k: v for k, v in value.items() if isinstance(v, dict) and "verdict" in v}
                    if dims:
                        all_pass = all(d.get("verdict") == "PASS" for d in dims.values())
                        results[pn] = EvaluationResult(
                            overall=Verdict.PASS if all_pass else Verdict.FAIL,
                            dimensions={k: DimensionResult(**v) for k, v in dims.items()},
                        )
                    else:
                        logger.warning("Unrecognized eval format for page %s: %s", key, list(value.keys()))
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse evaluation for page %s: %s", key, e)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Batch evaluator returned invalid JSON: %s", e)

    return results
