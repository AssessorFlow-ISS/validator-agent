"""LLM client — single funnel for ALL LLM calls in the Validator Agent.

Every LLM call (text, multimodal, structured) flows through this module.
This gives us:
  - One place for token tracking and cost estimation
  - One place for Langfuse tracing (every LLM call appears in the trace)
  - One place for guardrails (input/output scanning on every LLM call)
  - Per-agent model registry (no Model Broker proxy)

Three entry points:
  generate()            — text prompts (analyzers, synthesizer, classifier)
  generate_multimodal() — messages with images (visual understanding)
  generate_structured() — Pydantic-validated output (wraps generate())

Guardrails (wraps every LLM call):
  PRE-LLM:  GuardrailScanner.scan_input() — 11 injection patterns + evasion normalization + PII
            GuardrailScanner.prepare_system_prompt() — canary token injection
  POST-LLM: GuardrailScanner.scan_output() — canary leak detection
  Config:   GUARDRAILS_ENABLED, GUARDRAILS_CANARY_ENABLED (env vars)

Langfuse tracing:
  Uses Langfuse SDK directly (sync, fire-and-forget). The SDK batches
  and flushes in background threads — never blocks the pipeline.
  Controlled by TRACING_ADAPTER env var:
    stub    → no Langfuse calls (default, for tests)
    langfuse → real tracing

Architecture: pipeline functions are sync (run in asyncio.to_thread).
The async TracingPort is for the service layer. This module uses
Langfuse SDK directly because it needs to trace from sync code.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import time
from dataclasses import dataclass

from validator_agent.pipeline.schema_compat import clean_for_gemini

from validator_agent.pipeline.model_registry import (
    estimate_cost,
    get_client,
    get_model_id,
    get_tier,
)

logger = logging.getLogger(__name__)

# ─── Guardrails (rule-based, zero LLM cost) ──────────────────

_guardrail_scanner = None
_guardrail_initialized = False


def _get_scanner():
    """Lazy-init GuardrailScanner from env vars. Returns None if disabled."""
    global _guardrail_scanner, _guardrail_initialized
    if _guardrail_initialized:
        return _guardrail_scanner

    _guardrail_initialized = True
    if os.getenv("GUARDRAILS_ENABLED", "true").lower() not in ("true", "1", "yes"):
        logger.info("guardrails_disabled")
        return None

    from validator_agent.guardrails import GuardrailScanner
    _guardrail_scanner = GuardrailScanner(
        enabled=True,
        block_on_pii=os.getenv("GUARDRAILS_BLOCK_ON_PII", "false").lower() in ("true", "1"),
        block_on_injection=os.getenv("GUARDRAILS_BLOCK_ON_INJECTION", "false").lower() in ("true", "1"),
        canary_enabled=os.getenv("GUARDRAILS_CANARY_ENABLED", "true").lower() in ("true", "1", "yes"),
    )
    logger.info(
        "guardrails_initialized",
        block_on_pii=_guardrail_scanner._block_on_pii,
        block_on_injection=_guardrail_scanner._block_on_injection,
        canary_enabled=_guardrail_scanner._canary_enabled,
    )
    return _guardrail_scanner


# ─── Workflow Context ─────────────────────────────────────────
# Using module-level variable (not contextvars) because ThreadPoolExecutor
# does NOT propagate contextvars to worker threads in Python 3.12.
# The content analyzers run in parallel threads and need access to workflow_id.

_current_workflow_id: str = "unknown"


def set_workflow_context(workflow_id: str) -> None:
    """Set the workflow_id for the current process. Thread-safe for reads."""
    global _current_workflow_id
    _current_workflow_id = workflow_id


# Keep ContextVar for backward compat but also set module var
_workflow_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "workflow_id", default="unknown"
)


# ─── Response & Stats ─────────────────────────────────────────

@dataclass
class LlmResponse:
    """Unified LLM response."""

    content: str
    model_used: str = "unknown"
    model_tier: str = "HIGH"
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class LlmClientStats:
    """Accumulated token/cost stats for telemetry."""

    request_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    last_model_used: str = "unknown"


_stats = LlmClientStats()


def get_stats() -> LlmClientStats:
    """Return accumulated LLM call stats for this process."""
    return _stats


# ─── Langfuse (direct SDK, sync, fire-and-forget) ────────────

_langfuse_client = None
_langfuse_initialized = False
_parent_trace_cache: dict[str, object] = {}  # workflow_id -> parent trace


def _get_langfuse():
    """Lazy-init Langfuse client. Returns None if not configured."""
    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True
    if os.getenv("TRACING_ADAPTER", "stub") != "langfuse":
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse()
        logger.info("langfuse_initialized_in_llm_client")
    except ImportError:
        logger.warning("langfuse_not_installed")
    except Exception:
        logger.warning("langfuse_init_failed", exc_info=True)

    return _langfuse_client


# Component-to-stage mapping for hierarchical Langfuse traces
_COMPONENT_TO_STAGE = {
    "page_classifier": "Component 2: OCR + Classification",
    "page_classification": "Component 2: OCR + Classification",
    "visual_generator": "Component 2: Visual Understanding",
    "visual_evaluator": "Component 2: Visual Understanding",
    "analyzer_a": "Component 3: Content Safety",
    "analyzer_b": "Component 3: Content Safety",
    "analyzer_c": "Component 3: Content Safety",
    "analyzer_d": "Component 3: Content Safety",
    "content_synthesizer": "Component 3: Content Safety",
}

_stage_span_cache: dict[str, object] = {}  # stage_name -> span


def _get_parent_trace():
    """Get or create a parent trace for the current workflow. All spans nest under this."""
    lf = _get_langfuse()
    if lf is None:
        return None

    workflow_id = _current_workflow_id
    if workflow_id in _parent_trace_cache:
        return _parent_trace_cache[workflow_id]

    try:
        trace_id = hashlib.md5(workflow_id.encode()).hexdigest()
        # Create a named trace first so Langfuse shows a readable name
        lf.start_observation(
            trace_context={"trace_id": trace_id, "name": workflow_id},
            name=workflow_id,
            as_type="event",
            metadata={"agent": "validator-agent"},
        ).end()
        # Then create the parent span under that trace
        parent = lf.start_observation(
            trace_context={"trace_id": trace_id},
            name="validator-agent-pipeline",
            as_type="span",
            metadata={
                "workflow_id": workflow_id,
                "agent": "validator-agent",
                "pipeline": "content-validation",
            },
        )
        _parent_trace_cache[workflow_id] = parent
        return parent
    except Exception:
        return None


def _get_stage_span(component: str):
    """Get or create an intermediate stage span for hierarchical grouping.

    Maps component names to stage spans:
      page_classifier    -> Component 2: OCR + Classification
      visual_generator   -> Component 2: Visual Understanding
      analyzer_a/b/c/d   -> Component 3: Content Safety
      content_synthesizer -> Component 3: Content Safety

    Components not in the mapping (e.g., MRC, OCR tools) go directly under parent.
    """
    stage_name = _COMPONENT_TO_STAGE.get(component)
    if stage_name is None:
        return _get_parent_trace()

    cache_key = f"{_current_workflow_id}:{stage_name}"
    if cache_key in _stage_span_cache:
        return _stage_span_cache[cache_key]

    parent = _get_parent_trace()
    if parent is None:
        return None

    try:
        stage = parent.start_observation(
            name=stage_name,
            as_type="span",
            metadata={"stage": stage_name, "workflow_id": _current_workflow_id},
        )
        _stage_span_cache[cache_key] = stage
        return stage
    except Exception:
        return parent


def flush_trace() -> None:
    """End all stage spans + parent trace and flush to Langfuse. Call after pipeline completes."""
    workflow_id = _current_workflow_id

    # End stage spans first (children before parent)
    keys_to_remove = [k for k in _stage_span_cache if k.startswith(f"{workflow_id}:")]
    for key in keys_to_remove:
        stage = _stage_span_cache.pop(key, None)
        if stage is not None:
            try:
                stage.end()
            except Exception:
                pass

    # End parent trace
    parent = _parent_trace_cache.pop(workflow_id, None)
    if parent is not None:
        try:
            parent.end()
        except Exception:
            pass

    # Flush to Langfuse
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass


def _trace_llm_call(
    component: str,
    model_used: str,
    model_tier: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    latency_ms: float,
    prompt_version: str,
    is_multimodal: bool = False,
) -> None:
    """Emit a Langfuse generation span nested under the correct stage span. Fire-and-forget."""
    try:
        stage = _get_stage_span(component)
        if stage is None:
            return

        gen = stage.start_observation(
            name=f"validator-agent/{component}",
            as_type="generation",
            model=model_used,
            usage_details={"input": tokens_in, "output": tokens_out},
            cost_details={"total": cost},
            metadata={
                "component": component,
                "model_tier": model_tier,
                "latency_ms": round(latency_ms, 1),
                "prompt_version": prompt_version,
                "workflow_id": _current_workflow_id,
                "multimodal": is_multimodal,
            },
        )
        gen.end()
    except Exception:
        logger.warning("langfuse_trace_failed", component=component, exc_info=True)


# ─── generate() — text prompts ───────────────────────────────

def generate(
    *,
    task_key: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    response_format: str | None = None,
    response_schema: dict | None = None,
    prompt_version: str = "validator/pipeline@v1",
    session_id: str | None = None,
) -> LlmResponse:
    """Generate LLM response from a text prompt."""
    component = task_key.split(".")[-1] if "." in task_key else task_key

    client = get_client(component)
    model_id = get_model_id(component)
    tier = get_tier(component)

    # ── PRE-LLM GUARDRAILS ──
    scanner = _get_scanner()
    canary: str | None = None

    if scanner is not None:
        input_result = scanner.scan_input(prompt)
        _trace_guardrail("input", component, input_result)
        if not input_result.passed:
            logger.warning("guardrail_input_blocked component=%s violations=%s",
                           component, [v.pattern_name for v in input_result.violations])
            return LlmResponse(content="{}", model_used=model_id)

        if system_prompt:
            system_prompt, canary = scanner.prepare_system_prompt(system_prompt)

    # ── BUILD MESSAGES ──
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # ── LLM CALL ──
    resp = _call_openai(client, model_id, messages, max_tokens, temperature, response_format)
    cost = estimate_cost(resp.model_used, resp.tokens_input, resp.tokens_output)
    resp.cost_usd = cost
    resp.model_tier = str(tier)

    # ── POST-LLM GUARDRAILS ──
    if scanner is not None:
        output_result = scanner.scan_output(resp.content, canary=canary)
        _trace_guardrail("output", component, output_result)
        if not output_result.passed:
            logger.warning("guardrail_output_blocked component=%s violations=%s",
                           component, [v.pattern_name for v in output_result.violations])
            return LlmResponse(content="{}", model_used=resp.model_used,
                               tokens_input=resp.tokens_input, tokens_output=resp.tokens_output,
                               cost_usd=cost, latency_ms=resp.latency_ms)

    _update_stats(resp)
    _log_call(component, resp)
    _trace_llm_call(component, resp.model_used, str(tier), resp.tokens_input, resp.tokens_output, cost, resp.latency_ms, prompt_version)

    return resp


# ─── generate_multimodal() — messages with images ────────────

def generate_multimodal(
    *,
    task_key: str,
    messages: list[dict],
    max_tokens: int = 2000,
    temperature: float = 0.2,
    response_format: str | None = None,
    response_schema: dict | None = None,
    prompt_version: str = "validator/pipeline@v1",
) -> LlmResponse:
    """Generate LLM response from multimodal messages (text + images)."""
    component = task_key.split(".")[-1] if "." in task_key else task_key

    client = get_client(component)
    model_id = get_model_id(component)
    tier = get_tier(component)

    # ── PRE-LLM GUARDRAILS (text parts only) ──
    scanner = _get_scanner()
    canary: str | None = None

    if scanner is not None:
        text_parts = _extract_text_from_messages(messages)
        if text_parts:
            input_result = scanner.scan_input(text_parts)
            _trace_guardrail("input", component, input_result)
            if not input_result.passed:
                logger.warning("guardrail_input_blocked", component=component,
                               violations=[v.pattern_name for v in input_result.violations])
                return LlmResponse(content="{}", model_used=model_id)

        # Inject canary into system message if present
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"], canary = scanner.prepare_system_prompt(msg["content"])
                break

    # ── LLM CALL ──
    resp = _call_openai(client, model_id, messages, max_tokens, temperature, response_format)
    cost = estimate_cost(resp.model_used, resp.tokens_input, resp.tokens_output)
    resp.cost_usd = cost
    resp.model_tier = str(tier)

    # ── POST-LLM GUARDRAILS ──
    if scanner is not None:
        output_result = scanner.scan_output(resp.content, canary=canary)
        _trace_guardrail("output", component, output_result)
        if not output_result.passed:
            logger.warning("guardrail_output_blocked component=%s violations=%s",
                           component, [v.pattern_name for v in output_result.violations])
            return LlmResponse(content="{}", model_used=resp.model_used,
                               tokens_input=resp.tokens_input, tokens_output=resp.tokens_output,
                               cost_usd=cost, latency_ms=resp.latency_ms)

    _update_stats(resp)
    _log_call(component, resp, multimodal=True)
    _trace_llm_call(component, resp.model_used, str(tier), resp.tokens_input, resp.tokens_output, cost, resp.latency_ms, prompt_version, is_multimodal=True)

    return resp


# ─── generate_structured() — Pydantic output ─────────────────

def generate_structured(
    *,
    task_key: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    prompt_version: str = "validator/pipeline@v1",
) -> tuple[object, LlmResponse]:
    """Generate structured output conforming to a Pydantic model.

    Returns (parsed_model_instance, raw_llm_response).
    """
    schema = clean_for_gemini(response_model.model_json_schema())

    resp = generate(
        task_key=task_key,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format="json",
        response_schema=schema,
        prompt_version=prompt_version,
    )

    parsed = response_model.model_validate_json(resp.content)
    return parsed, resp


# ─── Tool tracing (non-LLM tools: MRC, OCR, Moderation) ──────

def trace_tool(
    tool_name: str,
    input_params: dict,
    output_summary: dict,
    latency_ms: float,
) -> None:
    """Emit a Langfuse tool span nested under the correct stage. Fire-and-forget.

    Use this for MRC, Document AI, OpenAI Moderation, guardrail scans.
    Guardrail scans nest under the stage of the component they protect.
    Other tools (MRC, OCR, moderation) go under the parent pipeline span.
    """
    try:
        # Guardrail scans: nest under the stage of the component
        component = input_params.get("component", "")
        if "guardrail" in tool_name and component:
            span = _get_stage_span(component)
        elif tool_name == "openai-moderation-prefilter":
            span = _get_stage_span("analyzer_a")  # moderation is part of content safety
        else:
            span = _get_parent_trace()

        if span is None:
            return

        tool = span.start_observation(
            name=f"tool/{tool_name}",
            as_type="tool",
            input=input_params,
            output=output_summary,
            metadata={
                "agent_name": "validator-agent",
                "workflow_id": _current_workflow_id,
                "latency_ms": round(latency_ms, 1),
            },
        )
        tool.end()
    except Exception:
        logger.warning("langfuse_tool_trace_failed tool_name=%s", tool_name, exc_info=True)


# ─── Guardrail helpers ────────────────────────────────────────

def _trace_guardrail(direction: str, component: str, result) -> None:
    """Log guardrail scan result to structlog + Langfuse. Fire-and-forget.

    Only traces WARN and BLOCK to Langfuse (not PASS) to keep the trace
    diagram clean. PASS scans are the norm — only violations are interesting.
    """
    if not result.violations:
        return  # PASS — no trace needed, keeps Langfuse diagram clean

    action = "BLOCK" if not result.passed else "WARN"
    violation_names = [v.pattern_name for v in result.violations]

    logger.warning(
        "guardrail_%s_%s component=%s violations=%s latency_ms=%.2f",
        direction, action.lower(), component, violation_names, result.latency_ms,
    )

    trace_tool(
        tool_name=f"guardrail-{direction}-scan",
        input_params={"component": component},
        output_summary={
            "action": action,
            "passed": result.passed,
            "violations": violation_names,
        },
        latency_ms=result.latency_ms,
    )


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract text content from multimodal messages for guardrail scanning."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
    return "\n".join(parts)


# ─── Internal helpers ─────────────────────────────────────────

def _call_openai(
    client, model_id: str, messages: list[dict],
    max_tokens: int, temperature: float, response_format: str | None,
) -> LlmResponse:
    """Execute the OpenAI SDK call and return unified LlmResponse."""
    kwargs: dict = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    start_ms = time.time() * 1000

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        logger.warning("llm_call_failed", exc_info=True)
        return LlmResponse(content="{}", model_used=model_id, latency_ms=time.time() * 1000 - start_ms)

    elapsed_ms = time.time() * 1000 - start_ms
    choice = response.choices[0]
    usage = response.usage

    return LlmResponse(
        content=choice.message.content or "{}",
        model_used=response.model,
        tokens_input=usage.prompt_tokens if usage else 0,
        tokens_output=usage.completion_tokens if usage else 0,
        latency_ms=elapsed_ms,
    )


def _update_stats(resp: LlmResponse) -> None:
    _stats.request_count += 1
    _stats.total_tokens += resp.tokens_input + resp.tokens_output
    _stats.total_cost_usd += resp.cost_usd
    _stats.last_model_used = resp.model_used


def _log_call(component: str, resp: LlmResponse, multimodal: bool = False) -> None:
    logger.info(
        "llm_call_complete",
        component=component,
        model=resp.model_used,
        tier=resp.model_tier,
        tokens_in=resp.tokens_input,
        tokens_out=resp.tokens_output,
        cost_usd=resp.cost_usd,
        latency_ms=round(resp.latency_ms, 1),
        multimodal=multimodal,
    )