"""Gemini JSON schema compatibility utilities.

Pydantic v2's model_json_schema() generates standard JSON Schema with
features that the Gemini API rejects in structured-output response schemas.
This module cleans a Pydantic-generated schema so it works with both
OpenAI and Gemini providers.

Inlined from af_shared.utils.schema_compat — no external dependency.
"""

from __future__ import annotations

import copy

_GEMINI_UNSUPPORTED_KEYS: frozenset[str] = frozenset(
    {"title", "$defs", "default", "additionalProperties", "minItems", "maxItems"}
)


def clean_for_gemini(schema: dict) -> dict:
    """Clean a Pydantic JSON schema for Gemini compatibility.

    Returns a new dict (the input is not mutated).
    """
    cleaned = _resolve_refs(copy.deepcopy(schema))
    cleaned = _strip_unsupported_keys(cleaned)
    return cleaned


def _resolve_refs(schema: dict) -> dict:
    """Inline $ref references and remove $defs."""
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def _resolve(node: dict) -> dict:
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            if ref_name in defs:
                resolved = copy.deepcopy(defs[ref_name])
                resolved.pop("title", None)
                return _resolve(resolved)
            return node
        result: dict = {}
        for key, value in node.items():
            if isinstance(value, dict):
                result[key] = _resolve(value)
            elif isinstance(value, list):
                result[key] = [_resolve(item) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result

    return _resolve(schema)


def _strip_unsupported_keys(schema: dict) -> dict:
    """Recursively remove keys Gemini does not support."""

    def _clean(node: dict) -> dict:
        for key in list(node.keys()):
            if key in _GEMINI_UNSUPPORTED_KEYS:
                del node[key]
            elif isinstance(node[key], dict):
                node[key] = _clean(node[key])
            elif isinstance(node[key], list):
                node[key] = [_clean(item) if isinstance(item, dict) else item for item in node[key]]

        if "anyOf" in node and isinstance(node["anyOf"], list):
            non_null = [t for t in node["anyOf"] if not (isinstance(t, dict) and t.get("type") == "null")]
            if len(non_null) == 1:
                node.pop("anyOf")
                node.update(non_null[0])

        return node

    return _clean(schema)
