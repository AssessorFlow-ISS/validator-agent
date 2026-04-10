"""JSON Schemas for structured LLM output in the Validator Agent.

Passed to the Model Broker via response_schema so Gemini/OpenAI enforce
the output structure at the model level (response_mime_type / json_schema).

Schemas match the exact keys parsed by the domain code:
  - CONTENT_SAFETY_SCHEMA: content_safety.py _parse_response()
    reads is_safe, reason_code, message
"""

CONTENT_SAFETY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_safe": {
            "type": "boolean",
            "description": "Whether the document content is safe for the platform",
        },
        "reason_code": {
            "type": "string",
            "enum": [
                "VALIDATION_PASSED",
                "HARMFUL_CONTENT",
                "PII_DETECTED",
                "COPYRIGHT_VIOLATION",
                "CONTENT_POLICY_VIOLATION",
            ],
            "description": "Structured reason code for the safety assessment",
        },
        "message": {
            "type": "string",
            "description": "Human-readable explanation of the safety assessment",
        },
    },
    "required": ["is_safe", "reason_code", "message"],
}
