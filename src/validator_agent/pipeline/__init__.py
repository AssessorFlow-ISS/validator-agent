"""Validator Agent pipeline — 3-component sequential gate.

Component 1: MRC (Material Readiness Checker) — Vertex AI EfficientNet-B0
Component 2: OCR Pipeline — Document AI + page classification + visual understanding
Component 3: Content Safety — moderation pre-filter + 4 parallel analyzers + synthesizer

Source: https://github.com/assessorflow/validator-agent-test
Integrated: 2026-04-05
"""
from validator_agent.pipeline.models import ValidatorResult
from validator_agent.pipeline.validator_pipeline import validate_file

__all__ = ["validate_file", "ValidatorResult"]
