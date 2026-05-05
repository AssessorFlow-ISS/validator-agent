"""Tests for the Terminal Signal model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from validator_agent.domain.terminal_signal import (
    ReasonCode,
    TerminalSignal,
    TerminalSignalStatus,
)


class TestTerminalSignalStatus:
    def test_proceed_value(self) -> None:
        assert TerminalSignalStatus.PROCEED == "PROCEED"

    def test_terminate_value(self) -> None:
        assert TerminalSignalStatus.TERMINATE == "TERMINATE"

    def test_three_statuses(self) -> None:
        assert len(TerminalSignalStatus) == 3


class TestReasonCode:
    def test_all_reason_codes_exist(self) -> None:
        expected = {
            "VALIDATION_PASSED",
            "BLURRY_UNREADABLE",
            "OCR_FAILED",
            "HARMFUL_IMAGE",
            "CONTENT_POLICY_VIOLATION",
            "HARMFUL_CONTENT",
            "RELIGIOUS_POLITICAL_VIOLATION",
            "CONTENT_SAFETY_UNAVAILABLE",
            "PII_DETECTED",
            "COPYRIGHT_VIOLATION",
        }
        actual = {code.value for code in ReasonCode}
        assert actual == expected

    def test_ten_reason_codes(self) -> None:
        assert len(ReasonCode) == 10


class TestTerminalSignal:
    def test_proceed_signal(self) -> None:
        signal = TerminalSignal(
            status=TerminalSignalStatus.PROCEED,
            reason_code=ReasonCode.VALIDATION_PASSED,
            message="All checks passed",
        )
        assert signal.status == TerminalSignalStatus.PROCEED
        assert signal.reason_code == ReasonCode.VALIDATION_PASSED
        assert signal.message == "All checks passed"

    def test_terminate_signal(self) -> None:
        signal = TerminalSignal(
            status=TerminalSignalStatus.TERMINATE,
            reason_code=ReasonCode.HARMFUL_CONTENT,
            message="Harmful content detected",
        )
        assert signal.status == TerminalSignalStatus.TERMINATE
        assert signal.reason_code == ReasonCode.HARMFUL_CONTENT

    def test_message_required(self) -> None:
        with pytest.raises(ValidationError):
            TerminalSignal(
                status=TerminalSignalStatus.PROCEED,
                reason_code=ReasonCode.VALIDATION_PASSED,
                message=None,
            )

    def test_message_max_length(self) -> None:
        with pytest.raises(ValidationError):
            TerminalSignal(
                status=TerminalSignalStatus.PROCEED,
                reason_code=ReasonCode.VALIDATION_PASSED,
                message="x" * 501,
            )

    def test_message_at_max_length(self) -> None:
        signal = TerminalSignal(
            status=TerminalSignalStatus.PROCEED,
            reason_code=ReasonCode.VALIDATION_PASSED,
            message="x" * 500,
        )
        assert len(signal.message) == 500

    def test_serialization_roundtrip(self) -> None:
        signal = TerminalSignal(
            status=TerminalSignalStatus.TERMINATE,
            reason_code=ReasonCode.PII_DETECTED,
            message="PII found in document",
        )
        data = signal.model_dump()
        restored = TerminalSignal.model_validate(data)
        assert restored == signal

    def test_json_serialization(self) -> None:
        signal = TerminalSignal(
            status=TerminalSignalStatus.PROCEED,
            reason_code=ReasonCode.VALIDATION_PASSED,
            message="Clean",
        )
        json_str = signal.model_dump_json()
        assert '"PROCEED"' in json_str
        assert '"VALIDATION_PASSED"' in json_str
