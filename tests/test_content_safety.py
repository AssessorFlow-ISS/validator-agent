"""Tests for the ContentSafetyReasoner."""
from __future__ import annotations

import pytest

from validator_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from validator_agent.domain.content_safety import ContentSafetyReasoner
from validator_agent.domain.terminal_signal import ReasonCode


@pytest.fixture
def reasoner() -> ContentSafetyReasoner:
    return ContentSafetyReasoner(model_broker=StubModelBrokerAdapter())


class TestContentSafetyReasoner:
    async def test_clean_text_is_safe(self, reasoner: ContentSafetyReasoner, clean_text: str) -> None:
        result = await reasoner.check(clean_text, "chapter1.pdf")
        assert result.is_safe is True
        assert result.reason_code == ReasonCode.VALIDATION_PASSED

    async def test_harmful_keywords_detected(self, reasoner: ContentSafetyReasoner) -> None:
        text = "This document discusses extremist views and radical ideology."
        result = await reasoner.check(text, "document.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_manifesto_keyword(self, reasoner: ContentSafetyReasoner) -> None:
        text = "The manifesto outlines dangerous strategies for disruption."
        result = await reasoner.check(text, "doc.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_bomb_keyword(self, reasoner: ContentSafetyReasoner) -> None:
        text = "Instructions for constructing a bomb from household items."
        result = await reasoner.check(text, "doc.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_weapon_keyword(self, reasoner: ContentSafetyReasoner) -> None:
        text = "This guide covers weapon assembly and maintenance procedures."
        result = await reasoner.check(text, "doc.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_pii_ssn_detected(self, reasoner: ContentSafetyReasoner) -> None:
        text = "Student record: SSN: 123-45-6789, enrolled in CS101."
        result = await reasoner.check(text, "records.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.PII_DETECTED

    async def test_pii_email_detected(self, reasoner: ContentSafetyReasoner) -> None:
        text = "Contact the student at john.smith@email.com for details."
        result = await reasoner.check(text, "contacts.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.PII_DETECTED

    async def test_pii_social_security_phrase(self, reasoner: ContentSafetyReasoner) -> None:
        text = "Please provide your social security number for verification."
        result = await reasoner.check(text, "form.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.PII_DETECTED

    async def test_copyright_isbn_detected(self, reasoner: ContentSafetyReasoner) -> None:
        text = "Reference: ISBN: 978-0-13-468599-1, published by Pearson."
        result = await reasoner.check(text, "textbook.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.COPYRIGHT_VIOLATION

    async def test_copyright_all_rights_reserved(self, reasoner: ContentSafetyReasoner) -> None:
        text = "This material is protected. All rights reserved by the publisher."
        result = await reasoner.check(text, "book.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.COPYRIGHT_VIOLATION

    async def test_empty_text_is_safe(self, reasoner: ContentSafetyReasoner) -> None:
        result = await reasoner.check("", "empty.pdf")
        assert result.is_safe is True
        assert result.reason_code == ReasonCode.VALIDATION_PASSED

    async def test_whitespace_only_is_safe(self, reasoner: ContentSafetyReasoner) -> None:
        result = await reasoner.check("   \n\t  ", "blank.pdf")
        assert result.is_safe is True
        assert result.reason_code == ReasonCode.VALIDATION_PASSED

    async def test_harmful_takes_priority_over_pii(self, reasoner: ContentSafetyReasoner) -> None:
        """When text has both harmful content and PII, harmful should win (higher priority)."""
        text = "The extremist sent details to john.doe@email.com about the plot."
        result = await reasoner.check(text, "mixed.pdf")
        assert result.is_safe is False
        assert result.reason_code == ReasonCode.HARMFUL_CONTENT

    async def test_prompt_version_format(self, reasoner: ContentSafetyReasoner) -> None:
        assert reasoner.prompt_version == "validator/content_safety@v1"
