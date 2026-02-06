"""Tests for email generator module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agent.email.generator import (
    CATEGORY_DESCRIPTIONS,
    EMAIL_QUESTIONS,
    HISTORY_TTL,
    _clear_cache,
    _condense_history_for_llm,
    _filter_history,
    _is_cache_valid,
    _is_future_date,
    _relative_time,
    build_mailto_link,
    generate_email,
    get_contacts_for_category,
    get_questions,
    strip_html,
)


class TestCategoryDefinitions:
    """Test category definitions and questions."""

    def test_category_descriptions_has_4_categories(self) -> None:
        """Verify we have exactly 4 categories."""
        assert len(CATEGORY_DESCRIPTIONS) == 4

    def test_category_descriptions_keys(self) -> None:
        """Verify category keys match expected values."""
        expected_keys = {"support", "renewals", "billing", "quotes"}
        assert set(CATEGORY_DESCRIPTIONS.keys()) == expected_keys

    def test_category_descriptions_values_are_strings(self) -> None:
        """Verify all descriptions are non-empty strings."""
        for key, value in CATEGORY_DESCRIPTIONS.items():
            assert isinstance(value, str), f"{key} description is not a string"
            assert len(value) > 10, f"{key} description is too short"

    def test_email_questions_has_4_questions(self) -> None:
        """Verify we have exactly 4 questions."""
        assert len(EMAIL_QUESTIONS) == 4

    def test_email_questions_structure(self) -> None:
        """Verify each question has id and label."""
        for q in EMAIL_QUESTIONS:
            assert "id" in q, "Question missing 'id'"
            assert "label" in q, "Question missing 'label'"
            assert q["id"] in CATEGORY_DESCRIPTIONS, f"Unknown category: {q['id']}"

    def test_email_questions_match_categories(self) -> None:
        """Verify questions cover all categories."""
        question_ids = {q["id"] for q in EMAIL_QUESTIONS}
        assert question_ids == set(CATEGORY_DESCRIPTIONS.keys())

    def test_get_questions_returns_email_questions(self) -> None:
        """Verify get_questions returns the EMAIL_QUESTIONS list."""
        assert get_questions() == EMAIL_QUESTIONS


class TestStripHtml:
    """Test HTML stripping function."""

    def test_strip_html_removes_tags(self) -> None:
        """Verify HTML tags are removed."""
        html = "<p>Hello <b>World</b></p>"
        assert strip_html(html) == "Hello World"

    def test_strip_html_handles_nbsp(self) -> None:
        """Verify &nbsp; is replaced with space."""
        html = "Hello&nbsp;World"
        assert strip_html(html) == "Hello World"

    def test_strip_html_normalizes_whitespace(self) -> None:
        """Verify multiple spaces are collapsed."""
        html = "Hello    World"
        assert strip_html(html) == "Hello World"

    def test_strip_html_handles_empty_string(self) -> None:
        """Verify empty string returns empty string."""
        assert strip_html("") == ""

    def test_strip_html_handles_none(self) -> None:
        """Verify None returns empty string."""
        assert strip_html(None) == ""  # type: ignore[arg-type]

    def test_strip_html_complex(self) -> None:
        """Verify complex HTML is handled."""
        html = "<div><p>Para 1</p>&nbsp;<p>Para 2</p></div>"
        result = strip_html(html)
        assert "Para 1" in result
        assert "Para 2" in result
        assert "<" not in result


class TestBuildMailtoLink:
    """Test mailto link building."""

    def test_build_mailto_basic(self) -> None:
        """Verify basic mailto link is built correctly."""
        link = build_mailto_link("test@example.com", "Hello", "Body text")
        assert link.startswith("mailto:test@example.com?")
        assert "subject=" in link
        assert "body=" in link

    def test_build_mailto_encodes_special_chars(self) -> None:
        """Verify special characters are encoded."""
        link = build_mailto_link("test@example.com", "Hello & Goodbye", "Line 1\nLine 2")
        # & should be encoded as %26
        assert "%26" in link or "&" not in link.split("?")[1].split("&")[0]
        # Newline should be encoded as %0A
        assert "%0A" in link

    def test_build_mailto_handles_spaces(self) -> None:
        """Verify spaces are encoded."""
        link = build_mailto_link("test@example.com", "Hello World", "Body")
        # Space can be encoded as %20 or +
        assert "%20" in link or "+" in link


class TestRelativeTime:
    """Test relative time conversion."""

    def test_relative_time_today(self) -> None:
        """Verify today returns 'today'."""
        today = time.strftime("%Y-%m-%d")
        assert _relative_time(today) == "today"

    def test_relative_time_yesterday(self) -> None:
        """Verify yesterday calculation."""
        yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
        assert _relative_time(yesterday) == "yesterday"

    def test_relative_time_days_ago(self) -> None:
        """Verify days ago calculation."""
        three_days_ago = time.strftime("%Y-%m-%d", time.localtime(time.time() - 3 * 86400))
        assert _relative_time(three_days_ago) == "3 days ago"

    def test_relative_time_weeks_ago(self) -> None:
        """Verify weeks ago calculation."""
        two_weeks_ago = time.strftime("%Y-%m-%d", time.localtime(time.time() - 14 * 86400))
        assert _relative_time(two_weeks_ago) == "2 weeks ago"

    def test_relative_time_months_ago(self) -> None:
        """Verify months ago calculation."""
        two_months_ago = time.strftime("%Y-%m-%d", time.localtime(time.time() - 60 * 86400))
        assert _relative_time(two_months_ago) == "2 months ago"

    def test_relative_time_years_ago(self) -> None:
        """Verify years ago calculation."""
        two_years_ago = time.strftime("%Y-%m-%d", time.localtime(time.time() - 730 * 86400))
        assert _relative_time(two_years_ago) == "2 years ago"

    def test_relative_time_empty(self) -> None:
        """Verify empty string returns 'unknown'."""
        assert _relative_time("") == "unknown"

    def test_relative_time_invalid(self) -> None:
        """Verify invalid date returns 'unknown'."""
        assert _relative_time("not-a-date") == "unknown"


class TestCondenseHistory:
    """Test history condensing for LLM."""

    def test_condense_history_extracts_fields(self) -> None:
        """Verify correct fields are extracted."""
        history = [
            {
                "contacts": [{"id": "123", "displayName": "John Doe"}],
                "details": "<p>Test details</p>",
                "regarding": "Test subject",
                "startTime": "2025-01-15T10:00:00Z",
            }
        ]
        result = _condense_history_for_llm(history)
        assert len(result) == 1
        assert result[0]["contactId"] == "123"
        assert result[0]["name"] == "John Doe"
        assert result[0]["details"] == "Test details"
        assert result[0]["regarding"] == "Test subject"
        assert result[0]["date"] == "2025-01-15"

    def test_condense_history_skips_no_contacts(self) -> None:
        """Verify records without contacts are skipped."""
        history = [
            {"contacts": [], "details": "Test", "regarding": "Subject"},
            {"details": "Test", "regarding": "Subject"},  # No contacts key
        ]
        result = _condense_history_for_llm(history)
        assert len(result) == 0

    def test_condense_history_handles_none_values(self) -> None:
        """Verify None values are handled."""
        history = [
            {
                "contacts": [{"id": "123"}],
                "details": None,
                "regarding": None,
                "startTime": None,
            }
        ]
        result = _condense_history_for_llm(history)
        assert len(result) == 1
        assert result[0]["details"] == ""
        assert result[0]["regarding"] == ""

    def test_condense_history_truncates_long_text(self) -> None:
        """Verify long text is truncated."""
        history = [
            {
                "contacts": [{"id": "123"}],
                "details": "x" * 1000,
                "regarding": "y" * 200,
            }
        ]
        result = _condense_history_for_llm(history)
        assert len(result[0]["details"]) <= 500
        assert len(result[0]["regarding"]) <= 100


class TestPreFiltering:
    """Test pre-filtering functions."""

    def test_is_future_date_returns_true_for_future(self) -> None:
        """Verify future dates are detected."""
        future_date = "2099-01-01T10:00:00Z"
        assert _is_future_date(future_date) is True

    def test_is_future_date_returns_false_for_past(self) -> None:
        """Verify past dates are not flagged."""
        past_date = "2020-01-01T10:00:00Z"
        assert _is_future_date(past_date) is False

    def test_is_future_date_returns_false_for_today(self) -> None:
        """Verify today is not flagged as future."""
        today = time.strftime("%Y-%m-%d")
        assert _is_future_date(today) is False

    def test_is_future_date_handles_none(self) -> None:
        """Verify None returns False."""
        assert _is_future_date(None) is False

    def test_is_future_date_handles_invalid(self) -> None:
        """Verify invalid date returns False."""
        assert _is_future_date("not-a-date") is False

    def test_filter_history_removes_future_dates(self) -> None:
        """Verify future-dated records are filtered out."""
        history = [
            {"startTime": "2020-01-01T10:00:00Z", "details": "Past"},
            {"startTime": "2099-01-01T10:00:00Z", "details": "Future"},
        ]
        result = _filter_history(history)
        assert len(result) == 1
        assert result[0]["details"] == "Past"

    def test_filter_history_removes_opportunity_lost(self) -> None:
        """Verify Opportunity Lost records are filtered out."""
        history = [
            {"startTime": "2020-01-01", "historyType": "E-mail Sent", "details": "Keep"},
            {"startTime": "2020-01-02", "historyType": "Opportunity Lost", "details": "Remove"},
        ]
        result = _filter_history(history)
        assert len(result) == 1
        assert result[0]["details"] == "Keep"

    def test_filter_history_removes_opportunity_inactive(self) -> None:
        """Verify Opportunity Inactive records are filtered out."""
        history = [
            {"startTime": "2020-01-01", "historyType": "Call Completed", "details": "Keep"},
            {"startTime": "2020-01-02", "historyType": "Opportunity Inactive", "details": "Remove"},
        ]
        result = _filter_history(history)
        assert len(result) == 1
        assert result[0]["details"] == "Keep"

    def test_filter_history_keeps_valid_records(self) -> None:
        """Verify valid records are kept."""
        history = [
            {"startTime": "2020-01-01", "historyType": "E-mail Sent", "details": "Valid"},
            {"startTime": "2020-01-02", "historyType": "Call Completed", "details": "Also Valid"},
        ]
        result = _filter_history(history)
        assert len(result) == 2

    def test_filter_history_handles_dict_history_type(self) -> None:
        """Verify dict historyType (API format) is handled correctly."""
        history = [
            {"startTime": "2020-01-01", "historyType": {"name": "E-mail Sent"}, "details": "Keep"},
            {"startTime": "2020-01-02", "historyType": {"name": "Opportunity Lost"}, "details": "Remove"},
        ]
        result = _filter_history(history)
        assert len(result) == 1
        assert result[0]["details"] == "Keep"


class TestCache:
    """Test caching functions."""

    def test_cache_invalid_when_empty(self) -> None:
        """Verify cache is invalid when empty."""
        _clear_cache()
        assert not _is_cache_valid()

    def test_clear_cache(self) -> None:
        """Verify cache is cleared."""
        _clear_cache()
        # After clearing, cache should be invalid
        assert not _is_cache_valid()


class TestGetContactsForCategory:
    """Test get_contacts_for_category function."""

    @pytest.mark.asyncio
    async def test_get_contacts_invalid_category(self) -> None:
        """Verify invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            await get_contacts_for_category("invalid_category")

    @pytest.mark.asyncio
    async def test_get_contacts_valid_category(self) -> None:
        """Verify valid category calls fetch and classify."""
        _clear_cache()

        mock_history = [
            {
                "contacts": [{"id": "123", "displayName": "John"}],
                "details": "Test support issue",
                "regarding": "Support ticket",
                "startTime": "2025-01-15T10:00:00Z",
            }
        ]

        mock_contacts = [
            {
                "contactId": "123",
                "name": "John",
                "company": "Acme",
                "reason": "Support issue reported",
                "lastContact": "2025-01-15",
            }
        ]

        with (
            patch("backend.agent.email.generator._get", return_value=mock_history),
            patch(
                "backend.agent.email.generator._classify_history_with_llm",
                new_callable=AsyncMock,
                return_value=mock_contacts,
            ),
        ):
            result = await get_contacts_for_category("support")
            assert isinstance(result, list)


class TestGenerateEmail:
    """Test generate_email function."""

    @pytest.mark.asyncio
    async def test_generate_email_no_email_address(self) -> None:
        """Verify error when contact has no email."""
        mock_contact = {"id": "123", "fullName": "John Doe", "company": "Acme"}

        with patch("backend.agent.email.generator._get", return_value=mock_contact):
            with pytest.raises(ValueError, match="no email address"):
                await generate_email("123", "support")

    @pytest.mark.asyncio
    async def test_generate_email_success(self) -> None:
        """Verify successful email generation."""
        _clear_cache()

        mock_contact = {
            "id": "123",
            "fullName": "John Doe",
            "company": "Acme Corp",
            "emailAddress": "john@acme.com",
        }

        mock_history = [
            {
                "contacts": [{"id": "123", "displayName": "John Doe"}],
                "details": "Test support issue",
                "regarding": "Support ticket",
                "startTime": "2025-01-15T10:00:00Z",
            }
        ]

        mock_llm_response = '{"subject": "Follow-up", "body": "Hi John,\\n\\nTest email."}'

        def mock_get(endpoint: str, params: dict) -> list | dict:
            if "/api/contacts/" in endpoint:
                return mock_contact
            return mock_history

        with (
            patch("backend.agent.email.generator._get", side_effect=mock_get),
            patch("backend.agent.email.generator.create_openai_chain") as mock_chain,
        ):
            mock_chain.return_value.invoke.return_value = mock_llm_response

            result = await generate_email("123", "support")

            assert "subject" in result
            assert "body" in result
            assert "mailtoLink" in result
            assert "contact" in result
            assert result["contact"]["email"] == "john@acme.com"
