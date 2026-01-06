"""
Tests for agent helper modules (formatters, llm_helpers, progress).

Tests the extracted modules from orchestrator.py refactoring.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"


# =============================================================================
# Formatters Tests
# =============================================================================

class TestFormatCompanySection:
    """Tests for format_company_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_company_section
        assert format_company_section(None) == ""
    
    def test_returns_empty_for_missing_company(self):
        """Returns empty string when company key missing."""
        from backend.agent.nodes.support.formatters import format_company_section
        assert format_company_section({}) == ""
        assert format_company_section({"other": "data"}) == ""
    
    def test_formats_basic_company_info(self):
        """Formats basic company information."""
        from backend.agent.nodes.support.formatters import format_company_section
        
        data = {
            "company": {
                "name": "Acme Corp",
                "company_id": "ACME-001",
                "status": "Active",
                "plan": "Enterprise",
            }
        }
        result = format_company_section(data)
        
        assert "COMPANY INFO" in result
        assert "Acme Corp" in result
        assert "ACME-001" in result
        assert "Active" in result
        assert "Enterprise" in result
    
    def test_includes_contacts_when_present(self):
        """Includes contacts in output."""
        from backend.agent.nodes.support.formatters import format_company_section
        
        data = {
            "company": {"name": "Test", "company_id": "T-001"},
            "contacts": [
                {"first_name": "John", "last_name": "Doe", "job_title": "CEO", "email": "john@test.com"}
            ]
        }
        result = format_company_section(data)
        
        assert "Key Contacts" in result
        assert "John Doe" in result
        assert "CEO" in result
    
    def test_limits_contacts_to_three(self):
        """Only shows first 3 contacts."""
        from backend.agent.nodes.support.formatters import format_company_section
        
        contacts = [
            {"first_name": f"Contact{i}", "last_name": "Test", "job_title": "Role", "email": f"c{i}@test.com"}
            for i in range(5)
        ]
        data = {"company": {"name": "Test"}, "contacts": contacts}
        result = format_company_section(data)
        
        assert "Contact0" in result
        assert "Contact1" in result
        assert "Contact2" in result
        assert "Contact3" not in result
        assert "Contact4" not in result


class TestFormatActivitiesSection:
    """Tests for format_activities_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_activities_section
        assert format_activities_section(None) == ""
    
    def test_handles_no_activities(self):
        """Shows message when no activities."""
        from backend.agent.nodes.support.formatters import format_activities_section
        
        result = format_activities_section({"activities": []})
        assert "No recent activities" in result
    
    def test_formats_activities(self):
        """Formats activity list."""
        from backend.agent.nodes.support.formatters import format_activities_section
        
        data = {
            "activities": [
                {"type": "Call", "subject": "Check-in call", "owner": "John", 
                 "due_datetime": "2025-01-15T10:00:00", "status": "Scheduled"}
            ],
            "count": 1,
            "days": 30
        }
        result = format_activities_section(data)
        
        assert "RECENT ACTIVITIES" in result
        assert "Call" in result
        assert "Check-in call" in result
        assert "John" in result


class TestFormatHistorySection:
    """Tests for format_history_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_history_section
        assert format_history_section(None) == ""
    
    def test_handles_no_history(self):
        """Shows message when no history."""
        from backend.agent.nodes.support.formatters import format_history_section
        
        result = format_history_section({"history": []})
        assert "No recent history" in result
    
    def test_formats_history_entries(self):
        """Formats history entries."""
        from backend.agent.nodes.support.formatters import format_history_section
        
        data = {
            "history": [
                {"type": "Note", "subject": "Call notes", "owner": "Jane",
                 "occurred_at": "2025-01-10T14:00:00", "description": "Good meeting"}
            ],
            "count": 1,
            "days": 30
        }
        result = format_history_section(data)
        
        assert "HISTORY LOG" in result
        assert "Note" in result
        assert "Call notes" in result


class TestFormatPipelineSection:
    """Tests for format_pipeline_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_pipeline_section
        assert format_pipeline_section(None) == ""
    
    def test_handles_empty_pipeline(self):
        """Shows message when no opportunities."""
        from backend.agent.nodes.support.formatters import format_pipeline_section
        
        result = format_pipeline_section({"summary": {"total_count": 0}})
        assert "No open opportunities" in result
    
    def test_formats_pipeline_summary(self):
        """Formats pipeline summary."""
        from backend.agent.nodes.support.formatters import format_pipeline_section
        
        data = {
            "summary": {
                "total_count": 5,
                "total_value": 100000,
                "stages": {"Proposal": {"count": 2, "total_value": 50000}}
            },
            "opportunities": []
        }
        result = format_pipeline_section(data)
        
        assert "PIPELINE SUMMARY" in result
        assert "5" in result
        assert "100,000" in result


class TestFormatRenewalsSection:
    """Tests for format_renewals_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_renewals_section
        assert format_renewals_section(None) == ""
    
    def test_handles_no_renewals(self):
        """Shows message when no renewals."""
        from backend.agent.nodes.support.formatters import format_renewals_section
        
        result = format_renewals_section({"renewals": []})
        assert "No renewals" in result
    
    def test_formats_renewals(self):
        """Formats renewal list."""
        from backend.agent.nodes.support.formatters import format_renewals_section
        
        data = {
            "renewals": [
                {"name": "Acme Corp", "company_id": "ACME-001", 
                 "renewal_date": "2025-03-01", "plan": "Pro", "health_flags": "Good"}
            ],
            "count": 1,
            "days": 90
        }
        result = format_renewals_section(data)
        
        assert "UPCOMING RENEWALS" in result
        assert "Acme Corp" in result


class TestFormatDocsSection:
    """Tests for format_docs_section."""
    
    def test_returns_empty_for_none(self):
        """Returns empty string for None input."""
        from backend.agent.nodes.support.formatters import format_docs_section
        assert format_docs_section(None) == ""
        assert format_docs_section("") == ""
    
    def test_formats_docs_answer(self):
        """Formats documentation answer."""
        from backend.agent.nodes.support.formatters import format_docs_section
        
        result = format_docs_section("Here is how to import contacts...")
        assert "DOCUMENTATION GUIDANCE" in result
        assert "Here is how to import contacts" in result


# =============================================================================
# LLM Helpers Tests
# =============================================================================


class TestCallAnswerChainMockMode:
    """Tests for call_answer_chain in mock mode."""

    def test_returns_tuple(self):
        """Returns tuple of (response, latency)."""
        from backend.agent.llm.helpers import call_answer_chain

        result = call_answer_chain(
            question="Test prompt",
            conversation_history_section="",
            company_section="Test company",
            activities_section="No activities",
            history_section="No history",
            pipeline_section="No pipeline",
            renewals_section="No renewals",
            docs_section="No docs",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)

    def test_returns_mock_latency(self):
        """Returns mock latency of 100ms in mock mode."""
        from backend.agent.llm.helpers import call_answer_chain

        _, latency = call_answer_chain(
            question="Test",
            conversation_history_section="",
            company_section="Company",
            activities_section="Activities",
            history_section="History",
            pipeline_section="Pipeline",
            renewals_section="Renewals",
            docs_section="Docs",
        )
        assert latency == 100


# =============================================================================
# _format_available_data Tests
# =============================================================================

class TestFormatAvailableData:
    """Tests for _format_available_data helper function."""

    def test_returns_string(self):
        """Returns a string."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data(None, None)
        assert isinstance(result, str)

    def test_handles_none_data(self):
        """Handles None data gracefully."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data(None, "Acme Corp")
        assert "No specific data available" in result

    def test_handles_empty_data(self):
        """Handles empty data dict."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data({}, "Acme Corp")
        assert "No specific data available" in result

    def test_formats_contacts(self):
        """Formats contacts count."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data({"contacts": 5}, "Acme Corp")
        assert "Contacts" in result
        assert "5" in result
        assert "Acme Corp" in result

    def test_formats_activities(self):
        """Formats activities count."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data({"activities": 10}, "Test Co")
        assert "Activities" in result
        assert "10" in result

    def test_formats_opportunities(self):
        """Formats opportunities count."""
        from backend.agent.llm.helpers import _format_available_data
        result = _format_available_data({"opportunities": 3}, "Test Co")
        assert "Opportunities" in result
        assert "3" in result

    def test_formats_multiple_data_types(self):
        """Formats multiple data types."""
        from backend.agent.llm.helpers import _format_available_data
        data = {
            "contacts": 5,
            "activities": 10,
            "opportunities": 3,
            "renewals": 1,
        }
        result = _format_available_data(data, "Acme")
        assert "Contacts" in result
        assert "Activities" in result
        assert "Opportunities" in result
        assert "Renewals" in result

    def test_excludes_zero_counts(self):
        """Excludes data types with zero counts."""
        from backend.agent.llm.helpers import _format_available_data
        data = {"contacts": 0, "activities": 5}
        result = _format_available_data(data, "Acme")
        assert "Activities" in result
        # Should not mention contacts with 0 count
        lines = result.split("\n")
        assert not any("Contacts" in line and ": 0" in line for line in lines)


# =============================================================================
# generate_follow_up_suggestions Tests
# =============================================================================

class TestGenerateFollowUpSuggestionsMock:
    """Tests for generate_follow_up_suggestions in mock mode."""

    def test_returns_list(self):
        """Returns a list of suggestions."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions
        result = generate_follow_up_suggestions(
            question="What's happening with Acme?",
            mode="data",
        )
        assert isinstance(result, list)

    def test_returns_up_to_three_suggestions(self):
        """Returns up to 3 suggestions."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions
        result = generate_follow_up_suggestions(
            question="Show me renewals",
            mode="data",
        )
        assert len(result) <= 3

    def test_suggestions_are_strings(self):
        """All suggestions are strings."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions
        result = generate_follow_up_suggestions(
            question="What's in the pipeline?",
            mode="data",
        )
        for suggestion in result:
            assert isinstance(suggestion, str)

    def test_uses_company_name_when_provided(self):
        """Uses company name in context-aware suggestions."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions
        result = generate_follow_up_suggestions(
            question="Tell me about Acme",
            mode="data",
            company_id="ACME-001",
            company_name="Acme Manufacturing",
            available_data={"opportunities": 5, "contacts": 3},
        )
        # Should have some suggestions
        assert len(result) > 0


# =============================================================================
# Additional Formatter Tests (coverage improvement)
# =============================================================================


class TestFormatDateHelper:
    """Tests for _format_date helper function."""

    def test_extracts_date_from_datetime(self):
        """Extracts date portion from ISO datetime string."""
        from backend.agent.nodes.support.formatters import _format_date
        assert _format_date("2025-01-15T10:30:00") == "2025-01-15"

    def test_returns_string_as_is_if_no_t(self):
        """Returns string as-is if no T separator."""
        from backend.agent.nodes.support.formatters import _format_date
        assert _format_date("2025-01-15") == "2025-01-15"

    def test_returns_na_for_none(self):
        """Returns N/A for None input."""
        from backend.agent.nodes.support.formatters import _format_date
        assert _format_date(None) == "N/A"

    def test_returns_na_for_empty_string(self):
        """Returns N/A for empty string."""
        from backend.agent.nodes.support.formatters import _format_date
        assert _format_date("") == "N/A"

    def test_handles_non_string_input(self):
        """Handles non-string input by converting."""
        from backend.agent.nodes.support.formatters import _format_date
        result = _format_date(12345)
        assert result == "12345"


class TestTruncateHelper:
    """Tests for _truncate helper function."""

    def test_returns_short_text_unchanged(self):
        """Returns short text unchanged."""
        from backend.agent.nodes.support.formatters import _truncate
        assert _truncate("Hello") == "Hello"

    def test_truncates_long_text(self):
        """Truncates text exceeding max_len."""
        from backend.agent.nodes.support.formatters import _truncate
        long_text = "x" * 150
        result = _truncate(long_text, max_len=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import _truncate
        assert _truncate(None) == ""

    def test_returns_empty_for_empty_string(self):
        """Returns empty string for empty input."""
        from backend.agent.nodes.support.formatters import _truncate
        assert _truncate("") == ""

    def test_uses_custom_max_len(self):
        """Respects custom max_len."""
        from backend.agent.nodes.support.formatters import _truncate
        result = _truncate("Hello World", max_len=5)
        assert result == "Hello..."

    def test_handles_exact_length(self):
        """Handles text exactly at max_len."""
        from backend.agent.nodes.support.formatters import _truncate
        result = _truncate("Hello", max_len=5)
        assert result == "Hello"


class TestSectionFormatter:
    """Tests for SectionFormatter class."""

    def test_format_returns_empty_for_none_data(self):
        """Returns empty string for None data."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="TEST",
            empty_message="No data",
            item_formatter=lambda x: str(x),
        )
        assert formatter.format(None) == ""

    def test_format_shows_empty_message_for_no_items(self):
        """Shows empty message when items list is empty."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="TEST",
            empty_message="No data found",
            item_formatter=lambda x: str(x),
        )
        result = formatter.format({"items": []})
        assert "No data found" in result
        assert "TEST" in result

    def test_format_with_count_and_days(self):
        """Includes count and days in header."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="ITEMS",
            empty_message="None",
            item_formatter=lambda x: f"- {x['name']}",
        )
        result = formatter.format({
            "items": [{"name": "Item1"}],
            "count": 5,
            "days": 30,
        })
        assert "(5 found, last 30 days)" in result

    def test_format_with_count_only(self):
        """Includes only count in header when no days."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="ITEMS",
            empty_message="None",
            item_formatter=lambda x: f"- {x['name']}",
        )
        result = formatter.format({
            "items": [{"name": "Item1"}],
            "count": 3,
        })
        assert "(3)" in result
        assert "days" not in result

    def test_format_with_days_only(self):
        """Includes only days in header when no count."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="ITEMS",
            empty_message="None",
            item_formatter=lambda x: f"- {x['name']}",
        )
        result = formatter.format({
            "items": [{"name": "Item1"}],
            "days": 60,
        })
        assert "(last 60 days)" in result

    def test_format_limits_items(self):
        """Respects max_items limit."""
        from backend.agent.nodes.support.formatters import SectionFormatter
        formatter = SectionFormatter(
            header="ITEMS",
            empty_message="None",
            item_formatter=lambda x: f"- {x['name']}",
            max_items=2,
        )
        items = [{"name": f"Item{i}"} for i in range(5)]
        result = formatter.format({"items": items})
        assert "Item0" in result
        assert "Item1" in result
        assert "Item2" not in result


class TestFormatActivityItem:
    """Tests for _format_activity item formatter."""

    def test_formats_complete_activity(self):
        """Formats activity with all fields."""
        from backend.agent.nodes.support.formatters import _format_activity
        activity = {
            "type": "Call",
            "subject": "Check-in",
            "owner": "John",
            "due_datetime": "2025-01-15T10:00:00",
            "status": "Scheduled",
        }
        result = _format_activity(activity)
        assert "[Call]" in result
        assert "Check-in" in result
        assert "John" in result
        assert "2025-01-15" in result
        assert "Scheduled" in result

    def test_uses_created_at_fallback(self):
        """Uses created_at when due_datetime missing."""
        from backend.agent.nodes.support.formatters import _format_activity
        activity = {
            "type": "Task",
            "subject": "Follow up",
            "owner": "Jane",
            "created_at": "2025-01-10T08:00:00",
            "status": "Open",
        }
        result = _format_activity(activity)
        assert "2025-01-10" in result

    def test_handles_missing_fields(self):
        """Handles missing fields with N/A."""
        from backend.agent.nodes.support.formatters import _format_activity
        result = _format_activity({})
        assert "N/A" in result


class TestFormatHistoryItem:
    """Tests for _format_history item formatter."""

    def test_formats_complete_history(self):
        """Formats history entry with all fields."""
        from backend.agent.nodes.support.formatters import _format_history
        entry = {
            "type": "Note",
            "subject": "Meeting summary",
            "owner": "Alice",
            "occurred_at": "2025-01-12T14:00:00",
            "description": "Great discussion about renewal",
        }
        result = _format_history(entry)
        assert "[Note]" in result
        assert "Meeting summary" in result
        assert "Alice" in result
        assert "2025-01-12" in result
        assert "Great discussion" in result

    def test_skips_empty_description(self):
        """Doesn't add Note line when no description."""
        from backend.agent.nodes.support.formatters import _format_history
        entry = {
            "type": "Email",
            "subject": "Sent proposal",
            "owner": "Bob",
            "occurred_at": "2025-01-11T09:00:00",
        }
        result = _format_history(entry)
        assert "Note:" not in result


class TestFormatOpportunityItem:
    """Tests for _format_opportunity item formatter."""

    def test_formats_complete_opportunity(self):
        """Formats opportunity with all fields."""
        from backend.agent.nodes.support.formatters import _format_opportunity
        opp = {
            "name": "Enterprise Deal",
            "stage": "Proposal",
            "value": 50000,
            "expected_close_date": "2025-03-15",
        }
        result = _format_opportunity(opp)
        assert "Enterprise Deal" in result
        assert "Proposal" in result
        assert "$50,000" in result
        assert "2025-03-15" in result

    def test_handles_zero_value(self):
        """Handles zero value opportunity."""
        from backend.agent.nodes.support.formatters import _format_opportunity
        opp = {"name": "Free Pilot", "stage": "Discovery", "value": 0}
        result = _format_opportunity(opp)
        assert "$0" in result


class TestFormatRenewalItem:
    """Tests for _format_renewal item formatter."""

    def test_formats_complete_renewal(self):
        """Formats renewal with all fields."""
        from backend.agent.nodes.support.formatters import _format_renewal
        renewal = {
            "name": "Acme Corp",
            "company_id": "ACME-001",
            "renewal_date": "2025-04-01",
            "plan": "Enterprise",
            "health_flags": "Good",
        }
        result = _format_renewal(renewal)
        assert "Acme Corp" in result
        assert "ACME-001" in result
        assert "2025-04-01" in result
        assert "Enterprise" in result
        assert "Good" in result


class TestFormatCompaniesListSection:
    """Tests for format_company_section with companies list."""

    def test_formats_companies_list(self):
        """Formats list of companies from search."""
        from backend.agent.nodes.support.formatters import format_company_section
        data = {
            "companies": [
                {"name": "Acme", "company_id": "A001", "industry": "Tech",
                 "segment": "Enterprise", "status": "Active", "health_flags": "Good"},
                {"name": "Beta Inc", "company_id": "B001", "industry": "Finance",
                 "segment": "SMB", "status": "Active", "health_flags": "At Risk"},
            ],
            "count": 2,
        }
        result = format_company_section(data)
        assert "COMPANY SEARCH RESULTS" in result
        assert "(2)" in result or "2 found" in result
        assert "Acme" in result
        assert "Beta Inc" in result

    def test_handles_empty_companies_list(self):
        """Returns empty for empty companies list (falsy check)."""
        from backend.agent.nodes.support.formatters import format_company_section
        # Empty list is falsy, so returns empty string
        data = {"companies": [], "count": 0}
        result = format_company_section(data)
        assert result == ""

    def test_format_companies_list_helper_handles_empty(self):
        """The format_section helper handles empty list via companies formatter."""
        from backend.agent.nodes.support.formatters import format_section
        result = format_section("companies", {"companies": [], "count": 0})
        assert "No companies found" in result


class TestFormatContactsSection:
    """Tests for format_contacts_section."""

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import format_contacts_section
        assert format_contacts_section(None) == ""

    def test_handles_no_contacts(self):
        """Shows message when no contacts."""
        from backend.agent.nodes.support.formatters import format_contacts_section
        result = format_contacts_section({"contacts": []})
        assert "No contacts found" in result

    def test_formats_contacts(self):
        """Formats contact list."""
        from backend.agent.nodes.support.formatters import format_contacts_section
        data = {
            "contacts": [
                {"first_name": "John", "last_name": "Doe", "job_title": "CEO",
                 "company_id": "ACME-001", "contact_role": "Decision Maker", "email": "john@acme.com"},
            ],
            "count": 1,
        }
        result = format_contacts_section(data)
        assert "CONTACTS" in result
        assert "(1)" in result or "1 found" in result
        assert "John Doe" in result
        assert "CEO" in result
        assert "Decision Maker" in result


class TestFormatGroupsSection:
    """Tests for format_groups_section."""

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import format_groups_section
        assert format_groups_section(None) == ""

    def test_returns_empty_for_empty_data(self):
        """Returns empty string for empty data."""
        from backend.agent.nodes.support.formatters import format_groups_section
        assert format_groups_section({}) == ""

    def test_formats_group_members(self):
        """Formats group members list."""
        from backend.agent.nodes.support.formatters import format_groups_section
        data = {
            "group_name": "Enterprise Accounts",
            "members": [
                {"name": "Acme Corp", "company_id": "A001"},
                {"name": "Beta Inc", "company_id": "B001"},
            ],
        }
        result = format_groups_section(data)
        assert "Enterprise Accounts" in result
        assert "2 members" in result
        assert "Acme Corp" in result

    def test_formats_groups_list(self):
        """Formats list of groups."""
        from backend.agent.nodes.support.formatters import format_groups_section
        data = {
            "groups": [
                {"name": "Enterprise", "group_id": "G001", "description": "Large accounts"},
                {"name": "SMB", "group_id": "G002", "description": "Small business"},
            ],
        }
        result = format_groups_section(data)
        assert "ACCOUNT GROUPS" in result
        assert "2 groups" in result
        assert "Enterprise" in result
        assert "Large accounts" in result


class TestFormatAttachmentsSection:
    """Tests for format_attachments_section."""

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import format_attachments_section
        assert format_attachments_section(None) == ""

    def test_handles_no_attachments(self):
        """Shows message when no attachments."""
        from backend.agent.nodes.support.formatters import format_attachments_section
        result = format_attachments_section({"attachments": []})
        assert "No attachments found" in result

    def test_formats_attachments(self):
        """Formats attachment list."""
        from backend.agent.nodes.support.formatters import format_attachments_section
        data = {
            "attachments": [
                {"name": "Contract.pdf", "file_type": "PDF",
                 "description": "Signed contract for 2025", "company_id": "ACME-001"},
            ],
            "count": 1,
        }
        result = format_attachments_section(data)
        assert "ATTACHMENTS" in result
        assert "(1)" in result or "1 found" in result
        assert "Contract.pdf" in result
        assert "PDF" in result


class TestFormatAccountContextSection:
    """Tests for format_account_context_section."""

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import format_account_context_section
        assert format_account_context_section(None) == ""
        assert format_account_context_section("") == ""

    def test_formats_account_context(self):
        """Formats account context."""
        from backend.agent.nodes.support.formatters import format_account_context_section
        result = format_account_context_section("Important notes about the account...")
        assert "ACCOUNT CONTEXT" in result
        assert "Important notes" in result


class TestFormatConversationHistorySection:
    """Tests for format_conversation_history_section."""

    def test_returns_empty_for_none(self):
        """Returns empty string for None."""
        from backend.agent.nodes.support.formatters import format_conversation_history_section
        assert format_conversation_history_section(None) == ""

    def test_returns_empty_for_empty_list(self):
        """Returns empty string for empty list."""
        from backend.agent.nodes.support.formatters import format_conversation_history_section
        assert format_conversation_history_section([]) == ""

    def test_formats_conversation_history(self):
        """Formats conversation messages."""
        from backend.agent.nodes.support.formatters import format_conversation_history_section
        messages = [
            {"role": "user", "content": "What's happening with Acme?"},
            {"role": "assistant", "content": "Acme Corp is doing well..."},
        ]
        result = format_conversation_history_section(messages)
        assert "RECENT CONVERSATION" in result
        assert "User:" in result
        assert "Assistant:" in result
        assert "Acme" in result

    def test_truncates_long_messages(self):
        """Truncates messages longer than 150 chars."""
        from backend.agent.nodes.support.formatters import format_conversation_history_section
        long_message = "x" * 200
        messages = [{"role": "user", "content": long_message}]
        result = format_conversation_history_section(messages)
        assert "..." in result
        assert len(result) < 200 + 50  # Header + truncated content

    def test_respects_max_messages(self):
        """Respects max_messages parameter."""
        from backend.agent.nodes.support.formatters import format_conversation_history_section
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        result = format_conversation_history_section(messages, max_messages=2)
        assert "Message 8" in result
        assert "Message 9" in result
        assert "Message 0" not in result


class TestFormatPipelineSectionOpportunities:
    """Additional tests for format_pipeline_section with opportunities."""

    def test_includes_opportunities_list(self):
        """Includes opportunities in output."""
        from backend.agent.nodes.support.formatters import format_pipeline_section
        data = {
            "summary": {
                "total_count": 2,
                "total_value": 75000,
                "stages": {"Proposal": {"count": 2, "total_value": 75000}},
            },
            "opportunities": [
                {"name": "Big Deal", "stage": "Proposal", "value": 50000, "expected_close_date": "2025-03-01"},
                {"name": "Small Deal", "stage": "Proposal", "value": 25000, "expected_close_date": "2025-02-15"},
            ],
        }
        result = format_pipeline_section(data)
        assert "Open Opportunities" in result
        assert "Big Deal" in result
        assert "$50,000" in result
