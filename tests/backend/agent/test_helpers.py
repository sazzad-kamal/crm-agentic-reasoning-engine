"""
Tests for agent helper modules (formatters, llm_helpers, progress).

Tests the extracted modules from orchestrator.py refactoring.
"""

import os
import time
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
        from backend.agent.formatters import format_company_section
        assert format_company_section(None) == ""
    
    def test_returns_empty_for_missing_company(self):
        """Returns empty string when company key missing."""
        from backend.agent.formatters import format_company_section
        assert format_company_section({}) == ""
        assert format_company_section({"other": "data"}) == ""
    
    def test_formats_basic_company_info(self):
        """Formats basic company information."""
        from backend.agent.formatters import format_company_section
        
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
        from backend.agent.formatters import format_company_section
        
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
        from backend.agent.formatters import format_company_section
        
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
        from backend.agent.formatters import format_activities_section
        assert format_activities_section(None) == ""
    
    def test_handles_no_activities(self):
        """Shows message when no activities."""
        from backend.agent.formatters import format_activities_section
        
        result = format_activities_section({"activities": []})
        assert "No recent activities" in result
    
    def test_formats_activities(self):
        """Formats activity list."""
        from backend.agent.formatters import format_activities_section
        
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
        from backend.agent.formatters import format_history_section
        assert format_history_section(None) == ""
    
    def test_handles_no_history(self):
        """Shows message when no history."""
        from backend.agent.formatters import format_history_section
        
        result = format_history_section({"history": []})
        assert "No recent history" in result
    
    def test_formats_history_entries(self):
        """Formats history entries."""
        from backend.agent.formatters import format_history_section
        
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
        from backend.agent.formatters import format_pipeline_section
        assert format_pipeline_section(None) == ""
    
    def test_handles_empty_pipeline(self):
        """Shows message when no opportunities."""
        from backend.agent.formatters import format_pipeline_section
        
        result = format_pipeline_section({"summary": {"total_count": 0}})
        assert "No open opportunities" in result
    
    def test_formats_pipeline_summary(self):
        """Formats pipeline summary."""
        from backend.agent.formatters import format_pipeline_section
        
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
        from backend.agent.formatters import format_renewals_section
        assert format_renewals_section(None) == ""
    
    def test_handles_no_renewals(self):
        """Shows message when no renewals."""
        from backend.agent.formatters import format_renewals_section
        
        result = format_renewals_section({"renewals": []})
        assert "No renewals" in result
    
    def test_formats_renewals(self):
        """Formats renewal list."""
        from backend.agent.formatters import format_renewals_section
        
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
        from backend.agent.formatters import format_docs_section
        assert format_docs_section(None) == ""
        assert format_docs_section("") == ""
    
    def test_formats_docs_answer(self):
        """Formats documentation answer."""
        from backend.agent.formatters import format_docs_section
        
        result = format_docs_section("Here is how to import contacts...")
        assert "DOCUMENTATION GUIDANCE" in result
        assert "Here is how to import contacts" in result


# =============================================================================
# Progress Tests
# =============================================================================

class TestAgentProgress:
    """Tests for AgentProgress class."""
    
    def test_starts_with_empty_steps(self):
        """Starts with no steps."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        assert progress.steps == []
    
    def test_add_step(self):
        """Can add a step."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        progress.add_step("route", "Analyzing question", "done")
        
        assert len(progress.steps) == 1
        assert progress.steps[0].id == "route"
        assert progress.steps[0].label == "Analyzing question"
        assert progress.steps[0].status == "done"
    
    def test_add_multiple_steps(self):
        """Can add multiple steps."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        progress.add_step("step1", "First step")
        progress.add_step("step2", "Second step")
        progress.add_step("step3", "Third step")
        
        assert len(progress.steps) == 3
    
    def test_default_status_is_done(self):
        """Default status is 'done'."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        progress.add_step("test", "Test step")
        
        assert progress.steps[0].status == "done"
    
    def test_get_elapsed_ms(self):
        """Tracks elapsed time."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        time.sleep(0.05)  # 50ms
        elapsed = progress.get_elapsed_ms()
        
        assert elapsed >= 40  # Allow some variance
        assert elapsed < 200  # But not too much
    
    def test_to_list(self):
        """Converts steps to list of dicts."""
        from backend.agent.progress import AgentProgress
        
        progress = AgentProgress()
        progress.add_step("s1", "Step 1", "done")
        progress.add_step("s2", "Step 2", "pending")
        
        result = progress.to_list()
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "s1"
        assert result[1]["status"] == "pending"


# =============================================================================
# LLM Helpers Tests
# =============================================================================

class TestMockLlmResponse:
    """Tests for mock_llm_response."""
    
    def test_returns_string(self):
        """Returns a string response."""
        from backend.agent.llm_helpers import mock_llm_response
        
        result = mock_llm_response("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_handles_unknown_company_prompt(self):
        """Returns appropriate response for unknown company."""
        from backend.agent.llm_helpers import mock_llm_response
        
        result = mock_llm_response("couldn't find an exact match for company")
        assert "couldn't find" in result.lower() or "clarify" in result.lower()
    
    def test_handles_renewal_prompt(self):
        """Returns renewal-related response."""
        from backend.agent.llm_helpers import mock_llm_response
        
        result = mock_llm_response("Show me upcoming renewals for Q1")
        assert "renewal" in result.lower()
    
    def test_handles_pipeline_prompt(self):
        """Returns pipeline-related response."""
        from backend.agent.llm_helpers import mock_llm_response
        
        result = mock_llm_response("What's in the pipeline?")
        assert "pipeline" in result.lower()
    
    def test_default_response(self):
        """Returns default response for other prompts."""
        from backend.agent.llm_helpers import mock_llm_response
        
        result = mock_llm_response("Tell me about the account")
        assert "account" in result.lower() or "summary" in result.lower()


class TestCallAnswerChainMockMode:
    """Tests for call_answer_chain in mock mode."""

    def test_returns_tuple(self):
        """Returns tuple of (response, latency)."""
        from backend.agent.llm_helpers import call_answer_chain

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
        from backend.agent.llm_helpers import call_answer_chain

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
