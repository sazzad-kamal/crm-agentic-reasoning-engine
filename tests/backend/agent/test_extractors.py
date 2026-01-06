"""
Tests for backend/agent/extractors.py - extraction helpers for parsing user questions.
"""

import pytest

from backend.agent.handlers.extractors import (
    extract_role_from_question,
    extract_company_criteria,
    extract_group_id,
    extract_attachment_query,
    extract_activity_type,
)


# =============================================================================
# extract_role_from_question Tests
# =============================================================================

class TestExtractRoleFromQuestion:
    """Tests for extract_role_from_question function."""

    def test_extracts_decision_maker(self):
        """Extracts Decision Maker role."""
        assert extract_role_from_question("Who is the decision maker?") == "Decision Maker"
        assert extract_role_from_question("Find decision-maker contacts") == "Decision Maker"
        assert extract_role_from_question("Show me the decision maker") == "Decision Maker"

    def test_extracts_champion(self):
        """Extracts Champion role."""
        assert extract_role_from_question("Who is the champion?") == "Champion"
        assert extract_role_from_question("Find our champion contact") == "Champion"

    def test_extracts_executive(self):
        """Extracts Executive role."""
        assert extract_role_from_question("Who are the executives?") == "Executive"
        assert extract_role_from_question("Find VP contacts") == "Executive"
        assert extract_role_from_question("Show me directors") == "Executive"

    def test_returns_none_for_no_role(self):
        """Returns None when no role specified."""
        assert extract_role_from_question("Show me all contacts") is None
        assert extract_role_from_question("Who should I call?") is None

    def test_case_insensitive(self):
        """Handles case insensitivity."""
        assert extract_role_from_question("DECISION MAKER") == "Decision Maker"
        assert extract_role_from_question("Champion") == "Champion"
        assert extract_role_from_question("VP") == "Executive"


# =============================================================================
# extract_company_criteria Tests
# =============================================================================

class TestExtractCompanyCriteria:
    """Tests for extract_company_criteria function."""

    def test_extracts_enterprise_segment(self):
        """Extracts Enterprise segment."""
        segment, industry = extract_company_criteria("Show enterprise accounts")
        assert segment == "Enterprise"

    def test_extracts_smb_segment(self):
        """Extracts SMB segment."""
        segment, industry = extract_company_criteria("List SMB companies")
        assert segment == "SMB"

    def test_extracts_midmarket_segment(self):
        """Extracts Mid-Market segment."""
        segment, industry = extract_company_criteria("Find mid-market accounts")
        assert segment == "Mid-Market"

        segment, _ = extract_company_criteria("Show midmarket companies")
        assert segment == "Mid-Market"

    def test_extracts_software_industry(self):
        """Extracts Software industry."""
        _, industry = extract_company_criteria("Companies in software")
        assert industry == "Software"

    def test_extracts_manufacturing_industry(self):
        """Extracts Manufacturing industry."""
        _, industry = extract_company_criteria("Manufacturing companies")
        assert industry == "Manufacturing"

    def test_extracts_healthcare_industry(self):
        """Extracts Healthcare industry."""
        _, industry = extract_company_criteria("Healthcare accounts")
        assert industry == "Healthcare"

    def test_extracts_food_industry(self):
        """Extracts Food industry."""
        _, industry = extract_company_criteria("Food companies")
        assert industry == "Food"

    def test_extracts_consulting_industry(self):
        """Extracts Consulting industry."""
        _, industry = extract_company_criteria("Consulting firms")
        assert industry == "Consulting"

    def test_extracts_retail_industry(self):
        """Extracts Retail industry."""
        _, industry = extract_company_criteria("Retail companies")
        assert industry == "Retail"

    def test_extracts_both_segment_and_industry(self):
        """Extracts both segment and industry."""
        segment, industry = extract_company_criteria("Enterprise software companies")
        assert segment == "Enterprise"
        assert industry == "Software"

    def test_returns_none_for_no_criteria(self):
        """Returns None for both when no criteria."""
        segment, industry = extract_company_criteria("Show me all companies")
        assert segment is None
        assert industry is None

    def test_case_insensitive(self):
        """Handles case insensitivity."""
        segment, industry = extract_company_criteria("ENTERPRISE SOFTWARE")
        assert segment == "Enterprise"
        assert industry == "Software"


# =============================================================================
# extract_group_id Tests
# =============================================================================

class TestExtractGroupId:
    """Tests for extract_group_id function."""

    def test_extracts_at_risk_group(self):
        """Extracts at-risk group ID."""
        assert extract_group_id("Show at risk accounts") == "GRP-AT-RISK"
        assert extract_group_id("List at-risk companies") == "GRP-AT-RISK"

    def test_extracts_champions_group(self):
        """Extracts champions group ID."""
        assert extract_group_id("Show champion accounts") == "GRP-CHAMPIONS"

    def test_extracts_churned_group(self):
        """Extracts churned group ID."""
        assert extract_group_id("List churned customers") == "GRP-CHURNED"

    def test_extracts_dormant_group(self):
        """Extracts dormant group ID."""
        assert extract_group_id("Find dormant accounts") == "GRP-DORMANT"

    def test_extracts_hot_leads_group(self):
        """Extracts hot leads group ID."""
        assert extract_group_id("Show hot lead companies") == "GRP-HOT-LEADS"

    def test_returns_none_for_no_group(self):
        """Returns None when no group keyword."""
        assert extract_group_id("Show me all companies") is None
        assert extract_group_id("List customer accounts") is None

    def test_case_insensitive(self):
        """Handles case insensitivity."""
        assert extract_group_id("AT RISK accounts") == "GRP-AT-RISK"
        assert extract_group_id("CHURNED customers") == "GRP-CHURNED"


# =============================================================================
# extract_attachment_query Tests
# =============================================================================

class TestExtractAttachmentQuery:
    """Tests for extract_attachment_query function."""

    def test_extracts_proposal(self):
        """Extracts proposal keyword."""
        result = extract_attachment_query("Find the proposal")
        assert "proposal" in result

    def test_extracts_contract(self):
        """Extracts contract keyword."""
        result = extract_attachment_query("Show me the contract")
        assert "contract" in result

    def test_extracts_document(self):
        """Extracts document keyword."""
        result = extract_attachment_query("Find the document")
        assert "document" in result

    def test_extracts_agreement(self):
        """Extracts agreement keyword."""
        result = extract_attachment_query("Show the agreement")
        assert "agreement" in result

    def test_extracts_pdf(self):
        """Extracts pdf keyword."""
        result = extract_attachment_query("Find the PDF file")
        assert "pdf" in result

    def test_extracts_report(self):
        """Extracts report keyword."""
        result = extract_attachment_query("Show me the report")
        assert "report" in result

    def test_extracts_multiple_keywords(self):
        """Extracts multiple keywords."""
        result = extract_attachment_query("Find the proposal and contract documents")
        assert "proposal" in result
        assert "contract" in result
        assert "document" in result

    def test_returns_none_for_no_keywords(self):
        """Returns None when no attachment keywords."""
        assert extract_attachment_query("Show me the company info") is None
        assert extract_attachment_query("List contacts") is None

    def test_case_insensitive(self):
        """Handles case insensitivity."""
        result = extract_attachment_query("Find the PROPOSAL")
        assert "proposal" in result


# =============================================================================
# extract_activity_type Tests
# =============================================================================

class TestExtractActivityType:
    """Tests for extract_activity_type function."""

    def test_extracts_call(self):
        """Extracts Call activity type."""
        assert extract_activity_type("Show recent calls") == "Call"
        assert extract_activity_type("Schedule a call") == "Call"

    def test_extracts_email(self):
        """Extracts Email activity type."""
        assert extract_activity_type("Show email activities") == "Email"
        assert extract_activity_type("Send an email") == "Email"

    def test_extracts_meeting(self):
        """Extracts Meeting activity type."""
        assert extract_activity_type("Schedule a meeting") == "Meeting"
        assert extract_activity_type("Show recent meetings") == "Meeting"

    def test_extracts_task(self):
        """Extracts Task activity type."""
        assert extract_activity_type("Show pending tasks") == "Task"
        assert extract_activity_type("Create a task") == "Task"

    def test_returns_none_for_no_type(self):
        """Returns None when no activity type specified."""
        assert extract_activity_type("Show recent activities") is None
        assert extract_activity_type("What happened recently?") is None

    def test_case_insensitive(self):
        """Handles case insensitivity."""
        assert extract_activity_type("Show CALLS") == "Call"
        assert extract_activity_type("EMAIL activities") == "Email"

    def test_priority_order(self):
        """Respects priority order (call > email > meeting > task)."""
        # When multiple types present, first match wins
        assert extract_activity_type("call or email") == "Call"
        assert extract_activity_type("email or meeting") == "Email"
