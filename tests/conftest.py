"""
Shared pytest fixtures for backend tests.

This module provides common fixtures used across test files:
- test_client: FastAPI TestClient instance
- mock_llm: Auto-patches LLM functions for testing
- db_connection: DuckDB connection instance
"""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# =============================================================================
# Environment Setup
# =============================================================================

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
# Disable rate limiting for tests
os.environ["ACME_RATE_LIMIT_ENABLED"] = "false"


# =============================================================================
# API Client Fixtures
# =============================================================================

@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app.

    This fixture provides a TestClient instance that can be used to make
    requests to the API without starting a real server.

    Usage:
        def test_endpoint(client):
            response = client.get("/api/data/companies")
            assert response.status_code == 200
    """
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def api_client(client):
    """Alias for client fixture for clearer naming in some contexts."""
    return client


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def db_connection():
    """
    Create a fresh DuckDB connection instance.

    This fixture provides access to the in-memory CRM data for testing
    data-related functionality.
    """
    from backend.agent.datastore.connection import get_connection
    return get_connection()


@pytest.fixture
def sample_company_id():
    """Return a known company ID for testing."""
    return "ACME-MFG"


@pytest.fixture
def sample_company_name():
    """Return a known company name for testing."""
    return "Acme Manufacturing"


# =============================================================================
# Mock LLM Response Functions
# =============================================================================


def _mock_answer_response(prompt: str) -> str:
    """Generate mock LLM answer based on prompt content."""
    if "renewal" in prompt.lower():
        return (
            "**Upcoming Renewals Summary**\n\n"
            "Based on the CRM data, here are the accounts with upcoming renewals:\n\n"
            "• Several accounts have renewals coming up in the specified timeframe\n"
            "• Review each account's health status before the renewal date\n\n"
            "**Suggested Actions:**\n"
            "1. Schedule check-in calls with at-risk accounts\n"
            "2. Prepare renewal proposals for key accounts\n"
            "3. Review recent activity levels to identify any concerns"
        )

    if "pipeline" in prompt.lower():
        return (
            "**Pipeline Summary**\n\n"
            "Here's the current pipeline status based on CRM data:\n\n"
            "• Open opportunities are progressing through various stages\n"
            "• Total pipeline value and deal count are shown in the data\n\n"
            "**Suggested Actions:**\n"
            "1. Focus on deals in Proposal and Negotiation stages\n"
            "2. Follow up on stalled opportunities\n"
            "3. Update expected close dates if needed"
        )

    # Default response for company status questions
    return (
        "**Account Summary**\n\n"
        "Based on the CRM data provided:\n\n"
        "• Recent activities show engagement with the account\n"
        "• Pipeline includes open opportunities in various stages\n"
        "• History log shows recent touchpoints\n\n"
        "**Suggested Actions:**\n"
        "1. Review recent activity and follow up if needed\n"
        "2. Check opportunity progress and update stages\n"
        "3. Confirm next steps with key contacts"
    )


def _mock_follow_up_suggestions(company_name: str | None, available_data: dict | None) -> list[str]:
    """Generate mock follow-up suggestions."""
    suggestions = []
    name = company_name or "the account"

    if available_data:
        if available_data.get("opportunities", 0) > 0:
            suggestions.append(f"What stage are {name}'s opportunities in?")
        if available_data.get("activities", 0) > 0:
            suggestions.append(f"What were {name}'s recent activities?")
        if available_data.get("contacts", 0) > 0:
            suggestions.append(f"Who are {name}'s key contacts?")
        if available_data.get("renewals", 0) > 0:
            suggestions.append(f"When is {name}'s renewal coming up?")

    suggestions = suggestions[:2]
    suggestions.append("Show me the overall pipeline summary")
    return suggestions[:3]


# =============================================================================
# LLM Mock Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def mock_llm():
    """
    Auto-patches all LLM functions for testing.

    This fixture automatically patches LLM API calls so tests don't require
    a real API key or make actual network requests.
    """
    from backend.agent.core import Source
    from backend.agent.route.query_planner import QueryPlan, SQLQuery

    def mock_call_answer_chain(*args, **kwargs) -> tuple[str, int]:
        question = kwargs.get("question", args[0] if args else "")
        return _mock_answer_response(question), 100

    async def mock_stream_answer_chain(*args, **kwargs):
        question = kwargs.get("question", args[0] if args else "")
        yield _mock_answer_response(question)

    def mock_call_account_rag(question: str, filters: dict[str, str]) -> tuple[str, list]:
        company_id = filters.get("company_id", "unknown")
        return (
            "Based on the account notes, the customer mentioned concerns about "
            "integration timeline during our last call.",
            [Source(type="account_note", id=f"{company_id}_notes", label="Account Notes")],
        )

    def mock_generate_follow_up_suggestions(
        question: str,
        company_id: str | None = None,
        company_name: str | None = None,
        conversation_history: str = "",
        available_data: dict | None = None,
        use_hardcoded_tree: bool = True,
    ) -> list[str]:
        # Still try hardcoded tree first
        if use_hardcoded_tree:
            from backend.agent.followup.tree import get_follow_ups
            follow_ups = get_follow_ups(question)
            if follow_ups:
                return follow_ups[:3]
        return _mock_follow_up_suggestions(company_name, available_data)

    def mock_get_query_plan(
        question: str,
        conversation_history: str = "",
        owner: str | None = None,
    ) -> QueryPlan:
        """Mock query planner that returns reasonable SQL queries."""
        q = question.lower()
        queries = []
        needs_rag = False

        # Detect if asking about specific company
        company_names = ["acme", "beta", "crown", "delta", "echo"]
        if any(name in q for name in company_names):
            needs_rag = True
            queries.append(SQLQuery(
                sql="SELECT * FROM companies LIMIT 1",
                purpose="company_info"
            ))
            queries.append(SQLQuery(
                sql="SELECT * FROM opportunities WHERE company_id = '$company_id' LIMIT 20",
                purpose="open_deals"
            ))

        # Detect aggregate queries
        if "renewal" in q:
            queries.append(SQLQuery(
                sql="SELECT * FROM companies WHERE renewal_date IS NOT NULL LIMIT 20",
                purpose="renewals"
            ))
        elif "pipeline" in q:
            queries.append(SQLQuery(
                sql="SELECT * FROM opportunities WHERE stage NOT LIKE '%Closed%' LIMIT 50",
                purpose="pipeline"
            ))
        elif any(kw in q for kw in ["forecast", "weighted"]):
            queries.append(SQLQuery(
                sql="SELECT stage, COUNT(*) as count, SUM(value) as total FROM opportunities WHERE stage NOT LIKE '%Closed%' GROUP BY stage",
                purpose="forecast"
            ))
        elif any(kw in q for kw in ["at risk", "at-risk", "stalled"]):
            queries.append(SQLQuery(
                sql="SELECT * FROM opportunities WHERE days_in_stage > 45 LIMIT 20",
                purpose="stale_deals"
            ))

        # Default: return something
        if not queries:
            queries.append(SQLQuery(
                sql="SELECT * FROM companies LIMIT 10",
                purpose="companies"
            ))

        return QueryPlan(queries=queries, needs_account_rag=needs_rag)

    with patch("backend.agent.answer.llm.call_answer_chain", mock_call_answer_chain), \
         patch("backend.agent.answer.llm.stream_answer_chain", mock_stream_answer_chain), \
         patch("backend.agent.fetch.fetch_rag._call_account_rag", mock_call_account_rag), \
         patch("backend.agent.followup.llm.generate_follow_up_suggestions", mock_generate_follow_up_suggestions), \
         patch("backend.agent.route.query_planner.get_query_plan", mock_get_query_plan):
        yield


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    return "Based on the CRM data, this is a mock response for testing purposes."


# =============================================================================
# Chat Request Fixtures
# =============================================================================

@pytest.fixture
def chat_request_company():
    """Sample chat request about a company."""
    return {"question": "What's going on with Acme Manufacturing?"}


@pytest.fixture
def chat_request_data():
    """Sample chat request for data mode."""
    return {"question": "Show me the pipeline", "mode": "data"}
