"""Tests for agent schemas."""

import pytest
from pydantic import ValidationError

from backend.agent.core.state import Source
from backend.agent.route.schemas import RouterResult
from backend.agent.fetch.tools.schemas import ToolResult
from backend.api.chat import ChatRequest


class TestSource:
    """Tests for Source model."""

    def test_source_creation(self):
        """Test basic Source creation."""
        source = Source(type="company", id="C001", label="Acme Corp")
        assert source.type == "company"
        assert source.id == "C001"
        assert source.label == "Acme Corp"

    def test_source_all_types(self):
        """Test Source with various type values."""
        for type_val in ["company", "doc", "activity", "opportunity", "history"]:
            source = Source(type=type_val, id="123", label="Test")
            assert source.type == type_val

    def test_source_requires_all_fields(self):
        """Test Source requires type, id, and label."""
        with pytest.raises(ValidationError):
            Source(type="company", id="123")  # missing label


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_chat_request_minimal(self):
        """Test ChatRequest with only question."""
        req = ChatRequest(question="What is Acme?")
        assert req.question == "What is Acme?"
        assert req.session_id is None

    def test_chat_request_full(self):
        """Test ChatRequest with all fields."""
        req = ChatRequest(
            question="Tell me about Acme",
            session_id="sess-123",
        )
        assert req.question == "Tell me about Acme"
        assert req.session_id == "sess-123"


class TestRouterResult:
    """Tests for RouterResult model (2 fields: company_id, intent)."""

    def test_router_result_defaults(self):
        """Test RouterResult with minimal fields."""
        result = RouterResult()
        assert result.company_id is None
        assert result.intent == "pipeline_summary"  # default

    def test_router_result_full(self):
        """Test RouterResult with all fields."""
        result = RouterResult(
            company_id="C001",
            intent="company",
        )
        assert result.company_id == "C001"
        assert result.intent == "company"

    def test_router_result_intent_values(self):
        """Test RouterResult with different intents."""
        for intent in ["company", "renewals", "pipeline_summary", "deals_at_risk", "contacts"]:
            result = RouterResult(intent=intent)
            assert result.intent == intent


class TestToolResult:
    """Tests for ToolResult model."""

    def test_tool_result_success(self):
        """Test ToolResult for successful operation."""
        result = ToolResult(
            data={"company": {"id": "C001", "name": "Acme"}},
            sources=[Source(type="company", id="C001", label="Acme")],
        )
        assert result.data["company"]["id"] == "C001"
        assert len(result.sources) == 1
        assert result.error is None

    def test_tool_result_with_error(self):
        """Test ToolResult with error."""
        result = ToolResult(
            data={"found": False},
            sources=[],
            error="Company not found",
        )
        assert result.error == "Company not found"
        assert result.data["found"] is False

    def test_tool_result_data_dict_type(self):
        """Test ToolResult data must be a dict."""
        # Valid dict
        r1 = ToolResult(data={"key": "value"}, sources=[])
        assert r1.data["key"] == "value"
        
        # Dict with nested data
        r2 = ToolResult(data={"items": [1, 2, 3], "count": 3}, sources=[])
        assert len(r2.data["items"]) == 3
        
        # Empty dict
        r3 = ToolResult(data={}, sources=[])
        assert r3.data == {}
