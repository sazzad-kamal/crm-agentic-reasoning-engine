"""Tests for agent schemas."""

import pytest
from pydantic import ValidationError

from backend.agent.schemas import (
    Source,
    Step,
    RawData,
    MetaInfo,
    ChatRequest,
    ChatResponse,
    RouterResult,
    ToolResult,
)


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


class TestStep:
    """Tests for Step model."""

    def test_step_creation(self):
        """Test basic Step creation."""
        step = Step(id="routing", label="Routing query")
        assert step.id == "routing"
        assert step.label == "Routing query"
        assert step.status == "done"  # default

    def test_step_status_values(self):
        """Test Step with different status values."""
        for status in ["done", "error", "skipped"]:
            step = Step(id="test", label="Test", status=status)
            assert step.status == status

    def test_step_default_status_is_done(self):
        """Test Step defaults to done status."""
        step = Step(id="test", label="Test")
        assert step.status == "done"


class TestRawData:
    """Tests for RawData model."""

    def test_raw_data_defaults(self):
        """Test RawData has empty list defaults."""
        raw = RawData()
        assert raw.companies == []
        assert raw.activities == []
        assert raw.opportunities == []
        assert raw.history == []
        assert raw.renewals == []
        assert raw.pipeline_summary is None

    def test_raw_data_with_data(self):
        """Test RawData with populated fields."""
        raw = RawData(
            companies=[{"id": "C001", "name": "Acme"}],
            activities=[{"id": "A001", "type": "call"}],
            pipeline_summary={"total": 5},
        )
        assert len(raw.companies) == 1
        assert len(raw.activities) == 1
        assert raw.pipeline_summary["total"] == 5


class TestMetaInfo:
    """Tests for MetaInfo model."""

    def test_meta_info_required_fields(self):
        """Test MetaInfo with required fields."""
        meta = MetaInfo(mode_used="docs", latency_ms=150)
        assert meta.mode_used == "docs"
        assert meta.latency_ms == 150
        assert meta.company_id is None
        assert meta.days is None

    def test_meta_info_with_optional_fields(self):
        """Test MetaInfo with all fields."""
        meta = MetaInfo(
            mode_used="data",
            latency_ms=200,
            company_id="C001",
            days=30,
        )
        assert meta.company_id == "C001"
        assert meta.days == 30

    def test_meta_info_mode_values(self):
        """Test MetaInfo with different mode values."""
        for mode in ["docs", "data", "data+docs"]:
            meta = MetaInfo(mode_used=mode, latency_ms=100)
            assert meta.mode_used == mode


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_chat_request_minimal(self):
        """Test ChatRequest with only question."""
        req = ChatRequest(question="What is Acme?")
        assert req.question == "What is Acme?"
        assert req.mode == "auto"  # default
        assert req.session_id is None
        assert req.user_id is None
        assert req.company_id is None

    def test_chat_request_full(self):
        """Test ChatRequest with all fields."""
        req = ChatRequest(
            question="Tell me about Acme",
            mode="data",
            session_id="sess-123",
            user_id="user-456",
            company_id="C001",
        )
        assert req.mode == "data"
        assert req.session_id == "sess-123"
        assert req.user_id == "user-456"
        assert req.company_id == "C001"

    def test_chat_request_mode_options(self):
        """Test ChatRequest with different modes."""
        for mode in ["auto", "docs", "data", "data+docs"]:
            req = ChatRequest(question="Test", mode=mode)
            assert req.mode == mode


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_chat_response_creation(self):
        """Test ChatResponse with required fields."""
        response = ChatResponse(
            answer="Acme is a company.",
            sources=[Source(type="company", id="C001", label="Acme")],
            steps=[Step(id="route", label="Routing")],
            raw_data=RawData(),
            meta=MetaInfo(mode_used="data", latency_ms=100),
        )
        assert response.answer == "Acme is a company."
        assert len(response.sources) == 1
        assert len(response.steps) == 1
        assert response.follow_up_suggestions == []  # default

    def test_chat_response_with_suggestions(self):
        """Test ChatResponse with follow-up suggestions."""
        response = ChatResponse(
            answer="Test answer",
            sources=[],
            steps=[],
            raw_data=RawData(),
            meta=MetaInfo(mode_used="docs", latency_ms=50),
            follow_up_suggestions=["What else?", "Tell me more"],
        )
        assert len(response.follow_up_suggestions) == 2


class TestRouterResult:
    """Tests for RouterResult model."""

    def test_router_result_defaults(self):
        """Test RouterResult with minimal fields."""
        result = RouterResult(mode_used="docs")
        assert result.mode_used == "docs"
        assert result.company_id is None
        assert result.company_name_query is None
        assert result.days == 90  # default
        assert result.intent == "general"  # default
        assert result.key_entities == []  # default

    def test_router_result_full(self):
        """Test RouterResult with all fields."""
        result = RouterResult(
            mode_used="data",
            company_id="C001",
            company_name_query="Acme",
            days=30,
            intent="company_status",
            query_expansion="Tell me about Acme Corp's status",
            llm_confidence=0.95,
            key_entities=["Acme", "status"],
            action_type="retrieve",
        )
        assert result.company_id == "C001"
        assert result.days == 30
        assert result.intent == "company_status"
        assert result.llm_confidence == 0.95
        assert len(result.key_entities) == 2

    def test_router_result_intent_values(self):
        """Test RouterResult with different intents."""
        for intent in ["company_status", "renewals", "pipeline", "docs", "general"]:
            result = RouterResult(mode_used="data", intent=intent)
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

    def test_tool_result_data_any_type(self):
        """Test ToolResult data can be any type."""
        # dict
        r1 = ToolResult(data={"key": "value"}, sources=[])
        assert r1.data["key"] == "value"
        
        # list
        r2 = ToolResult(data=[1, 2, 3], sources=[])
        assert len(r2.data) == 3
        
        # string
        r3 = ToolResult(data="text result", sources=[])
        assert r3.data == "text result"
