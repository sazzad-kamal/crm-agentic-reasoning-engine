"""
Tests for Agent config, LLM router, and audit functionality.

Tests the new production-grade features:
- Centralized configuration
- LLM-based routing (with mock)
- Audit logging
- Progress tracking
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, UTC

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from backend.agent.core.config import (
    AgentConfig,
    get_config,
    reset_config,
    is_mock_mode,
)
from backend.agent.output.audit import AgentAuditLogger, AgentAuditEntry
from backend.agent.core.state import AgentProgress


# =============================================================================
# Config Tests
# =============================================================================

class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def setup_method(self):
        """Reset config between tests."""
        reset_config()
    
    def test_default_config_loads(self):
        """Test that default config loads without errors."""
        config = AgentConfig()
        assert config.llm_model == "gpt-5.2"
        assert config.router_model == "gpt-4o-mini"

    def test_config_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4")
        monkeypatch.setenv("AGENT_ROUTER_MODEL", "gpt-4o")
        reset_config()

        config = AgentConfig()
        assert config.llm_model == "gpt-4"
        assert config.router_model == "gpt-4o"
    
    def test_config_validation_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError):
            AgentConfig(llm_temperature=3.0)
    
    def test_config_validation_negative_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError):
            AgentConfig(llm_temperature=-0.5)
    
    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_reset_config(self):
        """Test that reset_config clears singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2
    
    def test_is_mock_mode(self):
        """Test mock mode detection."""
        # Already set in module setup
        assert is_mock_mode() is True


# =============================================================================
# Audit Tests
# =============================================================================

class TestAgentAudit:
    """Tests for AgentAuditLogger."""
    
    def test_audit_entry_creation(self):
        """Test creating an audit entry."""
        entry = AgentAuditEntry(
            timestamp=datetime.now(UTC).isoformat(),
            question="Test question",
            mode_used="data",
            company_id="C001",
            latency_ms=150,
            source_count=3,
        )
        
        assert entry.question == "Test question"
        assert entry.mode_used == "data"
        assert entry.company_id == "C001"
        assert entry.latency_ms == 150
    
    def test_audit_entry_to_dict(self):
        """Test converting entry to dict excludes None values."""
        entry = AgentAuditEntry(
            timestamp="2024-01-01T00:00:00",
            question="Test",
            mode_used="docs",
        )
        
        d = entry.to_dict()
        assert "timestamp" in d
        assert "question" in d
        assert "company_id" not in d  # None values excluded
    
    def test_audit_logger_writes_to_file(self):
        """Test that audit logger writes to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        
        try:
            # Create logger with temp file
            logger = AgentAuditLogger(log_file=log_path)
            
            # Log a query
            logger.log_query(
                question="Test question",
                mode_used="data+docs",
                company_id="C001",
                latency_ms=100,
            )
            
            # Read back and verify
            with open(log_path, "r") as f:
                content = f.read()
            
            assert "Test question" in content
            assert "data+docs" in content
            assert "C001" in content
        finally:
            if log_path.exists():
                log_path.unlink()
    
# =============================================================================
# Progress Tracking Tests
# =============================================================================

class TestAgentProgress:
    """Tests for AgentProgress tracking."""
    
    def test_progress_add_step(self):
        """Test adding steps to progress."""
        progress = AgentProgress()
        
        progress.add_step("router", "Understanding question")
        progress.add_step("data", "Fetching data")
        
        steps = progress.to_list()
        assert len(steps) == 2
        assert steps[0]["id"] == "router"
        assert steps[1]["id"] == "data"
    
    def test_progress_step_status(self):
        """Test step status is recorded."""
        progress = AgentProgress()
        
        progress.add_step("router", "Understanding question", status="done")
        progress.add_step("error", "Failed", status="error")
        
        steps = progress.to_list()
        assert steps[0]["status"] == "done"
        assert steps[1]["status"] == "error"
    
    def test_progress_elapsed_time(self):
        """Test elapsed time calculation."""
        import time
        
        progress = AgentProgress()
        time.sleep(0.05)  # 50ms
        elapsed = progress.get_elapsed_ms()
        
        assert elapsed >= 50
        assert elapsed < 1000  # Shouldn't take a second


# =============================================================================
# LLM Router Tests (Mock Mode)
# =============================================================================

class TestLLMRouter:
    """Tests for LLM router with mock mode."""
    
    def setup_method(self):
        """Reset config between tests."""
        reset_config()
    
    def test_router_returns_default_in_mock_mode(self):
        """Test that router returns default routing in mock mode."""
        from backend.agent.llm.router import route_question

        result = route_question("What's going on with Acme Corp?")

        # In mock mode, returns data+docs with general intent
        assert result.mode_used == "data+docs"
        assert result.intent == "general"

    def test_router_returns_default_days_in_mock(self):
        """Test that router returns default days in mock mode."""
        from backend.agent.llm.router import route_question

        result = route_question("What happened in the last 30 days?")

        # Mock mode returns default 30 days
        assert result.days == 30
    
    def test_router_respects_explicit_mode(self):
        """Test that explicit mode is respected."""
        from backend.agent.llm.router import route_question
        
        result = route_question("Tell me about Acme", mode="docs")
        
        assert result.mode_used == "docs"
    
    def test_router_result_schema(self):
        """Test that RouterResult has all expected fields."""
        from backend.agent.llm.router import route_question
        
        result = route_question("Show me renewals")
        
        assert hasattr(result, "mode_used")
        assert hasattr(result, "company_id")
        assert hasattr(result, "days")
        assert hasattr(result, "intent")
        assert hasattr(result, "query_expansion")
        assert hasattr(result, "llm_confidence")


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for enhanced agent."""
    
    def setup_method(self):
        """Reset config between tests."""
        reset_config()
    
    def test_agent_uses_progress_tracking(self):
        """Test that agent uses progress tracking."""
        from backend.agent.graph import answer_question
        
        result = answer_question("Which accounts have renewals?")
        
        # Should have steps array
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # Steps should have expected structure
        step = result["steps"][0]
        assert "id" in step
        assert "label" in step
        assert "status" in step
    
    def test_agent_returns_meta_info(self):
        """Test that agent returns metadata."""
        from backend.agent.graph import answer_question
        
        result = answer_question("Show me pipeline")
        
        assert "meta" in result
        assert "mode_used" in result["meta"]
        assert "latency_ms" in result["meta"]
    
    def test_agent_handles_company_query(self):
        """Test agent handles company-specific queries."""
        from backend.agent.graph import answer_question
        
        result = answer_question("What's happening with Acme Manufacturing?")
        
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    def test_agent_handles_docs_query(self):
        """Test agent handles documentation queries."""
        from backend.agent.graph import answer_question
        
        result = answer_question("How do I create an opportunity?", mode="docs")
        
        assert "answer" in result
        assert result["meta"]["mode_used"] == "docs"


# =============================================================================
# Role-Based Starter Detection Tests (coverage improvement)
# =============================================================================


class TestDetectOwnerFromStarter:
    """Tests for detect_owner_from_starter function."""

    def test_detects_sales_rep_pipeline_starter(self):
        """Detects Sales Rep from 'my pipeline' starter."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's my pipeline") == "jsmith"
        assert detect_owner_from_starter("How is my pipeline?") == "jsmith"
        assert detect_owner_from_starter("show my pipeline") == "jsmith"

    def test_detects_csm_renewals_starter(self):
        """Detects CSM from renewals starter."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("any renewals at risk") == "amartin"
        assert detect_owner_from_starter("which renewals are at risk?") == "amartin"
        assert detect_owner_from_starter("at-risk renewals") == "amartin"

    def test_detects_manager_team_starter(self):
        """Detects Manager (None) from team starter."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's the team doing") is None
        assert detect_owner_from_starter("team performance") is None
        assert detect_owner_from_starter("how's my team?") is None

    def test_returns_none_for_non_starter(self):
        """Returns None for non-starter questions."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("Tell me about Acme Corp") is None
        assert detect_owner_from_starter("What's the weather?") is None

    def test_handles_case_insensitivity(self):
        """Handles case insensitivity."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("HOW'S MY PIPELINE") == "jsmith"
        assert detect_owner_from_starter("ANY RENEWALS AT RISK") == "amartin"

    def test_strips_question_marks(self):
        """Strips trailing question marks."""
        from backend.agent.llm.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's my pipeline???") == "jsmith"


# =============================================================================
# LLMRouterResponse Model Tests (coverage improvement)
# =============================================================================


class TestLLMRouterResponse:
    """Tests for LLMRouterResponse Pydantic model."""

    def test_creates_with_defaults(self):
        """Creates model with default values."""
        from backend.agent.llm.router import LLMRouterResponse

        response = LLMRouterResponse()

        assert response.mode == "data+docs"
        assert response.intent == "general"
        assert response.days == 30
        assert response.company_name is None
        assert response.confidence == 0.5

    def test_creates_with_custom_values(self):
        """Creates model with custom values."""
        from backend.agent.llm.router import LLMRouterResponse

        response = LLMRouterResponse(
            mode="data",
            intent="pipeline_summary",
            company_name="Acme Corp",
            days=90,
            confidence=0.95,
            key_entities=["Acme", "Q4"],
            action_type="analyze",
        )

        assert response.mode == "data"
        assert response.intent == "pipeline_summary"
        assert response.company_name == "Acme Corp"
        assert response.days == 90
        assert response.confidence == 0.95
        assert "Acme" in response.key_entities

    def test_validates_days_range(self):
        """Validates days must be between 1 and 365."""
        from backend.agent.llm.router import LLMRouterResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterResponse(days=0)

        with pytest.raises(ValidationError):
            LLMRouterResponse(days=400)

    def test_validates_confidence_range(self):
        """Validates confidence must be between 0.0 and 1.0."""
        from backend.agent.llm.router import LLMRouterResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterResponse(confidence=-0.1)

        with pytest.raises(ValidationError):
            LLMRouterResponse(confidence=1.5)

    def test_validates_mode_literal(self):
        """Validates mode must be a valid literal."""
        from backend.agent.llm.router import LLMRouterResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterResponse(mode="invalid_mode")

    def test_validates_intent_literal(self):
        """Validates intent must be a valid literal."""
        from backend.agent.llm.router import LLMRouterResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterResponse(intent="unknown_intent")

    def test_model_dump(self):
        """Tests model_dump serialization."""
        from backend.agent.llm.router import LLMRouterResponse

        response = LLMRouterResponse(
            mode="docs",
            intent="general",
            query_expansion="Expanded query",
        )

        data = response.model_dump()
        assert data["mode"] == "docs"
        assert data["query_expansion"] == "Expanded query"


# =============================================================================
# LLMRouterError Tests (coverage improvement)
# =============================================================================


class TestLLMRouterError:
    """Tests for LLMRouterError exception."""

    def test_is_exception(self):
        """LLMRouterError is an Exception."""
        from backend.agent.llm.router import LLMRouterError

        error = LLMRouterError("Router failed")
        assert isinstance(error, Exception)
        assert str(error) == "Router failed"

    def test_can_be_raised(self):
        """LLMRouterError can be raised and caught."""
        from backend.agent.llm.router import LLMRouterError

        with pytest.raises(LLMRouterError) as exc_info:
            raise LLMRouterError("Test error message")

        assert "Test error message" in str(exc_info.value)


# =============================================================================
# Extended LLM Router Tests (coverage improvement)
# =============================================================================


class TestLLMRouterExtended:
    """Extended tests for LLM router coverage."""

    def setup_method(self):
        """Reset config between tests."""
        reset_config()

    def test_route_question_with_company_id(self):
        """Tests route_question with pre-specified company_id."""
        from backend.agent.llm.router import route_question

        result = route_question(
            "What's happening?",
            company_id="ACME-001",
        )

        # Company ID should be passed through
        assert result.company_id == "ACME-001"

    def test_route_question_passes_owner_from_starter(self):
        """Tests that owner is detected from starter pattern."""
        from backend.agent.llm.router import route_question

        result = route_question("how's my pipeline?")

        assert result.owner == "jsmith"

    def test_route_question_explicit_mode_returns_minimal_routing(self):
        """Explicit mode returns minimal routing without LLM."""
        from backend.agent.llm.router import route_question

        result = route_question(
            "Tell me about documents",
            mode="docs",
        )

        assert result.mode_used == "docs"
        assert result.intent == "general"

    def test_llm_route_question_mock_mode_returns_defaults(self):
        """In mock mode, returns default data+docs routing."""
        from backend.agent.llm.router import llm_route_question

        result = llm_route_question("Show me the forecast")

        assert result.mode_used == "data+docs"
        assert result.days == 30

    def test_route_question_detects_owner_as_fallback(self):
        """Tests owner detection as fallback in route_question."""
        from backend.agent.llm.router import route_question

        # CSM starter
        result = route_question("any renewals at risk?")
        assert result.owner == "amartin"

        # Manager starter (None means sees all)
        result = route_question("how's the team doing?")
        assert result.owner is None

    def test_llm_route_question_with_conversation_history(self):
        """Tests llm_route_question with conversation history."""
        from backend.agent.llm.router import llm_route_question

        result = llm_route_question(
            "What about their pipeline?",
            conversation_history="User: Tell me about Acme\nAssistant: Acme is...",
        )

        # Should still return default in mock mode
        assert result.mode_used == "data+docs"

    def test_starter_owner_map_coverage(self):
        """Tests all patterns in STARTER_OWNER_MAP."""
        from backend.agent.llm.router import STARTER_OWNER_MAP, detect_owner_from_starter

        # Test all patterns in the map
        for pattern, expected_owner in STARTER_OWNER_MAP.items():
            result = detect_owner_from_starter(pattern)
            assert result == expected_owner, f"Pattern '{pattern}' should return '{expected_owner}'"
