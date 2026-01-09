"""
Tests for Agent config, LLM router, and audit functionality.

Tests the new production-grade features:
- Centralized configuration
- LLM-based routing (with mock)
- Audit logging
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
from backend.agent.audit import AgentAuditLogger, AgentAuditEntry


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
            company_id="C001",
            latency_ms=150,
            source_count=3,
        )

        assert entry.question == "Test question"
        assert entry.company_id == "C001"
        assert entry.latency_ms == 150
    
    def test_audit_entry_to_dict(self):
        """Test converting entry to dict excludes None values."""
        entry = AgentAuditEntry(
            timestamp="2024-01-01T00:00:00",
            question="Test",
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
                company_id="C001",
                latency_ms=100,
            )

            # Read back and verify
            with open(log_path, "r") as f:
                content = f.read()

            assert "Test question" in content
            assert "C001" in content
        finally:
            if log_path.exists():
                log_path.unlink()
    
# =============================================================================
# LLM Router Tests (Mock Mode)
# =============================================================================

class TestLLMRouter:
    """Tests for LLM router with mock mode."""

    def setup_method(self):
        """Reset config between tests."""
        reset_config()

    def test_router_returns_result_in_mock_mode(self):
        """Test that router returns routing result in mock mode."""
        from backend.agent.route.router import route_question

        result = route_question("What's going on with Acme Corp?")

        # In mock mode, returns result with company and intent
        assert result.intent is not None

    def test_router_result_schema(self):
        """Test that RouterResult has expected fields (company_id, intent)."""
        from backend.agent.route.router import route_question

        result = route_question("Show me renewals")

        # RouterResult only has 2 fields now
        assert hasattr(result, "company_id")
        assert hasattr(result, "intent")


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for enhanced agent."""

    def setup_method(self):
        """Reset config between tests."""
        reset_config()

    def test_agent_returns_answer(self):
        """Test that agent returns an answer."""
        from backend.agent.graph import agent_graph, build_thread_config

        state = {"question": "Show me pipeline", "sources": []}
        config = build_thread_config(None)
        result = agent_graph.invoke(state, config=config)

        assert "answer" in result

    def test_agent_handles_company_query(self):
        """Test agent handles company-specific queries."""
        from backend.agent.graph import agent_graph, build_thread_config

        state = {"question": "What's happening with Acme Manufacturing?", "sources": []}
        config = build_thread_config(None)
        result = agent_graph.invoke(state, config=config)

        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_agent_handles_general_query(self):
        """Test agent handles general queries."""
        from backend.agent.graph import agent_graph, build_thread_config

        state = {"question": "How do I create an opportunity?", "sources": []}
        config = build_thread_config(None)
        result = agent_graph.invoke(state, config=config)

        assert "answer" in result


# =============================================================================
# Role-Based Starter Detection Tests (coverage improvement)
# =============================================================================


class TestDetectOwnerFromStarter:
    """Tests for detect_owner_from_starter function."""

    def test_detects_sales_rep_pipeline_starter(self):
        """Detects Sales Rep from 'my pipeline' starter."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's my pipeline") == "jsmith"
        assert detect_owner_from_starter("How is my pipeline?") == "jsmith"
        assert detect_owner_from_starter("show my pipeline") == "jsmith"

    def test_detects_csm_renewals_starter(self):
        """Detects CSM from renewals starter."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("any renewals at risk") == "amartin"
        assert detect_owner_from_starter("which renewals are at risk?") == "amartin"
        assert detect_owner_from_starter("at-risk renewals") == "amartin"

    def test_detects_manager_team_starter(self):
        """Detects Manager (None) from team starter."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's the team doing") is None
        assert detect_owner_from_starter("team performance") is None
        assert detect_owner_from_starter("how's my team?") is None

    def test_returns_none_for_non_starter(self):
        """Returns None for non-starter questions."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("Tell me about Acme Corp") is None
        assert detect_owner_from_starter("What's the weather?") is None

    def test_handles_case_insensitivity(self):
        """Handles case insensitivity."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("HOW'S MY PIPELINE") == "jsmith"
        assert detect_owner_from_starter("ANY RENEWALS AT RISK") == "amartin"

    def test_strips_question_marks(self):
        """Strips trailing question marks."""
        from backend.agent.route.router import detect_owner_from_starter

        assert detect_owner_from_starter("how's my pipeline???") == "jsmith"


# =============================================================================
# LLMRouterResponse Model Tests (coverage improvement)
# =============================================================================


class TestLLMRouterResponse:
    """Tests for LLMRouterResponse Pydantic model (2 fields: intent, company_name)."""

    def test_creates_with_defaults(self):
        """Creates model with default values."""
        from backend.agent.route.router import LLMRouterResponse

        response = LLMRouterResponse()

        assert response.intent == "pipeline_summary"
        assert response.company_name is None

    def test_creates_with_custom_values(self):
        """Creates model with custom values."""
        from backend.agent.route.router import LLMRouterResponse

        response = LLMRouterResponse(
            intent="company",
            company_name="Acme Corp",
        )

        assert response.intent == "company"
        assert response.company_name == "Acme Corp"

    def test_validates_intent_literal(self):
        """Validates intent must be a valid literal."""
        from backend.agent.route.router import LLMRouterResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterResponse(intent="unknown_intent")

    def test_model_dump(self):
        """Tests model_dump serialization."""
        from backend.agent.route.router import LLMRouterResponse

        response = LLMRouterResponse(
            intent="renewals",
            company_name="Test Corp",
        )

        data = response.model_dump()
        assert data["intent"] == "renewals"
        assert data["company_name"] == "Test Corp"


# =============================================================================
# LLMRouterError Tests (coverage improvement)
# =============================================================================


class TestLLMRouterError:
    """Tests for LLMRouterError exception."""

    def test_is_exception(self):
        """LLMRouterError is an Exception."""
        from backend.agent.route.router import LLMRouterError

        error = LLMRouterError("Router failed")
        assert isinstance(error, Exception)
        assert str(error) == "Router failed"

    def test_can_be_raised(self):
        """LLMRouterError can be raised and caught."""
        from backend.agent.route.router import LLMRouterError

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

    def test_llm_route_question_returns_router_result(self):
        """llm_route_question returns RouterResult with company_id and intent."""
        from backend.agent.route.router import llm_route_question

        result = llm_route_question("Show me the forecast")

        # RouterResult has only company_id and intent
        assert hasattr(result, "company_id")
        assert hasattr(result, "intent")

    def test_llm_route_question_with_conversation_history(self):
        """Tests llm_route_question with conversation history."""
        from backend.agent.route.router import llm_route_question

        result = llm_route_question(
            "What about their pipeline?",
            conversation_history="User: Tell me about Acme\nAssistant: Acme is...",
        )

        # Should return RouterResult with intent
        assert result.intent is not None

    def test_starter_owner_map_coverage(self):
        """Tests all patterns in STARTER_OWNER_MAP."""
        from backend.agent.route.router import STARTER_OWNER_MAP, detect_owner_from_starter

        # Test all patterns in the map
        for pattern, expected_owner in STARTER_OWNER_MAP.items():
            result = detect_owner_from_starter(pattern)
            assert result == expected_owner, f"Pattern '{pattern}' should return '{expected_owner}'"
