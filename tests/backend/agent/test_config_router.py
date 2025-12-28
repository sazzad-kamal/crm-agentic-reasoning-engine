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

from backend.agent.config import (
    AgentConfig, 
    get_config, 
    reset_config, 
    is_mock_mode,
)
from backend.agent.audit import AgentAuditLogger, AgentAuditEntry
from backend.agent.orchestrator import AgentProgress


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
        assert config.use_llm_router is True
    
    def test_config_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4")
        monkeypatch.setenv("AGENT_USE_LLM_ROUTER", "false")
        reset_config()
        
        config = AgentConfig()
        assert config.llm_model == "gpt-4"
        assert config.use_llm_router is False
    
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
    
    def test_audit_read_recent_queries(self):
        """Test reading recent queries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        
        try:
            logger = AgentAuditLogger(log_file=log_path)
            
            # Log multiple queries
            for i in range(5):
                logger.log_query(
                    question=f"Question {i}",
                    mode_used="data",
                    latency_ms=100 + i * 10,
                )
            
            # Read back
            queries = logger.read_recent_queries(limit=3)
            assert len(queries) == 3
            # Most recent first
            assert "Question 4" in queries[0]["question"]
        finally:
            if log_path.exists():
                log_path.unlink()
    
    def test_audit_get_stats(self):
        """Test getting statistics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        
        try:
            logger = AgentAuditLogger(log_file=log_path)
            
            # Log queries with different modes
            logger.log_query(question="Q1", mode_used="data", latency_ms=100)
            logger.log_query(question="Q2", mode_used="docs", latency_ms=200)
            logger.log_query(question="Q3", mode_used="data", latency_ms=300)
            
            stats = logger.get_stats()
            assert stats["total_queries"] == 3
            assert stats["modes"]["data"] == 2
            assert stats["modes"]["docs"] == 1
            assert stats["avg_latency_ms"] == 200
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
    
    def test_router_falls_back_to_heuristics_in_mock(self):
        """Test that router uses heuristics in mock mode."""
        from backend.agent.llm_router import route_question
        
        result = route_question("What's going on with Acme Corp?")
        
        # Should get a valid result from heuristics
        assert result.mode_used in ["data", "docs", "data+docs"]
        assert result.intent in ["company_status", "general", "renewals", "pipeline", "activities", "history"]
    
    def test_router_extracts_timeframe(self):
        """Test that router extracts timeframe from question."""
        from backend.agent.llm_router import route_question
        
        result = route_question("What happened in the last 30 days?")
        
        assert result.days == 30
    
    def test_router_respects_explicit_mode(self):
        """Test that explicit mode is respected."""
        from backend.agent.llm_router import route_question
        
        result = route_question("Tell me about Acme", mode="docs")
        
        assert result.mode_used == "docs"
    
    def test_router_result_schema(self):
        """Test that RouterResult has all expected fields."""
        from backend.agent.llm_router import route_question
        
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
        from backend.agent.orchestrator import answer_question
        
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
        from backend.agent.orchestrator import answer_question
        
        result = answer_question("Show me pipeline")
        
        assert "meta" in result
        assert "mode_used" in result["meta"]
        assert "latency_ms" in result["meta"]
    
    def test_agent_handles_company_query(self):
        """Test agent handles company-specific queries."""
        from backend.agent.orchestrator import answer_question
        
        result = answer_question("What's happening with Acme Manufacturing?")
        
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    def test_agent_handles_docs_query(self):
        """Test agent handles documentation queries."""
        from backend.agent.orchestrator import answer_question
        
        result = answer_question("How do I create an opportunity?", mode="docs")
        
        assert "answer" in result
        assert result["meta"]["mode_used"] == "docs"
