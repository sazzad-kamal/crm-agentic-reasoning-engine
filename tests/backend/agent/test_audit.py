"""
Tests for backend.agent.audit module.

Tests the audit logging functionality.
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.agent.output.audit import (
    AgentAuditEntry,
    AgentAuditLogger,
    get_audit_logger,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def audit_logger(temp_log_file):
    """Create an audit logger with a temporary log file."""
    return AgentAuditLogger(log_file=temp_log_file)


# =============================================================================
# AgentAuditEntry Tests
# =============================================================================

class TestAgentAuditEntry:
    """Tests for AgentAuditEntry dataclass."""

    def test_entry_creation(self):
        """Test creating an audit entry."""
        entry = AgentAuditEntry(
            timestamp="2024-01-15T10:30:00Z",
            question="What is going on with Acme?",
            mode_used="data",
            company_id="ACME-MFG",
            latency_ms=150,
            source_count=3,
        )

        assert entry.question == "What is going on with Acme?"
        assert entry.mode_used == "data"
        assert entry.company_id == "ACME-MFG"
        assert entry.latency_ms == 150

    def test_entry_optional_fields(self):
        """Test entry with optional fields."""
        entry = AgentAuditEntry(
            timestamp="2024-01-15T10:30:00Z",
            question="Test",
            mode_used="docs",
        )

        assert entry.company_id is None
        assert entry.user_id is None
        assert entry.session_id is None
        assert entry.error is None

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = AgentAuditEntry(
            timestamp="2024-01-15T10:30:00Z",
            question="Test question",
            mode_used="data",
            company_id="ACME-MFG",
            latency_ms=100,
        )

        data = entry.to_dict()

        assert isinstance(data, dict)
        assert data["question"] == "Test question"
        assert data["mode_used"] == "data"
        assert "company_id" in data

    def test_entry_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        entry = AgentAuditEntry(
            timestamp="2024-01-15T10:30:00Z",
            question="Test",
            mode_used="docs",
        )

        data = entry.to_dict()

        # None values should be excluded
        assert "company_id" not in data
        assert "user_id" not in data
        assert "session_id" not in data
        assert "error" not in data

    def test_entry_with_error(self):
        """Test entry with error field."""
        entry = AgentAuditEntry(
            timestamp="2024-01-15T10:30:00Z",
            question="Test",
            mode_used="error",
            error="LLM service unavailable",
        )

        assert entry.error == "LLM service unavailable"
        data = entry.to_dict()
        assert data["error"] == "LLM service unavailable"


# =============================================================================
# AgentAuditLogger Tests
# =============================================================================

class TestAgentAuditLogger:
    """Tests for AgentAuditLogger class."""

    def test_logger_creates_directory(self, temp_log_file):
        """Test that logger creates parent directory if needed."""
        # Create path in non-existent subdirectory
        nested_path = temp_log_file.parent / "subdir" / "audit.jsonl"

        try:
            logger = AgentAuditLogger(log_file=nested_path)
            assert nested_path.parent.exists()
        finally:
            # Cleanup
            if nested_path.parent.exists():
                if nested_path.exists():
                    nested_path.unlink()
                nested_path.parent.rmdir()

    def test_log_query_basic(self, audit_logger, temp_log_file):
        """Test basic query logging."""
        audit_logger.log_query(
            question="What is going on with Acme?",
            mode_used="data",
            company_id="ACME-MFG",
            latency_ms=150,
            source_count=3,
        )

        # Read the log file
        with open(temp_log_file) as f:
            line = f.readline()
            data = json.loads(line)

        assert data["question"] == "What is going on with Acme?"
        assert data["mode_used"] == "data"
        assert data["company_id"] == "ACME-MFG"
        assert data["latency_ms"] == 150

    def test_log_query_truncates_long_questions(self, audit_logger, temp_log_file):
        """Test that long questions are truncated."""
        long_question = "x" * 600  # Over 500 char limit

        audit_logger.log_query(
            question=long_question,
            mode_used="data",
        )

        with open(temp_log_file) as f:
            data = json.loads(f.readline())

        assert len(data["question"]) == 500

    def test_log_query_with_all_fields(self, audit_logger, temp_log_file):
        """Test logging with all optional fields."""
        audit_logger.log_query(
            question="Test question",
            mode_used="data",
            company_id="ACME-MFG",
            latency_ms=200,
            source_count=5,
            user_id="user-123",
            session_id="sess-456",
        )

        with open(temp_log_file) as f:
            data = json.loads(f.readline())

        assert data["user_id"] == "user-123"
        assert data["session_id"] == "sess-456"

    def test_log_query_with_error(self, audit_logger, temp_log_file):
        """Test logging failed queries."""
        audit_logger.log_query(
            question="Test question",
            mode_used="error",
            error="Service unavailable",
        )

        with open(temp_log_file) as f:
            data = json.loads(f.readline())

        assert data["mode_used"] == "error"
        assert data["error"] == "Service unavailable"

    def test_log_query_includes_timestamp(self, audit_logger, temp_log_file):
        """Test that entries include ISO timestamp."""
        audit_logger.log_query(
            question="Test",
            mode_used="docs",
        )

        with open(temp_log_file) as f:
            data = json.loads(f.readline())

        assert "timestamp" in data
        # Should be parseable as ISO format
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    def test_multiple_log_entries(self, audit_logger, temp_log_file):
        """Test logging multiple entries."""
        for i in range(3):
            audit_logger.log_query(
                question=f"Question {i}",
                mode_used="data",
                latency_ms=100 + i * 50,
            )

        with open(temp_log_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "question" in data

# =============================================================================
# Module Function Tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_audit_logger_returns_instance(self):
        """Test that get_audit_logger returns a logger instance."""
        logger = get_audit_logger()
        assert isinstance(logger, AgentAuditLogger)

    def test_get_audit_logger_singleton(self):
        """Test that get_audit_logger returns same instance."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2
