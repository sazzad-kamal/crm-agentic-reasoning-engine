"""
Shared pytest fixtures for backend tests.

This module provides common fixtures used across test files:
- test_client: FastAPI TestClient instance
- mock_llm_env: Sets MOCK_LLM=1 for LLM-dependent tests
- datastore: CRMDataStore instance
"""

import os
import pytest
from fastapi.testclient import TestClient


# =============================================================================
# Environment Setup
# =============================================================================

# Set mock mode by default for all tests
os.environ.setdefault("MOCK_LLM", "1")
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
    # Clear settings cache to ensure rate limit disabled setting is used
    from backend.core.config import get_settings
    get_settings.cache_clear()
    
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
def datastore():
    """
    Create a fresh CRMDataStore instance.
    
    This fixture provides access to the in-memory CRM data for testing
    data-related functionality.
    """
    from backend.agent.datastore import CRMDataStore
    return CRMDataStore()


@pytest.fixture
def sample_company_id():
    """Return a known company ID for testing."""
    return "ACME-MFG"


@pytest.fixture
def sample_company_name():
    """Return a known company name for testing."""
    return "Acme Manufacturing"


# =============================================================================
# LLM Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    return "Based on the CRM data, this is a mock response for testing purposes."


@pytest.fixture
def mock_llm_env(monkeypatch):
    """
    Ensure MOCK_LLM is set for tests that need it.
    
    This is useful for tests that import LLM-dependent code.
    """
    monkeypatch.setenv("MOCK_LLM", "1")
    yield
    

# =============================================================================
# Chat Request Fixtures
# =============================================================================

@pytest.fixture
def chat_request_company():
    """Sample chat request about a company."""
    return {"question": "What's going on with Acme Manufacturing?"}


@pytest.fixture
def chat_request_docs():
    """Sample chat request about documentation."""
    return {"question": "How do I import contacts?", "mode": "docs"}


@pytest.fixture
def chat_request_data():
    """Sample chat request for data mode."""
    return {"question": "Show me the pipeline", "mode": "data"}
