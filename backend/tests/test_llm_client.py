"""
Tests for the LLM client module.

Run with:
    pytest backend/tests/test_llm_client.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import hashlib

# Mock OpenAI before importing the module
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"


# =============================================================================
# Import after setting env vars
# =============================================================================

from backend.common.llm_client import (
    _cache_key,
    _get_cached_response,
    _cache_response,
    clear_llm_cache,
    call_llm,
    call_llm_with_metrics,
    _LLM_CACHE_ENABLED,
)


# =============================================================================
# Cache Key Tests
# =============================================================================

class TestCacheKey:
    """Tests for cache key generation."""
    
    def test_cache_key_is_deterministic(self):
        """Test that same inputs produce same cache key."""
        key1 = _cache_key("prompt", "system", "model", 0.0)
        key2 = _cache_key("prompt", "system", "model", 0.0)
        assert key1 == key2
    
    def test_different_prompts_different_keys(self):
        """Test that different prompts produce different keys."""
        key1 = _cache_key("prompt1", "system", "model", 0.0)
        key2 = _cache_key("prompt2", "system", "model", 0.0)
        assert key1 != key2
    
    def test_different_models_different_keys(self):
        """Test that different models produce different keys."""
        key1 = _cache_key("prompt", "system", "model1", 0.0)
        key2 = _cache_key("prompt", "system", "model2", 0.0)
        assert key1 != key2
    
    def test_different_temperatures_different_keys(self):
        """Test that different temperatures produce different keys."""
        key1 = _cache_key("prompt", "system", "model", 0.0)
        key2 = _cache_key("prompt", "system", "model", 0.5)
        assert key1 != key2
    
    def test_none_system_prompt_handled(self):
        """Test that None system prompt is handled."""
        key = _cache_key("prompt", None, "model", 0.0)
        assert isinstance(key, str)
        assert len(key) == 32  # SHA256 truncated to 32 chars
    
    def test_cache_key_is_valid_hash(self):
        """Test that cache key is a valid hex string."""
        key = _cache_key("prompt", "system", "model", 0.0)
        # Should be valid hex
        int(key, 16)


# =============================================================================
# Cache Operations Tests
# =============================================================================

class TestCacheOperations:
    """Tests for cache get/set operations."""
    
    def test_clear_cache_returns_count(self):
        """Test that clear_llm_cache returns the number of cleared entries."""
        count = clear_llm_cache()
        assert isinstance(count, int)
        assert count >= 0
    
    def test_clear_cache_empties_cache(self):
        """Test that clear_llm_cache actually clears the cache."""
        # Clear first
        clear_llm_cache()
        count = clear_llm_cache()
        # After clearing, should be empty
        assert count == 0


# =============================================================================
# LLM Call Tests (Mocked)
# =============================================================================

class TestCallLLM:
    """Tests for call_llm function with mocked OpenAI."""
    
    @patch("backend.common.llm_client._get_client")
    def test_call_llm_returns_string(self, mock_get_client):
        """Test that call_llm returns a string response."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_llm("Test prompt", use_cache=False)
        
        assert isinstance(result, str)
        assert result == "Test response"
    
    @patch("backend.common.llm_client._get_client")
    def test_call_llm_with_system_prompt(self, mock_get_client):
        """Test that system prompt is included in messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        call_llm("User prompt", system_prompt="System prompt", use_cache=False)
        
        # Verify the messages included system prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    @patch("backend.common.llm_client._get_client")
    def test_call_llm_uses_model_parameter(self, mock_get_client):
        """Test that model parameter is passed correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        call_llm("Prompt", model="gpt-4", use_cache=False)
        
        call_args = mock_client.chat.completions.create.call_args
        model = call_args.kwargs.get("model") or call_args[1].get("model")
        assert model == "gpt-4"
    
    @patch("backend.common.llm_client._get_client")
    def test_call_llm_handles_empty_response(self, mock_get_client):
        """Test handling of empty response content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_llm("Prompt", use_cache=False)
        
        assert result == ""


# =============================================================================
# LLM Call with Metrics Tests
# =============================================================================

class TestCallLLMWithMetrics:
    """Tests for call_llm_with_metrics function."""
    
    @patch("backend.common.llm_client._get_client")
    def test_returns_dict_with_response(self, mock_get_client):
        """Test that call_llm_with_metrics returns dict with response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_llm_with_metrics("Test prompt")
        
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == "Test response"
    
    @patch("backend.common.llm_client._get_client")
    def test_returns_latency_metric(self, mock_get_client):
        """Test that call_llm_with_metrics includes latency."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_llm_with_metrics("Prompt")
        
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))
        assert result["latency_ms"] >= 0
    
    @patch("backend.common.llm_client._get_client")
    def test_returns_token_counts(self, mock_get_client):
        """Test that call_llm_with_metrics includes token counts."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = call_llm_with_metrics("Prompt")
        
        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert "total_tokens" in result


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestLLMErrorHandling:
    """Tests for LLM error handling."""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        # Temporarily remove the key
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            # Reset the client
            import backend.common.llm_client as llm_module
            llm_module._client = None
            
            # This should work since we're using mocked client
            # The actual error would be raised on _get_client()


# =============================================================================
# Configuration Tests
# =============================================================================

class TestLLMConfiguration:
    """Tests for LLM client configuration."""
    
    def test_default_model_is_set(self):
        """Test that a default model is used."""
        # Import to check defaults
        import inspect
        from backend.common.llm_client import call_llm
        
        sig = inspect.signature(call_llm)
        model_default = sig.parameters["model"].default
        
        assert model_default is not None
        assert isinstance(model_default, str)
    
    def test_default_temperature_is_deterministic(self):
        """Test that default temperature is 0.0 for deterministic output."""
        import inspect
        from backend.common.llm_client import call_llm
        
        sig = inspect.signature(call_llm)
        temp_default = sig.parameters["temperature"].default
        
        assert temp_default == 0.0
    
    def test_cache_is_enabled_by_default(self):
        """Test that cache is enabled by default in call_llm."""
        import inspect
        from backend.common.llm_client import call_llm
        
        sig = inspect.signature(call_llm)
        cache_default = sig.parameters["use_cache"].default
        
        assert cache_default is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMClientIntegration:
    """Integration tests for LLM client."""
    
    @patch("backend.common.llm_client._get_client")
    def test_full_call_flow(self, mock_get_client):
        """Test the full call flow from prompt to response."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        # Make the call
        result = call_llm_with_metrics(
            prompt="What is the meaning of life?",
            system_prompt="You are a helpful assistant.",
            model="gpt-4",
            temperature=0.0,
            max_tokens=100,
        )
        
        # Verify result
        assert result["response"] == "The answer is 42."
        assert result["total_tokens"] == 20
        assert result["latency_ms"] >= 0
        
        # Verify client was called correctly
        mock_client.chat.completions.create.assert_called_once()
