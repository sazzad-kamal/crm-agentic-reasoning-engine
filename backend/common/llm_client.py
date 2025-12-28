"""
Shared LLM client using LangChain for all RAG and agent components.

Features:
- Automatic LangSmith tracing (when LANGCHAIN_TRACING_V2=true)
- Built-in caching support
- Cost tracking
- Retry with exponential backoff
- Fallback models
- Structured output parsing
"""

import os
import time
import logging
import threading
from functools import lru_cache
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.callbacks import BaseCallbackHandler


# Configure module logger
logger = logging.getLogger(__name__)

# Enable caching if configured
_LLM_CACHE_ENABLED = os.environ.get("RAG_ENABLE_LLM_CACHE", "0") == "1"
if _LLM_CACHE_ENABLED:
    set_llm_cache(InMemoryCache())
    logger.info("LLM cache enabled (in-memory)")


# =============================================================================
# Cost Tracking
# =============================================================================

# OpenAI pricing per 1M tokens (as of late 2024)
MODEL_COSTS = {
    # GPT-4o models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    # GPT-3.5
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Default fallback
    "default": {"input": 2.50, "output": 10.00},
}


@dataclass
class CostTracker:
    """Track LLM usage costs across the session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0
    costs_by_model: dict = field(default_factory=dict)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add usage and return the cost for this call."""
        costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.calls += 1

        if model not in self.costs_by_model:
            self.costs_by_model[model] = {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
        self.costs_by_model[model]["input"] += input_tokens
        self.costs_by_model[model]["output"] += output_tokens
        self.costs_by_model[model]["cost"] += cost
        self.costs_by_model[model]["calls"] += 1

        return cost

    def get_summary(self) -> dict:
        """Get cost summary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_calls": self.calls,
            "by_model": self.costs_by_model,
        }


# Global cost tracker
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _cost_tracker
    _cost_tracker = CostTracker()


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for LLM API calls.

    Prevents hitting API rate limits by throttling requests.
    Thread-safe implementation.
    """
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    _request_tokens: float = field(default=0.0, init=False)
    _token_tokens: float = field(default=0.0, init=False)
    _last_refill: float = field(default_factory=time.time, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill request tokens
        self._request_tokens = min(
            self.requests_per_minute,
            self._request_tokens + (elapsed / 60) * self.requests_per_minute
        )

        # Refill API token tokens
        self._token_tokens = min(
            self.tokens_per_minute,
            self._token_tokens + (elapsed / 60) * self.tokens_per_minute
        )

    def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Acquire permission to make an LLM call.

        Args:
            estimated_tokens: Estimated tokens for this call

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self._lock:
            self._refill()

            wait_time = 0.0

            # Check request rate
            if self._request_tokens < 1:
                wait_time = max(wait_time, (1 - self._request_tokens) * 60 / self.requests_per_minute)

            # Check token rate
            if self._token_tokens < estimated_tokens:
                wait_time = max(wait_time, (estimated_tokens - self._token_tokens) * 60 / self.tokens_per_minute)

            if wait_time > 0:
                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s")
                return wait_time

            # Consume tokens
            self._request_tokens -= 1
            self._token_tokens -= estimated_tokens
            return 0.0

    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """Wait if rate limited, then proceed."""
        wait_time = self.acquire(estimated_tokens)
        if wait_time > 0:
            time.sleep(wait_time)


# Global rate limiter (configurable via env vars)
_RATE_LIMIT_RPM = int(os.environ.get("LLM_RATE_LIMIT_RPM", "60"))
_RATE_LIMIT_TPM = int(os.environ.get("LLM_RATE_LIMIT_TPM", "150000"))
_rate_limiter = RateLimiter(requests_per_minute=_RATE_LIMIT_RPM, tokens_per_minute=_RATE_LIMIT_TPM)


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    return _rate_limiter


class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler to track costs of LLM calls."""

    def on_llm_end(self, response, **kwargs):
        """Track costs when LLM call completes."""
        try:
            # Extract token usage from response
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage', {})
                model = response.llm_output.get('model_name', 'unknown')
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = _cost_tracker.add_usage(model, input_tokens, output_tokens)
                logger.debug(f"LLM cost: ${cost:.6f} ({input_tokens}+{output_tokens} tokens)")
        except Exception as e:
            logger.warning(f"Failed to track cost: {e}")


# =============================================================================
# LLM Client Protocol (for dependency injection / testing)
# =============================================================================

@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the LLM client interface."""

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Call the LLM and return the response text."""
        ...

    def call_with_metrics(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Call the LLM and return response with metrics."""
        ...


# =============================================================================
# Model Configuration
# =============================================================================

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_PREFIXES = {"o1", "o3", "o4", "gpt-5"}

# Fallback chain: if primary fails, try these in order
FALLBACK_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]


def _requires_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES)


@lru_cache(maxsize=16)
def _get_chat_model(
    model: str,
    temperature: float,
    max_tokens: int,
    with_fallbacks: bool = True,
) -> ChatOpenAI:
    """Get a cached ChatOpenAI instance for the given configuration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Build model kwargs based on model type
    model_kwargs = {}
    if _requires_max_completion_tokens(model):
        model_kwargs["max_completion_tokens"] = max_tokens
        # Reasoning models don't support custom temperature
        primary = ChatOpenAI(
            model=model,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
            model_kwargs=model_kwargs,
            callbacks=[CostTrackingCallback()],
        )
    else:
        primary = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
            callbacks=[CostTrackingCallback()],
        )

    # Add fallbacks for resilience
    if with_fallbacks and model not in FALLBACK_MODELS:
        fallbacks = [
            ChatOpenAI(
                model=fb_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                max_retries=2,
                request_timeout=30,
                callbacks=[CostTrackingCallback()],
            )
            for fb_model in FALLBACK_MODELS
        ]
        return primary.with_fallbacks(fallbacks)

    return primary


# =============================================================================
# Main API Functions
# =============================================================================

def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
    use_rate_limit: bool = True,
) -> str:
    """
    Call an OpenAI chat model and return the assistant's response.

    Uses LangChain's ChatOpenAI with automatic LangSmith tracing.

    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message to set context
        model: The model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        use_cache: Whether to use response caching (default: True)
        use_rate_limit: Whether to apply rate limiting (default: True)

    Returns:
        The assistant's message content as a string
    """
    # Apply rate limiting
    if use_rate_limit:
        estimated_tokens = len(prompt) // 4 + max_tokens  # Rough estimate
        _rate_limiter.wait_if_needed(estimated_tokens)

    chat = _get_chat_model(model, temperature, max_tokens)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    logger.debug(f"Calling LLM model={model}, prompt_len={len(prompt)}")

    response = chat.invoke(messages)
    result = response.content or ""

    logger.debug(f"LLM response received, len={len(result)}")
    return result


def call_llm_with_metrics(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_rate_limit: bool = True,
) -> dict:
    """
    Call an OpenAI chat model and return response with metrics.

    Returns a dict with:
        - response: The assistant's message content
        - latency_ms: Time taken for the call
        - prompt_tokens: Prompt token count
        - completion_tokens: Completion token count
        - total_tokens: Total token count
        - model: The model used
        - cost_usd: Estimated cost for this call
        - cached: Whether the response came from cache
    """
    # Apply rate limiting
    if use_rate_limit:
        estimated_tokens = len(prompt) // 4 + max_tokens
        _rate_limiter.wait_if_needed(estimated_tokens)

    chat = _get_chat_model(model, temperature, max_tokens)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    logger.debug(f"Calling LLM with metrics, model={model}, prompt_len={len(prompt)}")

    start_time = time.time()
    response = chat.invoke(messages)
    latency_ms = (time.time() - start_time) * 1000

    result = response.content or ""

    # Extract token usage from response metadata
    usage = response.response_metadata.get("token_usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # Calculate cost
    costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    cost_usd = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

    logger.info(
        f"LLM call completed: model={model}, latency={latency_ms:.0f}ms, "
        f"tokens={usage.get('total_tokens', 0)}, cost=${cost_usd:.6f}"
    )

    return {
        "response": result,
        "latency_ms": latency_ms,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", 0),
        "model": model,
        "cost_usd": cost_usd,
        "cached": False,  # LangChain handles caching internally
    }


def call_llm_safe(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    default: str = "",
) -> str:
    """
    Call LLM with graceful error handling - returns default on failure.

    Use this for non-critical LLM calls where failure should not break the pipeline.
    """
    try:
        result = call_llm(prompt, system_prompt, model, temperature, max_tokens)
        return result.strip() or default
    except Exception as e:
        logger.warning(f"LLM call failed, using default: {e}")
        return default


def clear_llm_cache() -> int:
    """Clear the LLM response cache. Returns 0 (LangChain manages cache internally)."""
    # LangChain's cache doesn't expose a clear method easily
    # Re-initialize the cache if needed
    if _LLM_CACHE_ENABLED:
        set_llm_cache(InMemoryCache())
        logger.info("LLM cache cleared")
    return 0


# =============================================================================
# Concrete LLM Client Implementation
# =============================================================================

class OpenAIClient:
    """
    Concrete LLM client implementation using LangChain's ChatOpenAI.

    Implements the LLMClient protocol for dependency injection.
    Provides automatic LangSmith tracing when configured.
    """

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Call the LLM and return the response text."""
        return call_llm(prompt, system_prompt, model, temperature, max_tokens)

    def call_with_metrics(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Call the LLM and return response with metrics."""
        return call_llm_with_metrics(prompt, system_prompt, model, temperature, max_tokens)

    def call_safe(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        default: str = "",
    ) -> str:
        """Call with graceful error handling - returns default on failure."""
        return call_llm_safe(prompt, system_prompt, model, temperature, max_tokens, default)


# Default client instance
_default_client: OpenAIClient | None = None


def get_llm_client() -> OpenAIClient:
    """Get the default LLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient()
    return _default_client


__all__ = [
    # Protocol
    "LLMClient",
    # Concrete implementation
    "OpenAIClient",
    "get_llm_client",
    # Module-level functions
    "call_llm",
    "call_llm_with_metrics",
    "call_llm_safe",
    "clear_llm_cache",
    # Cost tracking
    "CostTracker",
    "get_cost_tracker",
    "reset_cost_tracker",
    "MODEL_COSTS",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
]
