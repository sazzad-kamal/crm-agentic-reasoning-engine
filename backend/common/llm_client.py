"""
Shared LLM client helper for all RAG components.

Uses OpenAI API via the official Python client.
Requires OPENAI_API_KEY environment variable.

Features:
- Automatic retry with exponential backoff
- Proper logging
- Optional response caching
- Metrics collection
"""

import os
import time
import logging
import hashlib
from typing import Optional
from functools import lru_cache

from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


# Configure module logger
logger = logging.getLogger(__name__)

# Global client instance (lazy initialization)
_client: Optional[OpenAI] = None

# LLM response cache (enabled via environment or config)
_llm_cache: dict[str, str] = {}
_LLM_CACHE_ENABLED = os.environ.get("RAG_ENABLE_LLM_CACHE", "0") == "1"
_LLM_CACHE_MAX_SIZE = int(os.environ.get("RAG_LLM_CACHE_SIZE", "100"))

# Retry configuration
_MAX_RETRIES = int(os.environ.get("RAG_LLM_MAX_RETRIES", "3"))
_RETRY_MIN_WAIT = float(os.environ.get("RAG_LLM_RETRY_MIN_WAIT", "1.0"))
_RETRY_MAX_WAIT = float(os.environ.get("RAG_LLM_RETRY_MAX_WAIT", "10.0"))


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it before using the LLM client."
            )
        _client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    return _client


def _cache_key(prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> str:
    """Generate a cache key for an LLM request."""
    content = f"{model}:{temperature}:{system_prompt or ''}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _get_cached_response(cache_key: str) -> Optional[str]:
    """Get a cached LLM response if available."""
    if not _LLM_CACHE_ENABLED:
        return None
    response = _llm_cache.get(cache_key)
    if response:
        logger.debug(f"LLM cache hit for key {cache_key[:8]}...")
    return response


def _cache_response(cache_key: str, response: str) -> None:
    """Cache an LLM response."""
    if not _LLM_CACHE_ENABLED:
        return
    
    # Simple LRU-style eviction if cache is full
    if len(_llm_cache) >= _LLM_CACHE_MAX_SIZE:
        _llm_cache.pop(oldest := next(iter(_llm_cache)))
        logger.debug(f"LLM cache evicted key {oldest[:8]}...")
    
    _llm_cache[cache_key] = response
    logger.debug(f"LLM response cached with key {cache_key[:8]}...")


def clear_llm_cache() -> int:
    """Clear the LLM response cache. Returns number of entries cleared."""
    global _llm_cache
    count = len(_llm_cache)
    _llm_cache = {}
    logger.info(f"LLM cache cleared ({count} entries)")
    return count


# Retry decorator for transient errors
_retry_decorator = retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_retry_decorator
def _call_openai(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Make the actual OpenAI API call with retry logic."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-5-nano",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
) -> str:
    """
    Call an OpenAI chat model and return the assistant's response.
    
    Features automatic retry with exponential backoff for transient errors.
    
    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message to set context
        model: The model to use (default: gpt-5-nano)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        use_cache: Whether to use response caching (default: True)
        
    Returns:
        The assistant's message content as a string
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
        openai.APIError: If the API call fails after all retries
    """
    # Check cache first
    cache_key = _cache_key(prompt, system_prompt, model, temperature)
    if use_cache:
        cached = _get_cached_response(cache_key)
        if cached is not None:
            return cached
    
    client = _get_client()
    
    messages = [
        *([{"role": "system", "content": system_prompt}] if system_prompt else []),
        {"role": "user", "content": prompt},
    ]
    
    logger.debug(f"Calling LLM model={model}, prompt_len={len(prompt)}")
    
    try:
        response = _call_openai(client, messages, model, temperature, max_tokens)
        result = response.choices[0].message.content or ""
        
        logger.debug(f"LLM response received, len={len(result)}")
        
        # Cache the response
        if use_cache and temperature == 0.0:  # Only cache deterministic responses
            _cache_response(cache_key, result)
        
        return result
        
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded after {_MAX_RETRIES} retries: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection error after {_MAX_RETRIES} retries: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def call_llm_with_metrics(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-5-nano",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """
    Call an OpenAI chat model and return response with metrics.
    
    Features automatic retry with exponential backoff for transient errors.
    
    Returns a dict with:
        - response: The assistant's message content
        - latency_ms: Time taken for the call
        - prompt_tokens: Prompt token count
        - completion_tokens: Completion token count
        - total_tokens: Total token count
        - model: The model used
        - cached: Whether the response came from cache
    """
    # Check cache first
    cache_key = _cache_key(prompt, system_prompt, model, temperature)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        return {
            "response": cached,
            "latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": model,
            "cached": True,
        }
    
    client = _get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    logger.debug(f"Calling LLM with metrics, model={model}, prompt_len={len(prompt)}")
    
    start_time = time.time()
    
    try:
        response = _call_openai(client, messages, model, temperature, max_tokens)
        
        latency_ms = (time.time() - start_time) * 1000
        result = response.choices[0].message.content or ""
        
        logger.info(
            f"LLM call completed: model={model}, latency={latency_ms:.0f}ms, "
            f"tokens={response.usage.total_tokens if response.usage else 0}"
        )
        
        # Cache the response
        if temperature == 0.0:  # Only cache deterministic responses
            _cache_response(cache_key, result)
        
        return {
            "response": result,
            "latency_ms": latency_ms,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "model": model,
            "cached": False,
        }
        
    except (RateLimitError, APIConnectionError, APIError) as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"LLM call failed after {latency_ms:.0f}ms: {e}")
        raise


def call_llm_safe(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-5-nano",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    default: str = "",
) -> str:
    """
    Call LLM with graceful error handling - returns default on failure.
    
    Use this for non-critical LLM calls (e.g., query rewriting) where
    failure should not break the pipeline.
    
    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        default: Value to return on error (default: empty string)
        
    Returns:
        LLM response or default value on error
    """
    try:
        result = call_llm(prompt, system_prompt, model, temperature, max_tokens)
        return result.strip() or default
    except Exception as e:
        logger.warning(f"LLM call failed, using default: {e}")
        return default
