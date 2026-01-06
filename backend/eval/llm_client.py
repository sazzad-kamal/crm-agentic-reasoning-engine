"""
Simple LLM client for evaluation harnesses.

Provides a minimal call_llm function using LangChain's ChatOpenAI.
"""

import os
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


logger = logging.getLogger(__name__)


# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_PREFIXES = {"o1", "o3", "o4", "gpt-5"}


def _requires_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES)


@lru_cache(maxsize=16)
def _get_chat_model(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    """Get a cached ChatOpenAI instance for the given configuration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if _requires_max_completion_tokens(model):
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
            max_completion_tokens=max_tokens,
        )
    else:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
        )


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call an OpenAI chat model and return the assistant's response.

    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message to set context
        model: The model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        The assistant's message content as a string
    """
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


__all__ = ["call_llm"]
