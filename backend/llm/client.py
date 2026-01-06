"""
Shared LLM client infrastructure.

Provides ChatOpenAI factory and chain creation used by both agent/ and eval/.
"""

import os
import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


logger = logging.getLogger(__name__)


# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_PREFIXES = {"o1", "o3", "o4", "gpt-5"}


def _requires_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES)


@lru_cache(maxsize=16)
def get_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    streaming: bool = False,
) -> ChatOpenAI:
    """Get a cached ChatOpenAI instance.

    Args:
        model: The model name (e.g., "gpt-4o", "gpt-4o-mini")
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        streaming: Whether to enable streaming

    Returns:
        A configured ChatOpenAI instance
    """
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
            streaming=streaming,
        )
    else:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
            streaming=streaming,
        )


def create_chain(
    prompt_template: Any,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    structured_output: type[BaseModel] | None = None,
    streaming: bool = True,
) -> Any:
    """Create an LCEL chain with the given prompt template.

    Args:
        prompt_template: A LangChain prompt template
        model: The model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        structured_output: Optional Pydantic model for structured output
        streaming: Whether to enable streaming (default True)

    Returns:
        An LCEL chain (prompt | llm | parser)
    """
    # Don't use cached model for chains - create fresh for proper streaming
    api_key = os.environ.get("OPENAI_API_KEY")

    if _requires_max_completion_tokens(model):
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            max_retries=3,
            request_timeout=60,
            max_completion_tokens=max_tokens,
            streaming=streaming,
        )
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            streaming=streaming,
        )

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)

    return prompt_template | llm | StrOutputParser()


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call an OpenAI chat model and return the response.

    Simple message-based interface for eval and other non-chain use cases.

    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        The assistant's message content as a string
    """
    chat = get_chat_model(model, temperature, max_tokens, streaming=False)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    logger.debug(f"Calling LLM model={model}, prompt_len={len(prompt)}")

    response = chat.invoke(messages)
    result = response.content or ""

    logger.debug(f"LLM response received, len={len(result)}")
    return result


__all__ = [
    "get_chat_model",
    "create_chain",
    "call_llm",
]
