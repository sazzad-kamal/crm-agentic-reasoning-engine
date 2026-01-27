"""
Shared LLM client infrastructure.

All LLM interactions should go through this module for consistency.
"""

import logging
from functools import cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Models (internal - not exported)
_OPENAI_MODEL = "gpt-5.2"  # Text responses, judging, RAGAS
_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"  # SQL/code generation
_EMBEDDING_MODEL = "text-embedding-3-small"  # Embeddings for RAGAS

# Shared LLM settings
_MAX_RETRIES = 3
_TIMEOUT = 60
_SYSTEM_ROLE = "system"
_HUMAN_ROLE = "human"

# Max token limits
LONG_RESPONSE_MAX_TOKENS = 1024  # Detailed answers
SHORT_RESPONSE_MAX_TOKENS = 150  # Brief suggestions

def create_openai_chain(
    system_prompt: str,
    human_prompt: str,
    max_tokens: int = 2000,
    structured_output: type[BaseModel] | None = None,
    streaming: bool = True,
    model: str | None = None,
) -> Any:
    """Create an LCEL chain with OpenAI."""
    llm = ChatOpenAI(
        model=model or _OPENAI_MODEL,
        max_retries=_MAX_RETRIES,
        request_timeout=_TIMEOUT,  # type: ignore[call-arg]
        max_completion_tokens=max_tokens,
        streaming=streaming,
    )
    prompt_template = ChatPromptTemplate.from_messages([
        (_SYSTEM_ROLE, system_prompt),
        (_HUMAN_ROLE, human_prompt),
    ])

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)
    return prompt_template | llm | StrOutputParser()


def create_anthropic_chain(
    system_prompt: str,
    human_prompt: str,
    structured_output: type[BaseModel],
) -> Any:
    """Create an LCEL chain with Anthropic for structured output."""
    llm = ChatAnthropic(
        model=_ANTHROPIC_MODEL,
        max_retries=_MAX_RETRIES,
        timeout=_TIMEOUT,
    )  # type: ignore[call-arg]
    prompt_template = ChatPromptTemplate.from_messages([
        (_SYSTEM_ROLE, system_prompt),
        (_HUMAN_ROLE, human_prompt),
    ])
    return prompt_template | llm.with_structured_output(structured_output)


@cache
def get_langchain_chat_openai() -> ChatOpenAI:
    """Get LangChain ChatOpenAI (cached singleton)."""
    return ChatOpenAI(model=_OPENAI_MODEL)


@cache
def get_langchain_embeddings() -> OpenAIEmbeddings:
    """Get LangChain OpenAIEmbeddings (cached singleton)."""
    return OpenAIEmbeddings(model=_EMBEDDING_MODEL)


__all__ = [
    "LONG_RESPONSE_MAX_TOKENS",
    "SHORT_RESPONSE_MAX_TOKENS",
    "create_openai_chain",
    "create_anthropic_chain",
    "get_langchain_chat_openai",
    "get_langchain_embeddings",
]
