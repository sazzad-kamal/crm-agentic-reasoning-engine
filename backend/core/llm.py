"""
Shared LLM client infrastructure.

All LLM interactions should go through this module for consistency.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any

import anthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_PREFIXES = {"o1", "o3", "o4", "gpt-5"}

# Models
FAST_MODEL = "gpt-4o-mini"  # Fast, cost-efficient for text responses
REASONING_MODEL = "claude-sonnet-4-5-20241022"  # Better at structured/reasoning tasks
JUDGE_MODEL = "gpt-5.2"  # LLM judge for eval correctness
EMBEDDING_MODEL = "text-embedding-3-small"  # Embeddings for RAGAS

# Temperature settings
DETERMINISTIC_TEMPERATURE = 0.1  # Consistent, factual responses
CREATIVE_TEMPERATURE = 0.7  # Varied, exploratory suggestions

# Max token limits
LONG_RESPONSE_MAX_TOKENS = 1024  # Detailed answers
SHORT_RESPONSE_MAX_TOKENS = 150  # Brief suggestions


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    """Get Anthropic client (cached singleton)."""
    return anthropic.Anthropic()


@lru_cache
def get_openai_client():
    """Get raw OpenAI client (cached singleton)."""
    from openai import OpenAI

    return OpenAI()


def _create_chat_openai(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with proper token parameter handling."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Newer models use max_completion_tokens instead of max_tokens
    uses_completion_tokens = any(
        model.lower().startswith(p) for p in _MAX_COMPLETION_TOKENS_PREFIXES
    )

    if uses_completion_tokens:
        return ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            max_retries=3,
            request_timeout=60,  # type: ignore[call-arg]
            max_completion_tokens=max_tokens,
            streaming=streaming,
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        api_key=api_key,  # type: ignore[arg-type]
        max_retries=3,
        request_timeout=60,
        streaming=streaming,
    )


def create_chain(
    system_prompt: str,
    human_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    structured_output: type[BaseModel] | None = None,
    streaming: bool = True,
) -> Any:
    """Create an LCEL chain with the given prompts."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    llm = _create_chat_openai(model, temperature, max_tokens, streaming)

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)
    return prompt_template | llm | StrOutputParser()


def call_anthropic_structured(
    system: str,
    user_message: str,
    output_schema: type[BaseModel],
    model: str = REASONING_MODEL,
    max_tokens: int = 1024,
) -> BaseModel:
    """Call Anthropic with tool use for structured output."""
    response = get_anthropic_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "name": "output",
            "description": "Structured output",
            "input_schema": output_schema.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "output"},
    )
    return output_schema(**response.content[0].input)  # type: ignore[union-attr]


def call_openai_json(
    prompt: str,
    model: str = JUDGE_MODEL,
    max_tokens: int = 512,
    temperature: float = 0,
) -> dict:
    """Call OpenAI with JSON mode."""
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")  # type: ignore[no-any-return]


@lru_cache
def get_langchain_chat_openai(model: str = FAST_MODEL, temperature: float = 0) -> ChatOpenAI:
    """Get LangChain ChatOpenAI (cached singleton)."""
    return ChatOpenAI(model=model, temperature=temperature)


@lru_cache
def get_langchain_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """Get LangChain OpenAIEmbeddings (cached singleton)."""
    return OpenAIEmbeddings(model=model)


__all__ = [
    "FAST_MODEL",
    "REASONING_MODEL",
    "JUDGE_MODEL",
    "EMBEDDING_MODEL",
    "DETERMINISTIC_TEMPERATURE",
    "CREATIVE_TEMPERATURE",
    "LONG_RESPONSE_MAX_TOKENS",
    "SHORT_RESPONSE_MAX_TOKENS",
    "get_anthropic_client",
    "get_openai_client",
    "create_chain",
    "call_anthropic_structured",
    "call_openai_json",
    "get_langchain_chat_openai",
    "get_langchain_embeddings",
]
