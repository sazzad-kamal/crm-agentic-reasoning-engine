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

# Models (internal - not exported)
_OPENAI_MODEL = "gpt-5.2"  # Text responses, judging, RAGAS
_ANTHROPIC_MODEL = "claude-sonnet-4-5-20241022"  # SQL/code generation
_EMBEDDING_MODEL = "text-embedding-3-small"  # Embeddings for RAGAS

# Max token limits
LONG_RESPONSE_MAX_TOKENS = 1024  # Detailed answers
SHORT_RESPONSE_MAX_TOKENS = 150  # Brief suggestions


@lru_cache
def _get_anthropic_client() -> anthropic.Anthropic:
    """Get Anthropic client (cached singleton)."""
    return anthropic.Anthropic()


@lru_cache
def _get_openai_client():
    """Get raw OpenAI client (cached singleton)."""
    from openai import OpenAI

    return OpenAI()


def create_chain(
    system_prompt: str,
    human_prompt: str,
    max_tokens: int = 2000,
    structured_output: type[BaseModel] | None = None,
    streaming: bool = True,
) -> Any:
    """Create an LCEL chain with the given prompts."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model=_OPENAI_MODEL,
        api_key=api_key,  # type: ignore[arg-type]
        max_retries=3,
        request_timeout=60,  # type: ignore[call-arg]
        max_completion_tokens=max_tokens,
        streaming=streaming,
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)
    return prompt_template | llm | StrOutputParser()


def call_anthropic(
    system: str,
    user_message: str,
    output_schema: type[BaseModel],
    max_tokens: int = 1024,
) -> BaseModel:
    """Call Anthropic with tool use for structured output."""
    response = _get_anthropic_client().messages.create(
        model=_ANTHROPIC_MODEL,
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


def call_openai(prompt: str, system: str = "", max_tokens: int = 512) -> dict:
    """Call OpenAI with JSON mode."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_openai_client().chat.completions.create(
        model=_OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")  # type: ignore[no-any-return]


@lru_cache
def get_langchain_chat_openai() -> ChatOpenAI:
    """Get LangChain ChatOpenAI (cached singleton)."""
    return ChatOpenAI(model=_OPENAI_MODEL)


@lru_cache
def get_langchain_embeddings() -> OpenAIEmbeddings:
    """Get LangChain OpenAIEmbeddings (cached singleton)."""
    return OpenAIEmbeddings(model=_EMBEDDING_MODEL)


__all__ = [
    "LONG_RESPONSE_MAX_TOKENS",
    "SHORT_RESPONSE_MAX_TOKENS",
    "create_chain",
    "call_anthropic",
    "call_openai",
    "get_langchain_chat_openai",
    "get_langchain_embeddings",
]
