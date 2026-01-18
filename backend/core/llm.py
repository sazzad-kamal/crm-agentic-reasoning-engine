"""
Shared LLM client infrastructure.

Provides ChatOpenAI factory and chain creation used by both agent/ and eval/.
"""

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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


@lru_cache(maxsize=16)
def get_chat_model(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    streaming: bool = False,
) -> ChatOpenAI:
    """Get a cached ChatOpenAI instance."""
    return _create_chat_openai(model, temperature, max_tokens, streaming)


def create_chain(
    prompt_template: Any,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    structured_output: type[BaseModel] | None = None,
    streaming: bool = True,
) -> Any:
    """Create an LCEL chain with the given prompt template."""
    # Don't use cached model for chains - create fresh for proper streaming
    llm = _create_chat_openai(model, temperature, max_tokens, streaming)

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

    messages: list = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    logger.debug(f"Calling LLM model={model}, prompt_len={len(prompt)}")

    response = chat.invoke(messages)
    result = response.content or ""

    logger.debug(f"LLM response received, len={len(str(result))}")
    return str(result)


def load_prompt(path: Path) -> ChatPromptTemplate:
    """Load and parse a prompt.txt file into a ChatPromptTemplate.

    The file should contain [system] and [human] section markers.

    Args:
        path: Path to the prompt.txt file

    Returns:
        ChatPromptTemplate with system and human messages
    """
    text = path.read_text(encoding="utf-8")

    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.strip() == "[system]":
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "system"
            current_lines = []
        elif line.strip() == "[human]":
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "human"
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", sections.get("system", "")),
            ("human", sections.get("human", "")),
        ]
    )


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If JSON cannot be parsed
    """
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        if match := re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        raise ValueError(f"Failed to parse JSON from response: {text[:200]}") from None


__all__ = [
    "FAST_MODEL",
    "REASONING_MODEL",
    "JUDGE_MODEL",
    "EMBEDDING_MODEL",
    "DETERMINISTIC_TEMPERATURE",
    "CREATIVE_TEMPERATURE",
    "LONG_RESPONSE_MAX_TOKENS",
    "SHORT_RESPONSE_MAX_TOKENS",
    "get_chat_model",
    "create_chain",
    "call_llm",
    "load_prompt",
    "parse_json_response",
]
