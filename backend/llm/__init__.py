"""
Shared LLM infrastructure.

Used by both agent/ and eval/ for ChatOpenAI instantiation.
"""

from backend.llm.client import (
    get_chat_model,
    create_chain,
    call_llm,
)

__all__ = [
    "get_chat_model",
    "create_chain",
    "call_llm",
]
