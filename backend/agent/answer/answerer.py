"""
Answer node LLM functions.

Chain creation and invocation for answer generation.
"""

import json
import logging
from functools import cache
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant. Answer questions using ONLY the provided context.

RULES:
- Use exact numbers/dates from context
- If data isn't available, say so briefly
- Lead with key answer, use bullets for details
- Keep it short and conversational

FORMAT: Currency $1,250,000 | Dates: March 31, 2026

EXAMPLES:
User: "What opportunities does Beta Tech have?"
Good: "Beta Tech has 3 open opportunities totaling $245,000.
- Largest: Enterprise renewal ($150,000, closes March 31)
- Champion: Sarah Chen (VP Engineering)
- Risk: Competitor evaluation in progress"
Bad: "They have several opportunities"
Bad: "Based on the provided data, I can confirm..."
Bad: "Great question! Let me look into that..."

User: "What's the renewal amount for Acme Corp?"
Good: "Renewal amount is not available in the current data."
Bad: "I don't have that information; amounts are tracked in the system but..."
"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}

{sql_results_section}

{rag_context_section}"""


@cache
def _get_answer_chain() -> Any:
    """Get or create the answer chain (cached singleton).

    Returns the LCEL chain directly so LangGraph's astream_events
    can capture on_chat_model_stream events for token streaming.
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=LONG_RESPONSE_MAX_TOKENS,
    )
    logger.debug("Created answer chain")
    return chain


def call_answer_chain(
    question: str,
    sql_results: dict[str, Any] | None = None,
    rag_context: str = "",
    conversation_history: str = "",
) -> str:
    """Call the answer chain and return the answer string."""
    result: str = _get_answer_chain().invoke({
        "question": question,
        "conversation_history_section": f"=== RECENT CONVERSATION ===\n{conversation_history}\n" if conversation_history else "",
        "sql_results_section": f"=== CRM DATA ===\n{json.dumps(sql_results, indent=2, default=str)}\n" if sql_results else "",
        "rag_context_section": f"=== CONTEXT NOTES ===\n{rag_context}\n" if rag_context else "",
    })
    return result


__all__ = ["call_answer_chain"]
