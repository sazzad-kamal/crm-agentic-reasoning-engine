"""
Answer node LLM functions.

Chain creation and invocation for answer generation.
"""

import json
import logging
from functools import lru_cache
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite.
Your job is to answer questions using ONLY the provided CRM data context.

GROUNDING RULES:
- Use EXACT numbers and dates from context - never say "several", "some", "multiple", "recent"
- If specific data isn't in the context, say it's not available - don't over-explain

RESPONSE STYLE:
- Lead with the key answer in 1 sentence
- Use bullet points for supporting details
- Be conversational and natural, not robotic
- Keep it SHORT - no padding or filler

FORMATTING:
- Currency: $1,250,000
- Dates: March 31, 2026

EXAMPLES:
User asked: "What opportunities does Beta Tech have?"
Good: "Beta Tech has 3 open opportunities totaling $245,000.
- Largest: Enterprise renewal ($150,000, closes March 31)
- Champion: Sarah Chen (VP Engineering)
- Risk: Competitor evaluation in progress"
Bad (vague): "They have several opportunities"
Bad (robotic): "Based on the provided data, I can confirm that Beta Tech has opportunities..."
Bad (padded): "Great question! Let me look into that for you. So, basically..."

User asked: "What's the renewal amount for Acme Corp?"
Good: "Renewal amount is not available in the current data."
Bad (over-explaining): "I don't have that information; amounts are tracked in the system but..."
"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}

{sql_results_section}

{rag_context_section}"""


@lru_cache
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
