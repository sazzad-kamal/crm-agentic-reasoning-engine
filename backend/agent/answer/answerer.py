"""
Answer node LLM functions.

Chain creation and invocation for answer generation.
"""

import json
import logging
import re
from functools import cache
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant. Answer questions using ONLY the provided context.

RULES:
- Use exact numbers/dates from context
- If data isn't available, say so briefly
- Lead with key answer point
- Keep answers concise (2-4 sentences max, bullets when needed for details)

FORMAT: Currency $1,250,000 | Dates: March 31, 2026

SUGGESTED ACTION:
After answering, suggest ONE actionable next step based on the data.
- Use format: "Suggested action: [specific action]"
- Actions should be CRM-appropriate: schedule call, send email, create task, update stage
- Reference specific people/entities from the data when possible
- Only suggest if there's a clear action; skip for pure informational queries

EXAMPLES:
User: "What opportunities does Beta Tech have?"
Good: "Beta Tech has 3 open opportunities totaling $245,000.
- Largest: Enterprise renewal ($150,000, closes March 31)
- Champion: Sarah Chen (VP Engineering)
- Risk: Competitor evaluation in progress

Suggested action: Schedule a call with Sarah Chen to address the competitor evaluation."
Bad: "They have several opportunities"
Bad: "Based on the provided data, I can confirm..."

User: "What's the renewal amount for Acme Corp?"
Good: "Renewal amount is not available in the current data."
Bad: "I don't have that information; amounts are tracked in the system but..."
"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}

{sql_results_section}"""


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
    conversation_history: str = "",
) -> str:
    """Call the answer chain and return the answer string."""
    result: str = _get_answer_chain().invoke({
        "question": question,
        "conversation_history_section": f"=== RECENT CONVERSATION ===\n{conversation_history}\n" if conversation_history else "",
        "sql_results_section": f"=== CRM DATA ===\n{json.dumps(sql_results, indent=2, default=str)}\n" if sql_results else "",
    })
    return result


def extract_suggested_action(answer: str) -> tuple[str, str | None]:
    """Extract suggested action from answer text.

    Returns:
        Tuple of (clean_answer, action) where clean_answer has action removed.
    """
    match = re.search(r"\n*Suggested action:\s*(.+?)(?:\n|$)", answer, re.IGNORECASE)
    if match:
        action = match.group(1).strip()
        clean_answer = answer[: match.start()].rstrip()
        return clean_answer, action
    return answer, None


__all__ = ["call_answer_chain", "extract_suggested_action"]
