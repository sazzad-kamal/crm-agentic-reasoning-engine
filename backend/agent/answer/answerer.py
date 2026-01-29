"""Answer node LLM chain functions."""

import json
import logging
from datetime import datetime
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant. Answer questions using ONLY the provided context.
Today: {today}

RULES:
- Use exact numbers/dates from context
- Only say "data not available" if the answer cannot be found in the CRM DATA
- The CRM DATA contains SQL query results that directly answer the question - interpret and present them
- Use ALL provided data to formulate a complete answer
- Lead with key answer point
- Keep answers concise (2-4 sentences max, bullets when needed for details)

FORMAT: Currency $1,250,000 | Dates: March 31, 2026

EXAMPLES:
User: "What opportunities does Beta Tech have?"
Good: "Beta Tech has 3 open opportunities totaling $245,000.
- Largest: Enterprise renewal ($150,000, closes March 31)
- Champion: Sarah Chen (VP Engineering)
- Risk: Competitor evaluation in progress"
Bad: "They have several opportunities"
Bad: "Based on the provided data, I can confirm..."

User: "What's the renewal amount for Acme Corp?"
Good: "Renewal amount is not available in the current data."
Bad: "I don't have that information; amounts are tracked in the system but..."
"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}

{sql_results_section}"""


def _get_answer_chain() -> Any:
    """Get answer chain with current date in system prompt."""
    today = datetime.now().strftime("%Y-%m-%d")
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT.format(today=today),
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


__all__ = ["call_answer_chain"]
