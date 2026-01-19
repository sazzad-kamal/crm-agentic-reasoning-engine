"""
Answer node LLM functions.

Chain creation and invocation for answer generation.
"""

import json
import logging
import time
from functools import lru_cache
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite.
Your job is to answer questions using ONLY the provided CRM data context.

GROUNDING RULES:
- Use EXACT numbers and dates from context - never say "several", "some", "multiple", "recent"
- When asked "how many", extract the explicit count from the data
- If specific data isn't in the context, just say it's not available - don't over-explain

EXAMPLES:
Good: "Beta Tech has 3 open opportunities totaling $245,000"
Good: "Last activity: call on December 15, 2024 with John Smith"
Good: "Renewal amount is not available in the current data."
Bad: "They have several opportunities" (vague)
Bad: "Amount: I don't have that information; amounts are tracked in..." (over-explaining)

RESPONSE STYLE:
- Lead with the key answer in 1 sentence
- Use bullet points for supporting details
- Be conversational and natural, not robotic
- Keep it SHORT - no padding or filler

FORMATTING:
- Currency: $1,250,000
- Dates: March 31, 2026
- If no data found, acknowledge briefly and offer to help differently"""

_HUMAN_PROMPT = """Answer the user's question using ONLY the provided data below.

User's question: {question}

{conversation_history_section}

=== CRM DATA (SQL Query Results) ===
{sql_results}

{account_context_section}

Please provide a helpful, grounded response following the rules in your system prompt.
If the data is empty or doesn't contain the answer, acknowledge this briefly."""


@lru_cache
def get_answer_chain() -> Any:
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


def _format_sql_results(sql_results: dict[str, Any] | None) -> str:
    """Format SQL results as JSON string for the LLM prompt."""
    if not sql_results:
        return "(No data retrieved)"

    # Convert to pretty JSON for readability
    try:
        return json.dumps(sql_results, indent=2, default=str)
    except Exception:
        return str(sql_results)


def build_answer_input(
    question: str,
    sql_results: dict[str, Any] | None = None,
    account_context: str = "",
    conversation_history: str = "",
) -> dict[str, str]:
    """Build input dict for the answer chain."""
    # Format conversation history section
    conversation_history_section = ""
    if conversation_history:
        conversation_history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}\n"

    # Format account context section
    account_context_section = ""
    if account_context:
        account_context_section = f"=== ACCOUNT CONTEXT (RAG) ===\n{account_context}\n"

    return {
        "question": question,
        "conversation_history_section": conversation_history_section,
        "sql_results": _format_sql_results(sql_results),
        "account_context_section": account_context_section,
    }


def call_answer_chain(
    question: str,
    sql_results: dict[str, Any] | None = None,
    account_context: str = "",
    conversation_history: str = "",
) -> tuple[str, int]:
    """
    Call the answer chain and return (answer, latency_ms).

    Args:
        question: The user's question
        sql_results: Dict of SQL query results keyed by purpose
        account_context: RAG context from account notes/descriptions
        conversation_history: Formatted conversation history

    Returns:
        Tuple of (answer string, latency in ms)
    """
    chain = get_answer_chain()
    start_time = time.time()

    chain_input = build_answer_input(
        question=question,
        sql_results=sql_results,
        account_context=account_context,
        conversation_history=conversation_history,
    )

    answer = chain.invoke(chain_input)

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Answer chain completed in {latency_ms}ms")
    return answer, latency_ms


__all__ = [
    "get_answer_chain",
    "build_answer_input",
    "call_answer_chain",
]
