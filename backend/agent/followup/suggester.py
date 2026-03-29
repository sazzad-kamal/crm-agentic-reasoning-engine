"""Follow-up suggestion LLM chain functions."""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from backend.agent.sql.schema import get_schema_sql
from backend.agent.followup.entity_context import get_entity_context
from backend.core.llm import SHORT_RESPONSE_MAX_TOKENS, create_openai_chain

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful CRM assistant that suggests follow-up questions.

IMPORTANT: Only suggest questions that can be answered using the available database tables and columns.
Do NOT suggest questions about data that doesn't exist in the schema.

RULES:
- Each question MUST be under 10 words. Be punchy and direct.
- Ask ONE thing per question — never combine multiple asks with "and".
- Do NOT repeat dollar amounts, dates, or details already shown in the answer.
- Reference entity names but omit numbers the user already sees.

GENERATE 3 QUESTIONS:
1. Drill into a specific entity or detail from the answer
2. Related aspect of the same entity (company/contact/deal)
3. Explore something NEW — different entity or broader view

EXAMPLES:
User asked: "What opportunities does Acme Corp have?"
Answer: "Acme Corp has 3 open deals: Enterprise Upgrade ($50K, Proposal), Cloud Migration ($30K, Negotiation), Support Renewal ($12K, Closed Won)."
1. "What's the Cloud Migration timeline?"
2. "Who are the key contacts at Acme?"
3. "Which deals are closing this quarter?"

User asked: "Why is the Beta Tech deal stuck?"
Answer: "The Beta Tech Platform deal has been in Negotiation for 45 days. Last activity was a meeting with Sarah Chen on Jan 5."
1. "What happened in the last meeting?"
2. "Who else do we know at Beta Tech?"
3. "What other deals are at risk?"
"""

_HUMAN_PROMPT = """User's question: {question}

{answer_section}{schema_section}{entity_context_section}{conversation_history_section}"""


class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up question suggestions."""

    suggestions: list[str] = Field(
        description="Exactly 3 follow-up questions the user might want to ask next",
        min_length=3,
        max_length=3,
    )


@cache
def _get_followup_chain() -> Any:
    """Get or create the followup chain (cached singleton)."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=SHORT_RESPONSE_MAX_TOKENS,
        structured_output=FollowUpSuggestions,
    )
    logger.debug("Created followup chain")
    return chain


def generate_follow_up_suggestions(
    question: str,
    answer: str = "",
    conversation_history: str = "",
    use_hardcoded_tree: bool = True,
    sql_results: dict[str, Any] | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[str]:
    """Generate 3 follow-up question suggestions."""
    # Try hardcoded tree first (fast, deterministic)
    if use_hardcoded_tree:
        from backend.agent.followup.tree import get_follow_ups

        follow_ups: list[str] = get_follow_ups(question)
        if follow_ups:
            logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
            return follow_ups

    # LLM fallback for contextual suggestions
    try:
        chain = _get_followup_chain()
        answer_section = f"Assistant's answer: {answer}\n\n" if answer else ""
        history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}" if conversation_history else ""

        # Schema context for data-aware suggestions
        schema_section = f"=== DATABASE SCHEMA ===\n{get_schema_sql()}\n\n"

        # Entity context from sql_results
        entity_context_section = ""
        if sql_results and conn:
            entity_ctx = get_entity_context(sql_results, conn)
            if entity_ctx:
                entity_context_section = f"=== ENTITY CONTEXT ===\n{entity_ctx}\n\n"

        result: FollowUpSuggestions = chain.invoke({
            "question": question,
            "answer_section": answer_section,
            "schema_section": schema_section,
            "entity_context_section": entity_context_section,
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions via LLM")
        return result.suggestions

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


__all__ = ["generate_follow_up_suggestions"]
