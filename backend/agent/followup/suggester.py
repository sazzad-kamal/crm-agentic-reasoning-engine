"""
Followup node LLM functions.

Chain creation and invocation for follow-up question generation.
Uses hardcoded tree first, falls back to LLM for contextual suggestions.
"""

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from backend.core.llm import SHORT_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful CRM assistant that suggests follow-up questions.

GENERATE 3 QUESTIONS:
1. First question: Drill deeper into the topic the user asked about
2. Second question: Another angle on the same topic
3. Third question: Let user explore something NEW - different company or general CRM data question

RULES:
- Keep questions SHORT and actionable
- Questions should be natural follow-ups to what was asked
- Mix specific follow-ups with broader exploration options"""

_HUMAN_PROMPT = """Suggest 3 follow-up questions for the user.

User's question: {question}

{conversation_history_section}"""


class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up question suggestions."""

    suggestions: list[str] = Field(
        description="Exactly 3 follow-up questions the user might want to ask next",
        min_length=3,
        max_length=3,
    )


@lru_cache
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
    conversation_history: str = "",
    use_hardcoded_tree: bool = True,
) -> list[str]:
    """Generate follow-up question suggestions.

    Args:
        question: The user's original question
        conversation_history: Previous conversation context
        use_hardcoded_tree: Whether to try hardcoded tree first

    Returns:
        List of exactly 3 follow-up question suggestions
    """
    # Try hardcoded tree first (fast, deterministic)
    if use_hardcoded_tree:
        from backend.agent.followup.tree import get_follow_ups

        follow_ups = get_follow_ups(question)
        if follow_ups:
            logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
            return follow_ups

    # LLM fallback for contextual suggestions
    try:
        chain = _get_followup_chain()
        history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}" if conversation_history else ""

        result: FollowUpSuggestions = chain.invoke({
            "question": question,
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions via LLM")
        return result.suggestions

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


__all__ = ["generate_follow_up_suggestions"]
