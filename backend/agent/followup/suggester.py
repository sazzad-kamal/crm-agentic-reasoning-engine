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

_SYSTEM_PROMPT = "You are a helpful CRM assistant. Generate 3 follow-up question suggestions."

_HUMAN_PROMPT = """Suggest 3 follow-up questions for the user.

User's question: {question}
Current company: {company}

=== AVAILABLE DATA FOR THIS COMPANY ===
{available_data}

{conversation_history_section}

GENERATE 3 QUESTIONS:
1. First question: Drill deeper into current company's available data (use company name)
2. Second question: Another angle on current company's data (use company name)
3. Third question: Let user explore something NEW - different company or general CRM data question

RULES:
- Questions 1-2: ONLY ask about data types listed as available above
- Question 3: Can be general (renewals, pipeline summary) or about CRM features
- Always use company name, not "they" or "their"
- Keep questions SHORT"""


class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up question suggestions."""

    suggestions: list[str] = Field(
        description="List of 3 follow-up questions the user might want to ask next",
        min_length=1,
        max_length=5,
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
    company_name: str | None = None,
    conversation_history: str = "",
    available_data: dict | None = None,
    use_hardcoded_tree: bool = True,
) -> list[str]:
    """Generate follow-up question suggestions.

    Args:
        question: The user's original question
        company_name: Optional company name for context
        conversation_history: Previous conversation context
        available_data: Dict of available data types and counts
        use_hardcoded_tree: Whether to try hardcoded tree first

    Returns:
        List of up to 3 follow-up question suggestions
    """
    # Try hardcoded tree first (fast, deterministic)
    if use_hardcoded_tree:
        from backend.agent.followup.tree import get_follow_ups

        follow_ups = get_follow_ups(question)
        if follow_ups:
            logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
            return follow_ups[:3]

    # LLM fallback for contextual suggestions
    try:
        chain = _get_followup_chain()
        data_context = _format_available_data(available_data, company_name)
        history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}" if conversation_history else ""

        result: FollowUpSuggestions = chain.invoke({
            "question": question,
            "company": company_name or "None specified",
            "available_data": data_context,
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions via LLM")
        return result.suggestions[:3]

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


def _format_available_data(data: dict | None, company_name: str | None) -> str:
    """Format available data for the prompt context."""
    if not data:
        return "No specific data available. Suggest general CRM questions."

    lines = []
    company_label = company_name or "this company"

    mappings = [
        ("contacts", f"Contacts: {{}} contacts for {company_label}"),
        ("activities", "Activities: {} recent activities"),
        ("opportunities", "Opportunities: {} open opportunities"),
        ("history", "History: {} timeline entries"),
        ("renewals", "Renewals: {} upcoming renewals"),
    ]

    for key, template in mappings:
        if data.get(key, 0) > 0:
            lines.append(f"- {template.format(data[key])}")

    return "\n".join(lines) if lines else "No specific data available. Suggest general CRM questions."


__all__ = ["generate_follow_up_suggestions"]
