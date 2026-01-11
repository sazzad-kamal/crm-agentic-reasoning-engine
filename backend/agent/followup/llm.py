"""
Followup node LLM functions.

Chain creation and invocation for follow-up question generation.
Loads prompt from prompt.txt for clean separation.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.core.config import get_config
from backend.llm.client import create_chain
from backend.utils.prompt import load_prompt

logger = logging.getLogger(__name__)

FOLLOW_UP_PROMPT_TEMPLATE = load_prompt(Path(__file__).parent / "prompt.txt")


class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up question suggestions."""

    suggestions: list[str] = Field(
        description="List of 3 follow-up questions the user might want to ask next",
        min_length=1,
        max_length=5,
    )


# Cached chain (lazy initialization)
_followup_chain: Any = None


def _get_followup_chain() -> Any:
    """Get or create the followup chain."""
    global _followup_chain
    if _followup_chain is None:
        config = get_config()
        _followup_chain = create_chain(
            FOLLOW_UP_PROMPT_TEMPLATE,
            model=config.llm_model,  # Use general LLM (not Codex)
            temperature=0.7,
            max_tokens=150,
            structured_output=FollowUpSuggestions,
        )
        logger.debug("Created followup chain")
    return _followup_chain


def generate_follow_up_suggestions(
    question: str,
    company_id: str | None = None,
    company_name: str | None = None,
    conversation_history: str = "",
    available_data: dict | None = None,
    use_hardcoded_tree: bool = True,
) -> list[str]:
    """Generate follow-up question suggestions.

    Args:
        question: The user's original question
        company_id: Optional company ID for context
        company_name: Optional company name for context
        conversation_history: Previous conversation context
        available_data: Dict of available data types and counts
        use_hardcoded_tree: Whether to try hardcoded tree first

    Returns:
        List of up to 3 follow-up question suggestions
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return []

    # Try hardcoded tree first (100% reliable for demos)
    if use_hardcoded_tree:
        from backend.agent.followup.tree import get_follow_ups
        follow_ups = get_follow_ups(question)
        if follow_ups:  # Empty list means terminal/leaf node
            logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
            return follow_ups[:3]

    # LLM generation
    try:
        chain = _get_followup_chain()
        data_context = _format_available_data(available_data, company_name)
        history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}" if conversation_history else ""

        result: FollowUpSuggestions = chain.invoke({
            "question": question,
            "company": company_name or company_id or "None specified",
            "available_data": data_context,
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions")
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

    if data.get("pipeline_summary"):
        lines.append("- Pipeline: Overall pipeline summary available")

    return "\n".join(lines) if lines else "No specific data available. Suggest general CRM questions."


__all__ = [
    "FollowUpSuggestions",
    "generate_follow_up_suggestions",
]
