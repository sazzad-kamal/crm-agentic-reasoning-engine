"""Follow-up suggestion node for agent workflow."""

import logging

from backend.agent.sql.connection import get_connection
from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.agent.state import AgentState, format_conversation_for_prompt
from backend.agent.validate.contract import create_followup_validator

logger = logging.getLogger(__name__)

# Lazy-initialized contract validator
_followup_validator = None


def _get_followup_validator():
    """Get or create the followup validator (lazy init to avoid circular imports)."""
    global _followup_validator
    if _followup_validator is None:
        _followup_validator = create_followup_validator()
    return _followup_validator


def followup_node(state: AgentState) -> AgentState:
    """Generate follow-up suggestions based on current state."""
    logger.info("[Followup] Generating suggestions...")

    try:
        conn = get_connection()
        raw_suggestions = generate_follow_up_suggestions(
            question=state["question"],
            answer=state.get("answer", ""),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
            sql_results=state.get("sql_results"),
            conn=conn,
        )

        # Filter empty suggestions first
        if raw_suggestions:
            raw_suggestions = [s for s in raw_suggestions if s and s.strip()]

        # Apply contract validation: validate → repair → fallback
        validator = _get_followup_validator()
        contract_result = validator.enforce(raw_suggestions)

        if contract_result.was_repaired:
            logger.info(f"[Followup] Contract: repaired {len(contract_result.errors)} errors")
        elif contract_result.used_fallback:
            logger.info("[Followup] Contract: used fallback questions")

        suggestions = contract_result.output

        logger.info(f"[Followup] Generated {len(suggestions)} suggestions")

        return {"follow_up_suggestions": suggestions}

    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {"follow_up_suggestions": []}


__all__ = ["followup_node"]
