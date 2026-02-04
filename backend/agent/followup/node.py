"""Follow-up suggestion node for agent workflow."""

import logging

from backend.act_fetch import DEMO_FOLLOWUPS, DEMO_MODE, DEMO_STARTERS
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def followup_node(state: AgentState) -> AgentState:
    """Generate follow-up suggestions based on current state."""
    logger.info("[Followup] Generating suggestions...")

    # Demo mode: return contextual follow-ups based on current question
    if DEMO_MODE:
        question = state.get("question", "")
        followups = DEMO_FOLLOWUPS.get(question, list(DEMO_STARTERS))
        logger.info(f"[Followup] Demo mode - returning {len(followups)} contextual follow-ups for '{question}'")
        return {"follow_up_suggestions": followups}

    try:
        conn = get_connection()
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            answer=state.get("answer", ""),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
            sql_results=state.get("sql_results"),
            conn=conn,
        )

        # Filter empty suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]

        logger.info(f"[Followup] Generated {len(suggestions)} suggestions")

        return {"follow_up_suggestions": suggestions}

    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {"follow_up_suggestions": []}


__all__ = ["followup_node"]
