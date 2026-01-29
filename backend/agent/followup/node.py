"""Follow-up suggestion node for agent workflow."""

import logging

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def _is_answerable(suggestion: str) -> bool:
    """Check if a follow-up suggestion can produce SQL results."""
    try:
        plan = get_sql_plan(suggestion)
        rows, error = execute_sql(plan.sql, get_connection())
        return not error and bool(rows)
    except Exception:
        return False


def followup_node(state: AgentState) -> AgentState:
    """Generate follow-up suggestions based on current state."""
    logger.info("[Followup] Generating suggestions...")

    try:
        conn = get_connection()
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            answer=state.get("answer", ""),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
            sql_results=state.get("sql_results"),
            conn=conn,
        )

        # Filter empty and unanswerable suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]
            suggestions = [s for s in suggestions if _is_answerable(s)]

        logger.info(f"[Followup] {len(suggestions)} answerable suggestions")

        return {"follow_up_suggestions": suggestions}

    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {"follow_up_suggestions": []}


__all__ = ["followup_node"]
