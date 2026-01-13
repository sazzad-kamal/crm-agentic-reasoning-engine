"""
LangGraph routing node for agent workflow.

Uses SQL Sorcerer approach to generate SQL queries directly.
"""

import logging

from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.sql_planner import SQLPlan, get_sql_plan

logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    """
    Route node that generates SQL query from user question.

    Sets sql_plan in state for downstream nodes.
    """
    question = state["question"]
    logger.info(f"[Route] Processing: {question[:50]}...")

    try:
        sql_plan = get_sql_plan(
            question=question,
            conversation_history=format_history_for_prompt(state.get("messages", [])),
        )

        logger.info(f"[Route] SQL: {sql_plan.sql[:60]}..., needs_rag={sql_plan.needs_rag}")

        return {
            "sql_plan": sql_plan,
            "needs_rag": sql_plan.needs_rag,
        }

    except Exception as e:
        logger.error(f"[Route] Failed: {e}")

        return {
            "sql_plan": SQLPlan(sql="", needs_rag=False),
            "needs_rag": False,
            "error": f"Query planning failed: {e}",
        }
