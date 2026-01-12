"""
LangGraph routing node for agent workflow.

Uses slot-based query planning to generate SQL queries.
"""

import logging

from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.query_planner import (
    SlotPlan,
    detect_owner_from_starter,
    get_slot_plan,
)

logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    """
    Route node that generates SQL query plan from user question.

    Sets query_plan in state for downstream nodes.
    """
    question = state["question"]
    logger.info(f"[Route] Processing: {question[:50]}...")

    conversation_history = format_history_for_prompt(state.get("messages", []))

    # Detect owner from starter patterns
    owner = detect_owner_from_starter(question)

    try:
        # Get slot-based query plan from LLM
        slot_plan = get_slot_plan(
            question=question,
            conversation_history=conversation_history,
            owner=owner,
        )

        logger.info(f"[Route] Result: {len(slot_plan.queries)} queries, needs_rag={slot_plan.needs_rag}")

        return {
            "slot_plan": slot_plan,
            "owner": owner,
            "needs_rag": slot_plan.needs_rag,
        }

    except Exception as e:
        logger.error(f"[Route] Failed: {e}")

        return {
            "slot_plan": SlotPlan(queries=[], needs_rag=False),
            "owner": owner,
            "needs_rag": False,
            "error": f"Query planning failed: {e}",
        }
