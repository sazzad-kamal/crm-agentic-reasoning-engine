"""
LangGraph routing node for agent workflow.

Uses slot-based query planning to generate SQL queries.
"""

import logging
import time

from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.query_planner import (
    QueryPlan,
    detect_owner_from_starter,
    get_slot_plan,
    slot_plan_to_query_plan,
)

logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    """
    Route node that generates SQL query plan from user question.

    Sets query_plan in state for downstream nodes.
    """
    start_time = time.time()
    question = state["question"]

    logger.info(f"[Route] Processing: {question[:50]}...")

    # Format conversation history for the planner
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    # Detect owner from starter patterns
    owner = detect_owner_from_starter(question)

    try:
        # Get query plan from LLM using slot-based planning
        slot_plan = get_slot_plan(
            question=question,
            conversation_history=conversation_history,
            owner=owner,
        )
        query_plan = slot_plan_to_query_plan(slot_plan)
        needs_rag = slot_plan.needs_rag
        logger.debug(f"[Route] Slot planning: {len(slot_plan.queries)} queries, needs_rag={needs_rag}")

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[Route] Result: {len(query_plan.queries)} queries, latency={latency_ms}ms"
        )

        return {
            "query_plan": query_plan,
            "owner": owner,
            "needs_rag": needs_rag,
            "conversation_history": conversation_history,
            "steps": [
                {
                    "id": "router",
                    "label": "Understanding your question",
                    "status": "done",
                    "latency_ms": latency_ms,
                }
            ],
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[Route] Failed after {latency_ms}ms: {e}")

        return {
            "query_plan": QueryPlan(queries=[]),
            "owner": owner,
            "needs_rag": False,
            "conversation_history": conversation_history,
            "error": f"Query planning failed: {e}",
            "steps": [
                {
                    "id": "router",
                    "label": "Understanding your question",
                    "status": "error",
                    "latency_ms": latency_ms,
                }
            ],
        }
