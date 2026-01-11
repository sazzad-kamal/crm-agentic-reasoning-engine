"""
LangGraph routing node for agent workflow.

Uses schema-driven query planning to generate SQL queries.
"""

import logging
import time

from backend.agent.core.config import get_config
from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.query_planner import (
    QueryPlan,
    detect_owner_from_starter,
    get_query_plan,
    get_slot_plan,
    slot_plan_to_query_plan,
)

logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    """
    Route node that generates SQL query plan from user question.

    Sets query_plan in state for downstream nodes.
    """
    config = get_config()
    start_time = time.time()
    question = state["question"]

    logger.info(f"[Route] Processing: {question[:50]}...")

    # Format conversation history for the planner
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    # Detect owner from starter patterns
    owner = detect_owner_from_starter(question)

    try:
        # Get query plan from LLM
        needs_rag = False  # Default for raw SQL mode
        if config.use_slot_filling:
            # Slot-based planning: LLM outputs structured slots, we build SQL
            slot_plan = get_slot_plan(
                question=question,
                conversation_history=conversation_history,
                owner=owner,
            )
            query_plan = slot_plan_to_query_plan(slot_plan)
            needs_rag = slot_plan.needs_rag
            logger.debug(f"[Route] Using slot-based planning, converted {len(slot_plan.queries)} slots to SQL, needs_rag={needs_rag}")
        else:
            # Raw SQL planning: LLM generates SQL directly
            query_plan = get_query_plan(
                question=question,
                conversation_history=conversation_history,
                owner=owner,
            )

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[Route] Result: {len(query_plan.queries)} queries, latency={latency_ms}ms"
        )

        return {
            "query_plan": query_plan,  # QueryPlan with SQL queries
            "days": config.default_days,  # Always from config (90)
            "owner": owner,  # Pattern-matched from question
            "needs_rag": needs_rag,  # Whether RAG context is needed
            "conversation_history": conversation_history,
            "router_latency_ms": latency_ms,
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

        # Return empty query plan on failure
        return {
            "query_plan": QueryPlan(queries=[]),
            "days": config.default_days,
            "owner": owner,
            "needs_rag": False,  # Default to false on error
            "conversation_history": conversation_history,
            "router_latency_ms": latency_ms,
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
