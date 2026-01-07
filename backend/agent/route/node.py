"""
LangGraph routing node for agent workflow.

Handles question routing and parameter extraction.
"""

import logging
import time

from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.core.config import get_config
from backend.agent.route.router import route_question


logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    config = get_config()
    start_time = time.time()

    logger.info(f"[Route] Processing: {state['question'][:50]}...")

    # Format conversation history for the router (session memory handles company context)
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    try:
        router_result = route_question(
            state["question"],
            conversation_history=conversation_history,
        )

        # Validate router_result has required fields
        if not hasattr(router_result, "mode_used") or not router_result.mode_used:
            raise ValueError("Router returned invalid result: missing mode_used")

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[Route] Result: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}, "
            f"latency={latency_ms}ms"
        )

        return {
            "router_result": router_result,  # Full result for downstream access
            "mode_used": router_result.mode_used,
            "resolved_company_id": router_result.company_id,
            "company_name_query": router_result.company_name_query,
            "days": router_result.days or config.default_days,
            "intent": router_result.intent or "general",
            "owner": router_result.owner,
            "conversation_history": conversation_history,  # Formatted once, reused by other nodes
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

        # Determine fallback mode based on question content
        question_lower = state["question"].lower()
        fallback_mode = "docs"  # Default fallback
        if any(
            kw in question_lower for kw in ["company", "account", "customer", "pipeline", "renewal"]
        ):
            fallback_mode = "data"

        return {
            "mode_used": fallback_mode,
            "days": config.default_days,
            "intent": "general",
            "router_latency_ms": latency_ms,
            "error": f"Routing failed: {e}",
            "steps": [
                {
                    "id": "router",
                    "label": "Understanding your question",
                    "status": "error",
                    "latency_ms": latency_ms,
                }
            ],
        }
