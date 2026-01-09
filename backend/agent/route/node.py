"""
LangGraph routing node for agent workflow.

Handles question routing and parameter extraction.
"""

import logging
import time

from backend.agent.core.config import get_config
from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.router import detect_owner_from_starter, route_question

logger = logging.getLogger(__name__)


def route_node(state: AgentState) -> AgentState:
    config = get_config()
    start_time = time.time()
    question = state["question"]

    logger.info(f"[Route] Processing: {question[:50]}...")

    # Format conversation history for the router (session memory handles company context)
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    try:
        # LLM router returns only company_id and intent
        router_result = route_question(
            question,
            conversation_history=conversation_history,
        )

        # Detect owner from starter patterns (not LLM-derived)
        owner = detect_owner_from_starter(question)

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[Route] Result: company={router_result.company_id}, "
            f"intent={router_result.intent}, latency={latency_ms}ms"
        )

        return {  # type: ignore[typeddict-unknown-key]
            "router_result": router_result,  # LLM result (company_id, intent)
            "resolved_company_id": router_result.company_id,
            "days": config.default_days,  # Always from config (90)
            "intent": router_result.intent or "pipeline_summary",
            "owner": owner,  # Pattern-matched from question
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

        return {
            "days": config.default_days,
            "intent": "pipeline_summary",
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
