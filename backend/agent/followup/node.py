"""Follow-up suggestion node for agent workflow."""

import logging
import time

from backend.agent.core.config import get_config
from backend.agent.core.state import AgentState
from backend.agent.followup.llm import generate_follow_up_suggestions

logger = logging.getLogger(__name__)


def followup_node(state: AgentState) -> AgentState:
    config = get_config()
    start_time = time.time()

    if not config.enable_follow_up_suggestions:
        logger.info("[Followup] Suggestions disabled in config")
        return {"follow_up_suggestions": []}

    logger.info("[Followup] Generating suggestions...")

    # Use pre-formatted conversation_history from route_node
    conversation_history = state.get("conversation_history", "")

    # Get SQL results from fetch_sql
    sql_results = state.get("sql_results", {})

    # Extract company name from sql_results if available
    company_name = None
    company_info = sql_results.get("company_info", [])
    if not company_info:
        company_info = sql_results.get("companies", [])
    if company_info and isinstance(company_info, list) and len(company_info) > 0:
        company_name = company_info[0].get("name")

    # Build available data counts from sql_results
    available_data = {}
    for purpose, data in sql_results.items():
        if isinstance(data, list):
            available_data[purpose] = len(data)
        elif data:
            available_data[purpose] = 1

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            company_id=state.get("resolved_company_id"),
            company_name=company_name,
            conversation_history=conversation_history,
            available_data=available_data,
        )

        # Validate and limit suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]
            suggestions = suggestions[: config.max_followup_suggestions]

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Followup] Generated {len(suggestions)} suggestions in {latency_ms}ms")

        return {
            "follow_up_suggestions": suggestions,
            "steps": [
                {
                    "id": "followup",
                    "label": "Generating suggestions",
                    "status": "done",
                    "latency_ms": latency_ms,
                }
            ],
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"[Followup] Failed after {latency_ms}ms: {e}")
        return {
            "follow_up_suggestions": [],
            "steps": [
                {
                    "id": "followup",
                    "label": "Generating suggestions",
                    "status": "error",
                    "latency_ms": latency_ms,
                }
            ],
        }


__all__ = ["followup_node"]
