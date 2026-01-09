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

    # Extract company name from company_data
    company_data = state.get("company_data", {})
    company_name = None
    if company_data and company_data.get("found"):
        company_info = company_data.get("company", {})
        company_name = company_info.get("name")

    # Build available data counts from raw_data
    raw_data = state.get("raw_data", {})
    available_data = {
        "contacts": len(raw_data.get("contacts", [])) if isinstance(raw_data, dict) else 0,
        "activities": len(raw_data.get("activities", [])) if isinstance(raw_data, dict) else 0,
        "opportunities": len(raw_data.get("opportunities", []))
        if isinstance(raw_data, dict)
        else 0,
        "history": len(raw_data.get("history", [])) if isinstance(raw_data, dict) else 0,
        "renewals": len(raw_data.get("renewals", [])) if isinstance(raw_data, dict) else 0,
        "pipeline_summary": raw_data.get("pipeline_summary")
        if isinstance(raw_data, dict)
        else None,
    }

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
            "followup_latency_ms": latency_ms,
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
            "followup_latency_ms": latency_ms,
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
