"""Follow-up suggestion node for agent workflow."""

import logging

from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def followup_node(state: AgentState) -> AgentState:
    """Generate follow-up suggestions based on current state."""
    logger.info("[Followup] Generating suggestions...")

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
            company_name=company_name,
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
            available_data=available_data,
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
