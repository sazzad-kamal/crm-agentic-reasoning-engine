"""Action suggestion node for agent workflow."""

import logging

from backend.agent.action.suggester import call_action_chain
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def action_node(state: AgentState) -> AgentState:
    """Suggest an actionable next step based on the answer."""
    if state.get("error"):
        return {"suggested_action": None}

    logger.info("[Action] Evaluating action suggestion...")

    try:
        action = call_action_chain(
            question=state["question"],
            answer=state["answer"],
        )

        if action:
            logger.info("[Action] Suggested action")
        else:
            logger.info("[Action] No action suggested")

        return {"suggested_action": action}

    except Exception as e:
        logger.warning(f"[Action] Failed: {e}")
        return {"suggested_action": None}


__all__ = ["action_node"]
