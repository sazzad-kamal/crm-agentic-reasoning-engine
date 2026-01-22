"""Answer synthesis node for agent workflow."""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.answer.answerer import call_answer_chain, extract_suggested_action
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def answer_node(state: AgentState) -> AgentState:
    """Synthesize answer from SQL results."""
    logger.info("[Answer] Synthesizing response...")

    try:
        raw_answer = call_answer_chain(
            question=state["question"],
            sql_results=state.get("sql_results", {}),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
        )

        # Validate answer
        if not raw_answer or not raw_answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            raw_answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        # Extract suggested action (strips it from answer)
        answer, action = extract_suggested_action(raw_answer)

        logger.info("[Answer] Synthesized response")

        return {
            "answer": answer,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
            "suggested_actions": [action] if action else [],
        }

    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        return {
            "answer": error_answer,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=error_answer),
            ],
            "suggested_actions": [],
            "error": str(e),
        }


__all__ = ["answer_node"]
