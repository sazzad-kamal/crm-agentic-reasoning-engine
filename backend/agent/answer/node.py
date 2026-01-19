"""Answer synthesis node for agent workflow."""

import logging

from backend.agent.answer.answerer import call_answer_chain
from backend.agent.state import AgentState, Message, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def _build_messages(state: AgentState, answer: str) -> list[Message]:
    """Build updated message list with user question and assistant answer."""
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": state["question"]})
    messages.append({"role": "assistant", "content": answer})
    return messages


def answer_node(state: AgentState) -> AgentState:
    """
    Synthesize answer from SQL results and account context.

    Uses simplified call_answer_chain with sql_results and account_context.
    """
    logger.info("[Answer] Synthesizing response...")

    try:
        answer, _ = call_answer_chain(
            question=state["question"],
            sql_results=state.get("sql_results", {}),
            account_context=state.get("rag_context", ""),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
        )

        # Validate answer
        if not answer or not answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        logger.info("[Answer] Synthesized response")

        return {
            "answer": answer,
            "messages": _build_messages(state, answer),
        }

    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        return {
            "answer": error_answer,
            "messages": _build_messages(state, error_answer),
            "error": str(e),
        }


__all__ = ["answer_node"]
