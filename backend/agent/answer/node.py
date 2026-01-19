"""Answer synthesis node for agent workflow."""

import logging

from backend.agent.answer.answerer import call_answer_chain
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def answer_node(state: AgentState) -> AgentState:
    """Synthesize answer from SQL results and RAG context."""
    logger.info("[Answer] Synthesizing response...")

    try:
        answer, _ = call_answer_chain(
            question=state["question"],
            sql_results=state.get("sql_results", {}),
            rag_context=state.get("rag_context", ""),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
        )

        # Validate answer
        if not answer or not answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        logger.info("[Answer] Synthesized response")

        return {
            "answer": answer,
            "messages": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }

    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        return {
            "answer": error_answer,
            "messages": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": error_answer},
            ],
            "error": str(e),
        }


__all__ = ["answer_node"]
