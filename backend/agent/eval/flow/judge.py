"""LLM judge for flow evaluation."""

from __future__ import annotations

import logging

from backend.agent.eval.shared import run_llm_judge
from backend.agent.eval.prompts import FLOW_JUDGE_SYSTEM, FLOW_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
) -> dict:
    """Judge an answer using LLM."""
    prompt = FLOW_JUDGE_PROMPT.format(
        question=question,
        answer=answer[:1000],  # Truncate long answers
        context=context[:500] if context else "First question in conversation",
    )

    result = run_llm_judge(prompt, FLOW_JUDGE_SYSTEM)

    if "error" in result:
        logger.warning(f"Judge failed: {result['error']}")
        return {"relevance": 0, "grounded": 0, "explanation": f"Judge error: {result['error']}"}

    return {
        "relevance": result.get("answer_relevance", 0),
        "grounded": result.get("answer_grounded", 0),
        "explanation": result.get("explanation", ""),
    }
