"""LLM judge for suggested action quality."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)


class ActionJudgeResult(BaseModel):
    """Result from the action judge."""

    relevance: float = Field(description="0-1: Does action relate to question/answer?")
    actionability: float = Field(description="0-1: Is it specific and executable?")
    appropriateness: float = Field(description="0-1: Is it sensible for CRM context?")
    explanation: str = Field(description="Brief reasoning")


_SYSTEM_PROMPT = """Evaluate the suggested action from a CRM assistant.

Context: The assistant answers CRM questions and optionally suggests a next action.

Score each dimension 0.0 to 1.0:
1. Relevance: Does the action relate to the question and answer?
2. Actionability: Is it specific enough to execute? (not vague like "follow up")
3. Appropriateness: Is it sensible given CRM best practices?

Be strict - vague actions should score low on actionability."""

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}

Suggested Action: {action}"""


def judge_suggested_action(
    question: str,
    answer: str,
    action: str,
) -> tuple[bool, float, float, float, str]:
    """
    Judge suggested action quality.

    Returns:
        (passed, relevance, actionability, appropriateness, explanation)
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=ActionJudgeResult,
        streaming=False,
    )
    try:
        result: ActionJudgeResult = chain.invoke({
            "question": question,
            "answer": answer,
            "action": action,
        })
        passed = all(s >= 0.6 for s in (result.relevance, result.actionability, result.appropriateness))
        return (
            passed,
            result.relevance,
            result.actionability,
            result.appropriateness,
            result.explanation,
        )
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return False, 0.0, 0.0, 0.0, f"Judge error: {e}"


__all__ = ["ActionJudgeResult", "judge_suggested_action"]
