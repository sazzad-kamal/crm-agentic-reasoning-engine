"""LLM judge for suggested action quality."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.answer.action.models import SLO_ACTIONABILITY, SLO_APPROPRIATENESS, SLO_RELEVANCE


class ActionJudgeResult(BaseModel):
    """Result from the action judge."""

    relevance: float = Field(description="0-1: Does action relate to question/answer?")
    actionability: float = Field(description="0-1: Is it specific enough to execute and drive immediate progress (who + what + when)?")
    appropriateness: float = Field(description="0-1: Is it sensible for CRM context?")
    explanation: str = Field(description="Brief reasoning")


_SYSTEM_PROMPT = """Evaluate the suggested action from a CRM assistant.

Context: The assistant answers CRM questions and optionally suggests a next action.

Score each dimension 0.0 to 1.0:
1. Relevance: Does the action relate to the question and answer?
2. Actionability: Is it specific enough to execute and drive immediate progress? Score high when the action names who should act, what to do, and by when (e.g. "Schedule renewal call with Sarah Chen at Beta Tech by Feb 5"). Score low for vague actions like "follow up" with no owner or timeline.
3. Appropriateness: Is it sensible given CRM best practices?"""

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

    Raises:
        Exception: If the LLM chain fails (caller should handle).
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=ActionJudgeResult,
        streaming=False,
    )
    result: ActionJudgeResult = chain.invoke({
        "question": question,
        "answer": answer,
        "action": action,
    })
    passed = (
        result.relevance >= SLO_RELEVANCE
        and result.actionability >= SLO_ACTIONABILITY
        and result.appropriateness >= SLO_APPROPRIATENESS
    )
    return (
        passed,
        result.relevance,
        result.actionability,
        result.appropriateness,
        result.explanation,
    )


__all__ = ["ActionJudgeResult", "judge_suggested_action"]
