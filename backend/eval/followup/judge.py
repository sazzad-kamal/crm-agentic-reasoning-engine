"""LLM judge for followup suggestion quality."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Structured output from the followup judge."""

    passed: bool = Field(description="Whether the suggestions are acceptable quality")
    relevance_score: float = Field(
        description="Score 0-1: Are suggestions relevant to the original question?",
        ge=0.0,
        le=1.0,
    )
    diversity_score: float = Field(
        description="Score 0-1: Do suggestions cover different angles/topics?",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation of the judgment")
    errors: list[str] = Field(default_factory=list, description="List of issues found")


_SYSTEM_PROMPT = """You are evaluating follow-up question suggestions for a CRM assistant.

Given the user's original question and the generated follow-up suggestions, evaluate:

1. RELEVANCE (0-1): Are the suggestions related to the original question?
   - 1.0: All suggestions are clearly relevant follow-ups
   - 0.5: Some suggestions are relevant, others are not
   - 0.0: Suggestions are unrelated to the question

2. DIVERSITY (0-1): Do suggestions cover different angles?
   - 1.0: Each suggestion explores a different aspect or direction
   - 0.5: Some overlap but reasonable variety
   - 0.0: All suggestions are essentially the same question

3. PASSED: True if relevance >= 0.6 AND diversity >= 0.5

Consider:
- Follow-ups should be natural next questions a user might ask
- Good follow-ups help the user explore related data or drill deeper
- At least one suggestion should offer a different direction (different company, general question)

If passed=true, errors should be empty.
If passed=false, list specific issues."""

_HUMAN_PROMPT = """## Original Question
{question}

## Generated Follow-up Suggestions
{suggestions}

Evaluate the quality of these follow-up suggestions."""


def judge_followup_suggestions(
    question: str,
    suggestions: list[str],
) -> tuple[bool, float, float, list[str]]:
    """
    Use LLM to judge if follow-up suggestions are good quality.

    Args:
        question: The user's original question
        suggestions: List of generated follow-up suggestions

    Returns:
        Tuple of (passed, relevance_score, diversity_score, errors)
    """
    if not suggestions:
        return False, 0.0, 0.0, ["No suggestions generated"]

    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=JudgeResult,
        streaming=False,
    )

    try:
        formatted_suggestions = "\n".join(f"- {s}" for s in suggestions)

        result: JudgeResult = chain.invoke({
            "question": question,
            "suggestions": formatted_suggestions,
        })

        errors = result.errors
        if not result.passed and result.reasoning and not errors:
            errors = [result.reasoning]

        logger.debug(
            f"Followup Judge: passed={result.passed}, "
            f"relevance={result.relevance_score:.2f}, "
            f"diversity={result.diversity_score:.2f}"
        )

        return result.passed, result.relevance_score, result.diversity_score, errors

    except Exception as e:
        logger.warning(f"Followup Judge error: {e}")
        return False, 0.0, 0.0, [f"Judge API error: {e}"]


__all__ = ["judge_followup_suggestions"]
