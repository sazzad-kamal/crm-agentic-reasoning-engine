"""5-dimension LLM judge for answer text quality.

Scores CRM query answers on grounding, completeness, clarity,
accuracy, and actionability. Mirrors the action judge pattern
in eval/answer/action/judge.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.answer.text.models import (
    SLO_JUDGE_ACCURACY,
    SLO_JUDGE_ACTIONABILITY,
    SLO_JUDGE_CLARITY,
    SLO_JUDGE_COMPLETENESS,
    SLO_JUDGE_GROUNDING,
)


class TextJudgeResult(BaseModel):
    """Result from the 5-dimension text quality judge."""

    grounding: float = Field(description="0-1: Every claim has evidence tags, no fabrication")
    completeness: float = Field(description="0-1: Addresses all parts of the question")
    clarity: float = Field(description="0-1: Well-structured, easy to scan")
    accuracy: float = Field(description="0-1: Numbers and names match the CRM data provided")
    actionability: float = Field(description="0-1: Suggests practical next steps when appropriate")
    explanation: str = Field(description="Brief reasoning for scores")


_SYSTEM_PROMPT = """Evaluate the quality of a CRM data assistant's answer.

You are scoring the ANSWER text, not the data. The answer was generated from CRM query results.

Score each dimension 0.0 to 1.0:

1. **Grounding**: Every factual claim has an evidence tag like [E1], [E2]. No hallucinated data.
   Score 0 if claims lack evidence tags. Score 1 if every claim is tagged and traceable.

2. **Completeness**: The answer addresses ALL parts of the question. Multi-part questions
   should have all sub-questions answered. Score 0 if major parts are missing.

3. **Clarity**: The answer is well-organized, uses formatting (bullets, bold) when helpful,
   and is easy to scan. Score 0 for wall-of-text or confusing structure.

4. **Accuracy**: Numbers, names, dates, and statuses match the CRM data context provided.
   Score 0 if values are wrong or fabricated. Score 1 if all values are verifiable.

5. **Actionability**: When appropriate, the answer suggests what to do next (follow up,
   schedule a call, review a deal). Score high when actions are specific with who/what/when.
   Score 0.5 if no action is needed and none is suggested (neutral)."""

_HUMAN_PROMPT = """Question: {question}

CRM Data Context:
{context}

Answer to evaluate:
{answer}"""


def judge_answer_text(
    question: str,
    answer: str,
    context: str,
) -> tuple[bool, float, float, float, float, float, str]:
    """Judge answer text quality across 5 dimensions.

    Returns:
        (passed, grounding, completeness, clarity, accuracy, actionability, explanation)
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=TextJudgeResult,
        streaming=False,
    )
    result: TextJudgeResult = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context,
    })
    passed = (
        result.grounding >= SLO_JUDGE_GROUNDING
        and result.completeness >= SLO_JUDGE_COMPLETENESS
        and result.clarity >= SLO_JUDGE_CLARITY
        and result.accuracy >= SLO_JUDGE_ACCURACY
        and result.actionability >= SLO_JUDGE_ACTIONABILITY
    )
    return (
        passed,
        result.grounding,
        result.completeness,
        result.clarity,
        result.accuracy,
        result.actionability,
        result.explanation,
    )


__all__ = ["TextJudgeResult", "judge_answer_text"]
