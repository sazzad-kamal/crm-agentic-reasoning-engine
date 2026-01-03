"""LLM judge and refusal checking for E2E evaluation."""

from __future__ import annotations

from backend.agent.eval.shared import run_llm_judge
from backend.agent.eval.prompts import E2E_JUDGE_SYSTEM, E2E_JUDGE_PROMPT


def judge_e2e_response(
    question: str,
    answer: str,
    sources: list[str],
    category: str = "data",
) -> dict:
    """Judge an end-to-end response using LLM."""
    prompt = E2E_JUDGE_PROMPT.format(
        question=question,
        category=category.upper(),
        answer=answer,
        sources=", ".join(sources) if sources else "None",
    )

    result = run_llm_judge(prompt, E2E_JUDGE_SYSTEM)

    if "error" in result:
        return {
            "answer_relevance": 0,
            "answer_grounded": 0,
            "context_relevance": 0,
            "faithfulness": 0,
            "explanation": f"Judge error: {result['error']}",
        }

    return {
        "answer_relevance": result.get("answer_relevance", 0),
        "answer_grounded": result.get("answer_grounded", 0),
        "context_relevance": result.get("context_relevance", 0),
        "faithfulness": result.get("faithfulness", 0),
        "explanation": result.get("explanation", ""),
    }


def check_refusal_response(
    answer: str,
    expected_refusal: bool,
    refusal_keywords: list[str],
    forbidden_keywords: list[str],
) -> tuple[bool, bool]:
    """
    Check if the response correctly handles refusal.

    Returns:
        (refusal_correct, has_forbidden_content)
    """
    answer_lower = answer.lower()

    # Check for forbidden content
    has_forbidden = any(kw.lower() in answer_lower for kw in forbidden_keywords)

    if expected_refusal:
        # Should refuse - check for refusal keywords
        has_refusal = any(kw.lower() in answer_lower for kw in refusal_keywords)
        refusal_correct = has_refusal
    else:
        # No refusal expected - just check forbidden keywords
        refusal_correct = True

    return refusal_correct, has_forbidden
