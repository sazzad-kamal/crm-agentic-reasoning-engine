"""
Shared LLM-as-judge utilities for evaluation harnesses.

Provides common prompts and helper functions for RAG and agent evaluation.
"""

import json
import re
import logging
from dataclasses import dataclass

from backend.common.llm_client import call_llm


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class JudgeScores:
    """Common judge scores used across eval harnesses."""
    answer_relevance: int  # 0 or 1
    answer_grounded: int   # 0 or 1
    explanation: str = ""


# =============================================================================
# Shared Prompts
# =============================================================================

# Base system prompt template for answer evaluation
JUDGE_SYSTEM_BASE = """You are an expert evaluator for a {domain}.
Evaluate the quality of the assistant's answer to a user question.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on the provided sources/data?
   - 1 if the answer appears grounded in real data (mentions specific {data_type})
   - 0 if the answer seems made up or generic

Respond in JSON:
{{
  "answer_relevance": 0 or 1,
  "answer_grounded": 0 or 1,
  "explanation": "brief explanation"
}}"""


def format_judge_system(domain: str, data_type: str = "facts") -> str:
    """Format the judge system prompt for a specific domain."""
    return JUDGE_SYSTEM_BASE.format(domain=domain, data_type=data_type)


# Pre-formatted system prompts for common use cases
JUDGE_SYSTEM_CRM = format_judge_system(
    domain="CRM assistant",
    data_type="companies, dates, values"
)

JUDGE_SYSTEM_RAG = format_judge_system(
    domain="RAG system answering documentation questions",
    data_type="document references and technical details"
)


# Standard judge prompt template
JUDGE_PROMPT = """Question: {question}

Answer: {answer}

Sources cited: {sources}

Evaluate this response:"""


# =============================================================================
# Helper Functions
# =============================================================================

def parse_judge_response(response: str) -> dict:
    """
    Parse a JSON response from the judge LLM.

    Handles markdown code blocks and extracts JSON.

    Args:
        response: Raw response from judge LLM

    Returns:
        Parsed dictionary with scores
    """
    if not response or not response.strip():
        raise ValueError("Empty response from judge LLM")

    text = response

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # Try to extract JSON object
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(text.strip())


def judge_answer(
    question: str,
    answer: str,
    sources: list[str],
    system_prompt: str = JUDGE_SYSTEM_CRM,
    model: str = "gpt-4o-mini",
) -> JudgeScores:
    """
    Judge an answer using LLM-as-judge.

    Args:
        question: The user's question
        answer: The generated answer
        sources: List of source identifiers
        system_prompt: System prompt for judge context
        model: LLM model to use for judging

    Returns:
        JudgeScores with relevance, groundedness, and explanation
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        sources=", ".join(sources) if sources else "None",
    )

    try:
        response = call_llm(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.0,
            max_tokens=500,
        )

        result = parse_judge_response(response)
        return JudgeScores(
            answer_relevance=result.get("answer_relevance", 0),
            answer_grounded=result.get("answer_grounded", 0),
            explanation=result.get("explanation", ""),
        )

    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return JudgeScores(
            answer_relevance=0,
            answer_grounded=0,
            explanation=f"Judge error: {str(e)}",
        )


__all__ = [
    "JudgeScores",
    "JUDGE_SYSTEM_BASE",
    "JUDGE_SYSTEM_CRM",
    "JUDGE_SYSTEM_RAG",
    "JUDGE_PROMPT",
    "format_judge_system",
    "parse_judge_response",
    "judge_answer",
]
