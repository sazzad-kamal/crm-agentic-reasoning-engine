"""
Shared LLM-as-judge utilities using LangChain.

Provides structured output parsing and evaluation prompts for RAG and agent evaluation.
"""

import logging
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from backend.common.llm_client import call_llm
from backend.rag.pipeline.constants import JUDGE_MODEL


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models with Pydantic
# =============================================================================

class JudgeScores(BaseModel):
    """Structured judge scores with automatic parsing."""
    answer_relevance: int = Field(
        description="1 if the answer directly addresses the question, 0 if off-topic"
    )
    answer_grounded: int = Field(
        description="1 if the answer contains specific data (companies, dates, values), 0 if generic"
    )
    explanation: str = Field(
        description="Brief explanation of the scores"
    )


# =============================================================================
# Output Parser
# =============================================================================

judge_parser = PydanticOutputParser(pydantic_object=JudgeScores)


# =============================================================================
# Prompt Templates
# =============================================================================

JUDGE_SYSTEM_CRM = """You are an expert evaluator for a CRM assistant.
Evaluate the quality of the assistant's answer to a user question.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on real data?
   - 1 if the answer contains SPECIFIC data (company names, dates, dollar amounts, counts)
   - 0 if the answer is generic or vague (uses words like "several", "some", "recent")

{format_instructions}"""

JUDGE_SYSTEM_RAG = """You are an expert evaluator for a RAG system answering documentation questions.
Evaluate the quality of the assistant's answer to a user question.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on the provided sources?
   - 1 if the answer contains specific technical details from documentation
   - 0 if the answer seems made up or generic

{format_instructions}"""

JUDGE_HUMAN_TEMPLATE = """Question: {question}

Answer: {answer}

Sources cited: {sources}

Evaluate this response:"""

# Create the full prompt template
JUDGE_PROMPT_CRM = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_CRM),
    ("human", JUDGE_HUMAN_TEMPLATE),
]).partial(format_instructions=judge_parser.get_format_instructions())

JUDGE_PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_RAG),
    ("human", JUDGE_HUMAN_TEMPLATE),
]).partial(format_instructions=judge_parser.get_format_instructions())


# =============================================================================
# Helper Functions
# =============================================================================

def format_judge_system(domain: str, data_type: str = "facts") -> str:
    """Format the judge system prompt for a specific domain."""
    return f"""You are an expert evaluator for a {domain}.
Evaluate the quality of the assistant's answer to a user question.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on the provided sources/data?
   - 1 if the answer contains specific {data_type}
   - 0 if the answer seems made up or generic

{judge_parser.get_format_instructions()}"""


def judge_answer(
    question: str,
    answer: str,
    sources: list[str],
    domain: str = "crm",
    model: str = JUDGE_MODEL,
) -> JudgeScores:
    """
    Judge an answer using LLM-as-judge with structured output.

    Args:
        question: The user's question
        answer: The generated answer
        sources: List of source identifiers
        domain: "crm" or "rag" for different evaluation criteria
        model: LLM model to use for judging

    Returns:
        JudgeScores with relevance, groundedness, and explanation
    """
    # Select the appropriate prompt
    prompt = JUDGE_PROMPT_CRM if domain == "crm" else JUDGE_PROMPT_RAG

    # Format the prompt
    formatted = prompt.format_messages(
        question=question,
        answer=answer,
        sources=", ".join(sources) if sources else "None",
    )

    try:
        # Call LLM
        response = call_llm(
            prompt=formatted[-1].content,  # Human message
            system_prompt=formatted[0].content,  # System message
            model=model,
            temperature=0.0,
            max_tokens=500,
        )

        # Parse with Pydantic
        result = judge_parser.parse(response)
        return result

    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return JudgeScores(
            answer_relevance=0,
            answer_grounded=0,
            explanation=f"Judge error: {str(e)}",
        )


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Old-style prompt strings for backwards compatibility
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

JUDGE_PROMPT = """Question: {question}

Answer: {answer}

Sources cited: {sources}

Evaluate this response:"""


def parse_judge_response(response: str) -> dict:
    """Parse a JSON response from the judge LLM (legacy compatibility)."""
    import json
    import re

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


__all__ = [
    "JudgeScores",
    "judge_answer",
    "judge_parser",
    "JUDGE_PROMPT_CRM",
    "JUDGE_PROMPT_RAG",
    # Legacy
    "JUDGE_SYSTEM_BASE",
    "JUDGE_PROMPT",
    "format_judge_system",
    "parse_judge_response",
]
