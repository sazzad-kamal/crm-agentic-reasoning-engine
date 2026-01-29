"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import math
import warnings
from functools import cache
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.llm import get_langchain_chat_openai
from backend.eval.answer.text.suppression import (
    install_event_loop_error_suppression,
    suppress_ragas_logging,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness

logger = logging.getLogger(__name__)

# Install suppression handlers at module load
install_event_loop_error_suppression()


@cache
def _evaluators() -> tuple[Any, ...]:
    """Get shared RAGAS evaluators (cached singleton)."""
    logger.info("Initializing RAGAS evaluators")
    llm = LangchainLLMWrapper(get_langchain_chat_openai())

    return (AnswerCorrectness(llm=llm), AnswerRelevancy(llm=llm), Faithfulness(llm=llm))


# Number of RAGAS metrics evaluated (answer_correctness + answer_relevancy + faithfulness)
RAGAS_METRICS_COUNT = 3


def _extract_scores(eval_result: Any) -> dict[str, float | list[str]]:
    """Extract scores from RAGAS evaluation result."""
    df = eval_result.to_pandas()
    nan_metrics: list[str] = []

    def get_score(name: str) -> float:
        try:
            val = df[name].iloc[0]
        except (KeyError, IndexError):
            nan_metrics.append(name)
            return 0.0
        if val is None or (isinstance(val, float) and math.isnan(val)):
            nan_metrics.append(name)
            return 0.0
        return float(val)

    return {
        "answer_correctness": get_score("answer_correctness"),
        "answer_relevancy": get_score("answer_relevancy"),
        "faithfulness": get_score("faithfulness"),
        "nan_metrics": nan_metrics,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4), reraise=True)
def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str,
) -> dict[str, float | list[str]]:
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    Args:
        question: The user's question
        answer: The agent's answer
        contexts: List of retrieved context strings
        reference_answer: Ground truth answer for answer_correctness metric

    Returns:
        dict with answer_relevancy, faithfulness, answer_correctness (0.0-1.0)
        and nan_metrics list tracking which metrics returned NaN
    """
    dataset = Dataset.from_dict({
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
        "reference": [reference_answer],
    })

    with suppress_ragas_logging():
        eval_result = evaluate(dataset, metrics=list(_evaluators()), show_progress=False)
        return _extract_scores(eval_result)


__all__ = ["RAGAS_METRICS_COUNT", "evaluate_single"]
