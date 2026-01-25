"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import math
import warnings
from functools import cache
from typing import Any, cast

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.llm import get_langchain_chat_openai, get_langchain_embeddings
from backend.eval.answer.text.suppression import (
    install_event_loop_error_suppression,
    suppress_ragas_logging,
)

# Import metric CLASSES (not singleton instances) for thread-safe instantiation
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
    )

logger = logging.getLogger(__name__)

# Install suppression handlers at module load
install_event_loop_error_suppression()


@cache
def _get_ragas_llm() -> Any:
    """Get shared LLM for RAGAS (cached singleton)."""
    logger.info("Initializing RAGAS LLM")
    return LangchainLLMWrapper(get_langchain_chat_openai())


@cache
def _get_ragas_embeddings() -> Any:
    """Get shared embeddings for RAGAS (cached singleton)."""
    logger.info("Initializing RAGAS embeddings")
    return LangchainEmbeddingsWrapper(get_langchain_embeddings())


@cache
def _get_ragas_metrics(include_reference: bool = False) -> tuple[Any, ...]:
    """Get shared RAGAS metrics (cached singleton).

    Args:
        include_reference: If True, include metrics that require reference answer
                          (ContextRecall, AnswerCorrectness)

    Returns:
        Tuple of RAGAS metric instances
    """
    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()
    logger.info("Initializing RAGAS metrics")

    base = (
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
    )
    if include_reference:
        return base + (AnswerCorrectness(llm=llm),)
    return base


def _extract_scores(df: Any) -> dict[str, float | str | list[str] | None]:
    """Extract scores from RAGAS evaluation DataFrame."""
    nan_metrics: list[str] = []

    def get_score(name: str) -> float:
        if name in df.columns and len(df) > 0:
            val = df[name].iloc[0]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                nan_metrics.append(name)
                return 0.0
            return float(val)
        return 0.0

    return {
        "answer_relevancy": get_score("answer_relevancy"),
        "faithfulness": get_score("faithfulness"),
        "answer_correctness": get_score("answer_correctness"),
        "error": None,
        "nan_metrics": nan_metrics,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4), reraise=True)
def _run_evaluation_with_retry(dataset: Dataset, metrics: tuple) -> dict[str, float | str | list[str] | None]:
    """Run RAGAS evaluation with automatic retry on failure."""
    eval_result = evaluate(dataset, metrics=list(metrics), show_progress=False)
    df = eval_result.to_pandas()  # type: ignore[union-attr]
    return _extract_scores(df)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    verbose: bool = False,
) -> dict[str, float | str | list[str] | None]:
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    Args:
        question: The user's question
        answer: The agent's answer
        contexts: List of retrieved context strings
        reference_answer: Optional ground truth answer for answer_correctness metric
        verbose: Show RAGAS output (default: suppress)

    Returns:
        dict with answer_relevancy, faithfulness, context_precision, answer_correctness (0.0-1.0)
        Also includes 'error' key (None if success, error message string if failed)
    """
    # Skip RAGAS if no contexts - metrics would be meaningless
    if not contexts:
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "error": "No contexts provided - skipping RAGAS",
            "nan_metrics": [],
        }

    dataset = Dataset.from_dict({
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
        **({"reference": [reference_answer]} if reference_answer else {}),
    })
    metrics = _get_ragas_metrics(include_reference=bool(reference_answer))

    try:
        if verbose:
            result = _run_evaluation_with_retry(dataset, metrics)
        else:
            with suppress_ragas_logging():
                result = _run_evaluation_with_retry(dataset, metrics)

        nan_metrics = cast(list[str], result.get("nan_metrics", []))
        if nan_metrics:
            result["error"] = f"RAGAS returned NaN for: {', '.join(nan_metrics)}"
            logger.debug(f"RAGAS partial failure - NaN metrics: {nan_metrics}")

        return result

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed after retries: {e}")
        failed_metrics = ["answer_relevancy", "faithfulness"]
        if reference_answer:
            failed_metrics.append("answer_correctness")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "error": str(e),
            "nan_metrics": failed_metrics,
        }


__all__ = ["evaluate_single"]
