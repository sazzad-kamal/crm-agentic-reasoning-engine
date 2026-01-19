"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import math
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, cast

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.llm import (
    EMBEDDING_MODEL,
    FAST_MODEL,
    get_langchain_chat_openai,
    get_langchain_embeddings,
)

# Import metric CLASSES (not singleton instances) for thread-safe instantiation
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_ragas_logging() -> Generator[None, None, None]:
    """Temporarily suppress RAGAS logging, restoring original level on exit."""
    ragas_logger = logging.getLogger("ragas")
    original_level = ragas_logger.level
    ragas_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        ragas_logger.setLevel(original_level)


# Suppress "Event loop is closed" errors from httpx/aiohttp during cleanup
# These are harmless but flood the logs on Windows
def _suppress_event_loop_closed_errors() -> None:
    """Install custom handlers to suppress event loop closed errors."""
    import asyncio

    # Suppress via excepthook (catches uncaught exceptions)
    original_excepthook = sys.excepthook

    def custom_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        if isinstance(exc_value, RuntimeError) and "Event loop is closed" in str(exc_value):
            return  # Suppress this error
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = custom_excepthook

    # Suppress via asyncio exception handler (catches async errors)
    def silent_exception_handler(loop: Any, context: dict) -> None:
        exception = context.get("exception")
        if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
            return  # Suppress this error
        # Fall back to default for other errors
        loop.default_exception_handler(context)

    # Set the handler on the current event loop if one exists
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silent_exception_handler)
    except RuntimeError:
        pass  # No event loop running yet

    # Add logging filter to suppress these errors from httpx/asyncio loggers
    class EventLoopClosedFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = str(record.getMessage())
            return "Event loop is closed" not in msg

    for logger_name in ["asyncio", "httpx", "httpcore", "aiohttp"]:
        logging.getLogger(logger_name).addFilter(EventLoopClosedFilter())

    # Suppress RAGAS executor errors (APIConnectionError, etc.) - we track failures separately
    class RagasExecutorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = str(record.getMessage())
            # Suppress "Exception raised in Job[X]" messages from ragas.executor
            return "Exception raised in Job" not in msg

    logging.getLogger("ragas.executor").addFilter(RagasExecutorFilter())


_suppress_event_loop_closed_errors()


@lru_cache
def _get_ragas_llm() -> Any:
    """Get shared LLM for RAGAS (cached singleton)."""
    logger.info(f"Initializing RAGAS LLM ({FAST_MODEL})")
    return LangchainLLMWrapper(get_langchain_chat_openai())


@lru_cache
def _get_ragas_embeddings() -> Any:
    """Get shared embeddings for RAGAS (cached singleton)."""
    logger.info(f"Initializing RAGAS embeddings ({EMBEDDING_MODEL})")
    return LangchainEmbeddingsWrapper(get_langchain_embeddings())


@lru_cache
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
        ContextPrecision(llm=llm),
    )
    if include_reference:
        return base + (ContextRecall(llm=llm), AnswerCorrectness(llm=llm))
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
        "context_precision": get_score("context_precision"),
        "context_recall": get_score("context_recall"),
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
    # RAGAS requires non-empty contexts
    if not contexts:
        contexts = ["No context provided"]

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
            with _suppress_ragas_logging():
                result = _run_evaluation_with_retry(dataset, metrics)

        nan_metrics = cast(list[str], result.get("nan_metrics", []))
        if nan_metrics:
            result["error"] = f"RAGAS returned NaN for: {', '.join(nan_metrics)}"
            logger.debug(f"RAGAS partial failure - NaN metrics: {nan_metrics}")

        return result

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed after retries: {e}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "error": str(e),
            "nan_metrics": ["answer_relevancy", "faithfulness", "context_precision", "context_recall", "answer_correctness"],
        }


__all__ = ["evaluate_single"]
