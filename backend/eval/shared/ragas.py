"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import random
import sys
import threading
import time
import warnings
from typing import Any

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from backend.core.llm import EMBEDDING_MODEL, FAST_MODEL

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

# Thread-safe singleton instances for RAGAS LLM/embeddings
_ragas_llm = None
_ragas_embeddings = None
_ragas_lock = threading.Lock()


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


def _get_ragas_llm() -> Any:
    """Get shared LLM for RAGAS (thread-safe singleton)."""
    global _ragas_llm
    if _ragas_llm is None:
        with _ragas_lock:
            if _ragas_llm is None:
                logger.info(f"Initializing RAGAS LLM ({FAST_MODEL})")
                _ragas_llm = LangchainLLMWrapper(ChatOpenAI(model=FAST_MODEL, temperature=0))
    return _ragas_llm


def _get_ragas_embeddings() -> Any:
    """Get shared embeddings for RAGAS (thread-safe singleton)."""
    global _ragas_embeddings
    if _ragas_embeddings is None:
        with _ragas_lock:
            if _ragas_embeddings is None:
                logger.info(f"Initializing RAGAS embeddings ({EMBEDDING_MODEL})")
                _ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    return _ragas_embeddings


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
    # Suppress RAGAS output unless verbose
    if not verbose:
        logging.getLogger("ragas").setLevel(logging.ERROR)

    # RAGAS requires non-empty contexts
    if not contexts:
        contexts = ["No context provided"]

    # Old-style RAGAS metrics use different column names
    dataset_dict: dict[str, Any] = {
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
    }

    # Get SHARED LLM and embeddings instances (thread-safe singletons)
    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    # All metrics share the same LLM/embeddings instances
    metrics: list[Any] = [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
    ]

    if reference_answer:
        dataset_dict["reference"] = [reference_answer]
        metrics.extend([
            ContextRecall(llm=llm),
            AnswerCorrectness(llm=llm),
        ])

    dataset = Dataset.from_dict(dataset_dict)

    # Retry logic with exponential backoff for transient failures
    max_retries = 3
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            # Metrics already have llm/embeddings set via constructor (thread-safe)
            eval_result = evaluate(
                dataset,
                metrics=metrics,
                show_progress=False,  # Suppress tqdm progress bars
            )

            # Convert to pandas DataFrame
            df = eval_result.to_pandas()  # type: ignore[union-attr]

            # Track which metrics returned NaN (internal RAGAS failure)
            nan_metrics: list[str] = []

            def get_score(
                name: str,
                _df: Any = df,
                _nan_metrics: list[str] = nan_metrics,
            ) -> float:
                if name in _df.columns and len(_df) > 0:
                    val = _df[name].iloc[0]
                    if val is None or (isinstance(val, float) and val != val):  # Check for NaN
                        _nan_metrics.append(name)
                        return 0.0
                    return float(val)
                return 0.0

            result: dict[str, float | str | list[str] | None] = {
                "answer_relevancy": get_score("answer_relevancy"),
                "faithfulness": get_score("faithfulness"),
                "context_precision": get_score("context_precision"),
                "context_recall": get_score("context_recall"),
                "answer_correctness": get_score("answer_correctness"),
                "error": None,
                "nan_metrics": nan_metrics,  # Track which metrics returned NaN
            }

            # If any metrics returned NaN, retry (might be transient JSON parsing failure)
            if nan_metrics and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.debug(f"RAGAS returned NaN for {nan_metrics}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            # If any metrics returned NaN on final attempt, mark as partial failure
            if nan_metrics:
                result["error"] = f"RAGAS returned NaN for: {', '.join(nan_metrics)}"
                logger.debug(f"RAGAS partial failure - NaN metrics: {nan_metrics}")

            return result

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.debug(f"RAGAS evaluation failed: {last_error}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.warning(f"RAGAS evaluation failed after {max_retries} attempts: {last_error}")

    # All retries exhausted
    all_metrics = ["answer_relevancy", "faithfulness", "context_precision", "context_recall", "answer_correctness"]
    return {
        "answer_relevancy": 0.0,
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "error": last_error or "RAGAS evaluation failed",
        "nan_metrics": all_metrics,  # All metrics failed
    }


__all__ = ["evaluate_single"]
