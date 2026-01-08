"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)


def _is_mock_mode() -> bool:
    """Check if MOCK_LLM mode is enabled."""
    return os.environ.get("MOCK_LLM", "0") == "1"


def _mock_evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
) -> dict[str, float | str | None]:
    """Return mock RAGAS scores for testing without OpenAI API."""
    # Return realistic mock scores based on content presence
    has_answer = bool(answer and len(answer) > 10)
    has_context = bool(contexts and contexts[0] != "No context provided")

    if has_answer and has_context:
        return {
            "answer_relevancy": 0.85,
            "faithfulness": 0.80,
            "context_precision": 0.75,
            "context_recall": 0.70 if reference_answer else 0.0,
            "answer_correctness": 0.65 if reference_answer else 0.0,
            "error": None,
        }
    elif has_answer:
        return {
            "answer_relevancy": 0.70,
            "faithfulness": 0.50,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.40 if reference_answer else 0.0,
            "error": None,
        }
    else:
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "error": None,
        }


# Only import RAGAS dependencies when not in mock mode
if not _is_mock_mode():
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

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


def _get_ragas_llm() -> Any:
    """Get LLM for RAGAS using LangChain wrapper."""
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def _get_ragas_embeddings() -> Any:
    """Get embeddings for RAGAS using LangChain wrapper."""
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    verbose: bool = False,
) -> dict[str, float | str | None]:
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
    # Return mock scores in mock mode (no OpenAI API needed)
    if _is_mock_mode():
        return _mock_evaluate_single(question, answer, contexts, reference_answer)

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

    # Get LLM and embeddings factories for thread-safe metric instantiation
    def llm_factory() -> Any:
        return _get_ragas_llm()

    def embeddings_factory() -> Any:
        return _get_ragas_embeddings()

    # Create fresh metric instances per call for thread safety
    # RAGAS 0.4.x metrics require llm to be passed via llm_factory
    metrics: list[Any] = [
        AnswerRelevancy(llm=llm_factory(), embeddings=embeddings_factory()),
        Faithfulness(llm=llm_factory()),
    ]

    # Add context precision always (doesn't need reference)
    metrics.append(ContextPrecision(llm=llm_factory()))

    if reference_answer:
        dataset_dict["reference"] = [reference_answer]
        metrics.extend([
            ContextRecall(llm=llm_factory()),
            AnswerCorrectness(llm=llm_factory()),
        ])

    dataset = Dataset.from_dict(dataset_dict)

    try:
        # Metrics already have llm/embeddings set via constructor (thread-safe)
        eval_result = evaluate(
            dataset,
            metrics=metrics,
        )

        # Convert to pandas DataFrame
        df = eval_result.to_pandas()  # type: ignore[union-attr]

        def get_score(name: str) -> float:
            if name in df.columns and len(df) > 0:
                val = df[name].iloc[0]
                if val is None or (isinstance(val, float) and val != val):  # Check for NaN
                    return 0.0
                return float(val)
            return 0.0

        return {
            "answer_relevancy": get_score("answer_relevancy"),
            "faithfulness": get_score("faithfulness"),
            "context_precision": get_score("context_precision"),
            "context_recall": get_score("context_recall"),
            "answer_correctness": get_score("answer_correctness"),
            "error": None,  # Success
        }
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"RAGAS evaluation failed: {error_msg}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "error": error_msg,  # Track the failure
        }


__all__ = ["evaluate_single"]
