"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# Use old-style metrics (deprecated but work with evaluate())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, faithfulness

logger = logging.getLogger(__name__)


def _get_ragas_llm() -> LangchainLLMWrapper:
    """Get LLM for RAGAS using LangChain wrapper."""
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Get embeddings for RAGAS using LangChain wrapper."""
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    verbose: bool = False,
) -> dict[str, float]:
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

    # Get LLM and embeddings
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    # Select metrics based on available data
    # context_precision and answer_correctness require reference (ground truth)
    metrics: list[Any] = [answer_relevancy, faithfulness]

    if reference_answer:
        dataset_dict["reference"] = [reference_answer]
        metrics.extend([context_precision, answer_correctness])

    dataset = Dataset.from_dict(dataset_dict)

    try:
        # Pass llm and embeddings to evaluate() for old-style metrics
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        # Convert to pandas DataFrame
        df = result.to_pandas()

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
            "answer_correctness": get_score("answer_correctness"),
        }
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "answer_correctness": 0.0,
        }


__all__ = ["evaluate_single"]
