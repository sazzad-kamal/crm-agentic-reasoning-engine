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

# Import metric CLASSES (not singleton instances) for thread-safe instantiation
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, Faithfulness

logger = logging.getLogger(__name__)


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

    if reference_answer:
        dataset_dict["reference"] = [reference_answer]
        metrics.extend([
            ContextPrecision(llm=llm_factory()),
            AnswerCorrectness(llm=llm_factory()),
        ])

    dataset = Dataset.from_dict(dataset_dict)

    try:
        # Metrics already have llm/embeddings set via constructor (thread-safe)
        result = evaluate(
            dataset,
            metrics=metrics,
        )

        # Convert to pandas DataFrame
        df = result.to_pandas()  # type: ignore[union-attr]

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
