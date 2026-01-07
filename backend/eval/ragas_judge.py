"""RAGAS-based evaluation for RAG quality metrics."""

from __future__ import annotations

import logging
import os
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, faithfulness

logger = logging.getLogger(__name__)


def _get_ragas_llm() -> Any:
    """Get LLM for RAGAS using native factory."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return llm_factory("gpt-4o-mini", client=client)


def _get_ragas_embeddings() -> Any:
    """Get embeddings for RAGAS using native class."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return RagasOpenAIEmbeddings(client=client, model="text-embedding-3-small")


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    Args:
        question: The user's question
        answer: The agent's answer
        contexts: List of retrieved context strings
        reference_answer: Optional ground truth answer for answer_correctness metric

    Returns:
        dict with answer_relevancy, faithfulness, context_precision, answer_correctness (0.0-1.0)
    """
    # RAGAS requires non-empty contexts
    if not contexts:
        contexts = ["No context provided"]

    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    # Configure metrics with explicit LLM/embeddings (required in RAGAS 0.4.x)
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    # Select metrics - add answer_correctness if reference provided
    metrics = [answer_relevancy, faithfulness, context_precision]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    if reference_answer:
        dataset_dict["ground_truth"] = [reference_answer]
        answer_correctness.llm = ragas_llm
        metrics.append(answer_correctness)

    dataset = Dataset.from_dict(dataset_dict)

    try:
        result = evaluate(dataset, metrics=metrics)

        # RAGAS 0.4.x returns lists (one score per sample) - extract first element
        def get_score(name: str) -> float:
            val = result.get(name, 0.0)  # type: ignore[union-attr]
            if isinstance(val, list):
                return float(val[0]) if val else 0.0
            return float(val) if val else 0.0

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
