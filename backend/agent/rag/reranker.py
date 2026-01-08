"""Reranker module for improving retrieval precision."""

from __future__ import annotations

import logging
from functools import lru_cache

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazy-load reranker postprocessor (singleton)."""
    from llama_index.core.postprocessor import SentenceTransformerRerank

    from backend.agent.rag.config import RERANKER_MODEL, RERANKER_TOP_K

    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    return SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANKER_TOP_K,
    )


def rerank_nodes(
    nodes: list[NodeWithScore],
    query: str,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using cross-encoder.

    Args:
        nodes: Retrieved nodes from vector search
        query: Original question
        top_k: Number of nodes to return (uses config default if None)

    Returns:
        Top-k nodes sorted by reranker score
    """
    if not nodes:
        return nodes

    from backend.agent.rag.config import RERANKER_TOP_K

    effective_top_k = top_k if top_k is not None else RERANKER_TOP_K

    if len(nodes) <= effective_top_k:
        return nodes  # No need to rerank if already under limit

    reranker = _get_reranker()

    # LlamaIndex's postprocessor handles everything
    from llama_index.core.schema import QueryBundle

    query_bundle = QueryBundle(query_str=query)

    reranked: list[NodeWithScore] = reranker.postprocess_nodes(nodes, query_bundle)

    logger.debug(f"Reranked {len(nodes)} nodes -> {len(reranked)}")
    return reranked


__all__ = [
    "rerank_nodes",
]
