"""
Model preloading for faster first query response.

Loads embedding and reranker models at startup to eliminate cold start latency.
Called from main.py during application startup.

Usage:
    from backend.rag.retrieval.preload import preload_models
    preload_models()  # Call during startup
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.rag.retrieval.constants import EMBEDDING_MODEL, RERANKER_MODEL


logger = logging.getLogger(__name__)


# Cached model instances
_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None


def get_embedding_model() -> SentenceTransformer:
    """Get the cached embedding model, loading if necessary."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """Get the cached reranker model, loading if necessary."""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def preload_models(parallel: bool = True) -> dict:
    """
    Preload embedding and reranker models at startup.

    Args:
        parallel: Load models in parallel (faster) or sequentially

    Returns:
        Dict with load times for each model
    """
    logger.info("Preloading RAG models...")
    start = time.time()
    results = {}

    def load_embedding():
        t0 = time.time()
        get_embedding_model()
        return time.time() - t0

    def load_reranker():
        t0 = time.time()
        get_reranker_model()
        return time.time() - t0

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            embedding_future = executor.submit(load_embedding)
            reranker_future = executor.submit(load_reranker)

            results["embedding_ms"] = int(embedding_future.result() * 1000)
            results["reranker_ms"] = int(reranker_future.result() * 1000)
    else:
        results["embedding_ms"] = int(load_embedding() * 1000)
        results["reranker_ms"] = int(load_reranker() * 1000)

    total_ms = int((time.time() - start) * 1000)
    results["total_ms"] = total_ms

    logger.info(
        f"Models preloaded in {total_ms}ms "
        f"(embedding: {results['embedding_ms']}ms, reranker: {results['reranker_ms']}ms)"
    )

    return results


def is_preloaded() -> bool:
    """Check if models are already loaded."""
    return _embedding_model is not None and _reranker_model is not None


__all__ = [
    "preload_models",
    "get_embedding_model",
    "get_reranker_model",
    "is_preloaded",
]
