"""Semantic retriever for Act! CRM documentation.

Retrieves relevant documentation chunks and synthesizes answers
grounded in the source material.
"""

import logging
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode

from backend.agent.rag.indexer import get_index

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG retrieval and synthesis."""

    answer: str
    sources: list[dict]
    confidence: float


def retrieve_and_answer(question: str, top_k: int = 5) -> RAGResult:
    """Retrieve relevant docs and synthesize an answer.

    Args:
        question: User's question about Act! CRM
        top_k: Number of chunks to retrieve

    Returns:
        RAGResult with answer and source citations
    """
    try:
        index: VectorStoreIndex = get_index()

        # Create query engine with citation mode
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode=ResponseMode.COMPACT,
            streaming=False,
        )

        # Query the index
        response = query_engine.query(question)

        # Extract sources
        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                "source": node.metadata.get("source", "Unknown"),
                "score": round(node.score, 3) if node.score else 0.0,
            })

        # Calculate confidence based on retrieval scores
        avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0.0
        confidence = min(avg_score, 1.0)

        logger.info(
            f"[RAG] Retrieved {len(sources)} chunks, "
            f"avg_score={avg_score:.3f}, answer_len={len(str(response))}"
        )

        return RAGResult(
            answer=str(response),
            sources=sources,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return RAGResult(
            answer="I couldn't find relevant information in the Act! documentation.",
            sources=[],
            confidence=0.0,
        )


def search_docs(query: str, top_k: int = 3) -> list[dict]:
    """Search documentation without synthesis (retrieval only).

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        List of matching document chunks with metadata
    """
    try:
        index = get_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "source": node.metadata.get("source", "Unknown"),
                "page": node.metadata.get("page", None),
                "score": round(node.score, 3) if node.score else 0.0,
            })

        return results

    except Exception as e:
        logger.error(f"[RAG] Search failed: {e}")
        return []


__all__ = ["retrieve_and_answer", "search_docs", "RAGResult"]
