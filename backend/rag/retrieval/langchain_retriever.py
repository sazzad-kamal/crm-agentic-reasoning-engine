"""
LangChain-compatible retriever wrapper for the RAG backend.

This module wraps our custom RetrievalBackend in LangChain's BaseRetriever
interface, enabling:
- LangSmith tracing for all retrieval operations
- LCEL chain composition with | operator
- Standard LangChain callback support
"""

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from backend.rag.retrieval.base import RetrievalBackend, create_backend
from backend.rag.models import ScoredChunk


logger = logging.getLogger(__name__)


class AcmeCRMRetriever(BaseRetriever):
    """
    LangChain-compatible retriever wrapping the hybrid Qdrant + BM25 backend.

    This retriever appears in LangSmith traces and can be composed into LCEL chains.

    Usage:
        retriever = AcmeCRMRetriever()
        docs = retriever.invoke("How do I create an opportunity?")

        # Or in an LCEL chain:
        chain = retriever | format_docs | llm | parser
    """

    # Retrieval parameters
    k: int = Field(default=5, description="Number of documents to retrieve")
    k_dense: int = Field(default=20, description="Candidates from dense search")
    k_bm25: int = Field(default=20, description="Candidates from BM25 search")
    use_reranker: bool = Field(default=True, description="Whether to use cross-encoder reranking")
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")

    # Private backend instance
    _backend: RetrievalBackend | None = PrivateAttr(default=None)

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, backend: RetrievalBackend | None = None, **kwargs):
        """
        Initialize the retriever.

        Args:
            backend: Optional pre-initialized RetrievalBackend. If None, creates one.
            **kwargs: Retrieval parameters (k, k_dense, k_bm25, etc.)
        """
        super().__init__(**kwargs)
        self._backend = backend

    @property
    def backend(self) -> RetrievalBackend:
        """Lazy-load the retrieval backend."""
        if self._backend is None:
            logger.info("Initializing RetrievalBackend for LangChain retriever")
            self._backend = create_backend()
        return self._backend

    def _scored_chunk_to_document(self, scored: ScoredChunk) -> Document:
        """Convert a ScoredChunk to a LangChain Document."""
        return Document(
            page_content=scored.chunk.text,
            metadata={
                "chunk_id": scored.chunk.chunk_id,
                "doc_id": scored.chunk.doc_id,
                "title": scored.chunk.title,
                "dense_score": scored.dense_score,
                "bm25_score": scored.bm25_score,
                "rrf_score": scored.rrf_score,
                "rerank_score": scored.rerank_score,
                **scored.chunk.metadata,
            },
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant documents for the query.

        This method is called by LangChain and appears in LangSmith traces.

        Args:
            query: The search query
            run_manager: Callback manager for tracing

        Returns:
            List of LangChain Document objects
        """
        logger.debug(f"Retrieving documents for: {query[:50]}...")

        # Use the hybrid retrieval backend
        scored_chunks = self.backend.retrieve_candidates(
            query=query,
            k_dense=self.k_dense,
            k_bm25=self.k_bm25,
            top_n=self.k,
            use_reranker=self.use_reranker,
        )

        # Filter by score threshold if set
        if self.score_threshold > 0:
            scored_chunks = [
                s for s in scored_chunks
                if s.rerank_score >= self.score_threshold
            ]

        # Convert to LangChain Documents
        documents = [
            self._scored_chunk_to_document(scored)
            for scored in scored_chunks
        ]

        logger.info(f"Retrieved {len(documents)} documents for query")

        return documents


# =============================================================================
# Factory Functions
# =============================================================================

def create_langchain_retriever(
    k: int = 5,
    use_reranker: bool = True,
    backend: RetrievalBackend | None = None,
) -> AcmeCRMRetriever:
    """
    Create a LangChain-compatible retriever.

    Args:
        k: Number of documents to retrieve
        use_reranker: Whether to use cross-encoder reranking
        backend: Optional pre-initialized backend

    Returns:
        AcmeCRMRetriever instance
    """
    return AcmeCRMRetriever(
        backend=backend,
        k=k,
        use_reranker=use_reranker,
    )


__all__ = [
    "AcmeCRMRetriever",
    "create_langchain_retriever",
]
