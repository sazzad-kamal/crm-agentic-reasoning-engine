"""
RAG Pipeline for Acme CRM documentation.

Implements:
- Query preprocessing and normalization
- Query rewriting (clarification)
- HyDE (Hypothetical Document Embeddings)
- Retrieval with gating and per-doc caps
- Context building
- Answer generation with citations
- Progress step logging

Usage:
    from backend.rag.pipeline import answer_question
    from backend.rag.retrieval import create_backend
    
    backend = create_backend()
    result = answer_question("What is an Opportunity?", backend)
    print(result["answer"])
"""

import logging
from typing import Optional, Callable

from backend.common.models import DocumentChunk
from backend.rag.retrieval.base import RetrievalBackend
from backend.rag.pipeline.constants import ANSWER_MODEL, ANSWER_MAX_TOKENS, MAX_CONTEXT_TOKENS
from backend.rag.pipeline.utils import estimate_tokens, preprocess_query, extract_citations
from backend.rag.pipeline.base import PipelineProgress
from backend.rag.pipeline.gating import apply_lexical_gate, apply_per_doc_cap
from backend.common.context_builder import build_context
from backend.common.llm_client import call_llm_safe, call_llm_with_metrics


logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

_QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for a CRM documentation search system.
Your job is to take a user's question about Acme CRM Suite and rewrite it to be clearer and more specific.
Keep the rewritten query in natural language (not keywords).
If the query is already clear, return it mostly unchanged.
Only output the rewritten query, nothing else."""

_HYDE_SYSTEM = """You are an expert on Acme CRM Suite documentation.
Given a question, write a short hypothetical answer (2-3 sentences) as if it came from the documentation.
This will be used for semantic search, so include relevant terminology and concepts.
Only output the hypothetical answer, nothing else."""

_ANSWER_SYSTEM = """You are an AI assistant answering questions about Acme CRM Suite.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this documented in the provided sources."
3. Cite your sources using [doc_id] format, e.g., [opportunities_pipeline_and_forecasts].
4. Be concise but complete.
5. If multiple docs cover different aspects, synthesize the information and cite all relevant sources.

Context from Acme CRM Suite documentation:
{context}

Question: {question}

Answer (with citations):"""


def _format_answer_prompt(context: str, question: str) -> str:
    """Format the documentation answer prompt."""
    return _ANSWER_SYSTEM.format(context=context, question=question)


# =============================================================================
# Pipeline Components
# =============================================================================

def rewrite_query(query: str) -> str:
    """
    Use LLM to rewrite vague queries into clearer ones.
    
    Args:
        query: Original user query
        
    Returns:
        Rewritten query (or original if rewriting fails)
    """
    logger.debug(f"Rewriting query: {query[:50]}...")
    rewritten = call_llm_safe(
        prompt=f"Rewrite this CRM question to be clearer: {query}",
        system_prompt=_QUERY_REWRITE_SYSTEM,
        max_tokens=150,
        default=query,
    )
    if rewritten != query:
        logger.debug(f"Query rewritten to: {rewritten[:50]}...")
    return rewritten


def generate_hyde_answer(query: str) -> str:
    """
    Generate a hypothetical answer for HyDE retrieval.
    
    Args:
        query: The user's question
        
    Returns:
        A hypothetical answer to use for embedding
    """
    logger.debug(f"Generating HyDE answer for: {query[:50]}...")
    hyde = call_llm_safe(
        prompt=f"Question: {query}",
        system_prompt=_HYDE_SYSTEM,
        max_tokens=200,
        default="",
    )
    if hyde:
        logger.debug(f"HyDE generated: {hyde[:50]}...")
    return hyde


def generate_answer(
    question: str,
    context: str,
    chunks_used: list[DocumentChunk],
) -> dict:
    """
    Generate an answer using the LLM with the provided context.
    
    Args:
        question: The user's question
        context: Formatted context string
        chunks_used: List of chunks used (for metadata)
        
    Returns:
        Dict with answer and metadata
    """
    prompt = _format_answer_prompt(context=context, question=question)
    
    logger.info(f"Generating answer for question: {question[:50]}...")
    result = call_llm_with_metrics(
        prompt=prompt,
        model=ANSWER_MODEL,  # Use best model for user-facing answers
        max_tokens=ANSWER_MAX_TOKENS,
    )
    
    logger.info(f"Answer generated in {result['latency_ms']:.0f}ms, {result['total_tokens']} tokens")
    
    return {
        "answer": result["response"],
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "cached": result.get("cached", False),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def answer_question(
    question: str,
    backend: RetrievalBackend,
    *,
    k: int = 8,
    use_hyde: bool = True,
    use_rewrite: bool = None,  # None = use config
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
) -> dict:
    """
    Full RAG pipeline to answer a question.
    
    Args:
        question: User's question
        backend: Initialized RetrievalBackend
        k: Number of chunks to retrieve
        use_hyde: Whether to use HyDE for retrieval
        use_rewrite: Whether to rewrite the query (None = use config)
        verbose: Print debug information
        progress_callback: Optional callback for step progress (step_id, label, elapsed_ms)
        
    Returns:
        Dict containing:
        - answer: The generated answer
        - used_chunks: List of DocumentChunk objects used
        - rewritten_question: The rewritten query
        - hyde_answer: The hypothetical answer (if used)
        - doc_ids_used: List of unique doc_ids cited
        - metrics: Dict with latency, token counts, etc.
        - steps: List of processing steps with timing
    """
    progress = PipelineProgress(callback=progress_callback)
    
    # Default to True for rewrite
    if use_rewrite is None:
        use_rewrite = True
    
    metrics = {
        "total_latency_ms": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }
    
    try:
        logger.info(f"Processing question: {question[:80]}...")
        
        # Step 1: Preprocess
        progress.start_step("preprocess", "Preprocessing query")
        question = preprocess_query(question)
        progress.complete_step("preprocess", "Query preprocessed")
        
        if verbose:
            print(f"Original question: {question}")
        
        # Step 2: Query rewrite
        progress.start_step("rewrite", "Analyzing query")
        rewritten_question = question
        if use_rewrite:
            rewritten_question = rewrite_query(question)
            metrics["llm_calls"] += 1
            if verbose:
                print(f"Rewritten question: {rewritten_question}")
        progress.complete_step("rewrite", "Query analyzed")
        
        # Step 3: HyDE
        progress.start_step("hyde", "Generating search context")
        hyde_answer = ""
        retrieval_query = rewritten_question
        if use_hyde:
            hyde_answer = generate_hyde_answer(rewritten_question)
            metrics["llm_calls"] += 1
            if hyde_answer:
                retrieval_query = f"{rewritten_question} {hyde_answer}"
            if verbose:
                print(f"HyDE answer: {hyde_answer[:100]}...")
        progress.complete_step("hyde", "Search context ready")
        
        # Step 4: Retrieval
        progress.start_step("retrieval", "Searching documents")
        scored_chunks = backend.retrieve_candidates(
            query=retrieval_query,
            k_dense=k * 3,
            k_bm25=k * 3,
            top_n=k * 2,
            use_reranker=True,
        )
        progress.complete_step("retrieval", f"Found {len(scored_chunks)} relevant sections")
        
        logger.debug(f"Retrieved {len(scored_chunks)} candidates after reranking")
        if verbose:
            print(f"Retrieved {len(scored_chunks)} candidates after reranking")
        
        # Step 5: Gating and filtering
        progress.start_step("filter", "Filtering results")
        gated_chunks = apply_lexical_gate(scored_chunks)
        if verbose:
            print(f"After lexical gate: {len(gated_chunks)} chunks")
        
        capped_chunks = apply_per_doc_cap(gated_chunks)
        if verbose:
            print(f"After per-doc cap: {len(capped_chunks)} chunks")
        
        final_chunks = [sc.chunk for sc in capped_chunks[:k]]
        progress.complete_step("filter", f"Selected {len(final_chunks)} best matches")
        
        if verbose:
            print(f"Final chunks: {len(final_chunks)}")
            for i, chunk in enumerate(final_chunks):
                print(f"  {i+1}. [{chunk.doc_id}] {chunk.text[:50]}...")
        
        # Step 6: Build context
        progress.start_step("context", "Building context")
        context = build_context(final_chunks, max_tokens=MAX_CONTEXT_TOKENS)
        progress.complete_step("context", "Context prepared")
        
        if verbose:
            print(f"Context size: ~{estimate_tokens(context)} tokens")
        
        # Step 7: Generate answer
        progress.start_step("generate", "Generating answer")
        answer_result = generate_answer(question, context, final_chunks)
        metrics["llm_calls"] += 1
        metrics["total_latency_ms"] += answer_result["latency_ms"]
        metrics["total_tokens"] += answer_result["total_tokens"]
        progress.complete_step("generate", "Answer generated")
        
        # Extract citations using utility function
        cited_docs = extract_citations(answer_result["answer"])
        
        # Get all doc_ids from used chunks
        used_doc_ids = list(set(c.doc_id for c in final_chunks))
        
        logger.info(f"Question answered: {len(final_chunks)} chunks used, {len(cited_docs)} citations")
        
        result = {
            "answer": answer_result["answer"],
            "used_chunks": final_chunks,
            "rewritten_question": rewritten_question,
            "hyde_answer": hyde_answer,
            "doc_ids_used": used_doc_ids,
            "cited_docs": cited_docs,
            "num_chunks_used": len(final_chunks),
            "context_tokens": estimate_tokens(context),
            "steps": progress.get_steps(),
            "metrics": {
                "answer_latency_ms": answer_result["latency_ms"],
                "total_latency_ms": int(progress.total_elapsed_ms()),
                "prompt_tokens": answer_result["prompt_tokens"],
                "completion_tokens": answer_result["completion_tokens"],
                "total_tokens": answer_result["total_tokens"],
                "llm_calls": metrics["llm_calls"],
                "cached": answer_result.get("cached", False),
            },
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
