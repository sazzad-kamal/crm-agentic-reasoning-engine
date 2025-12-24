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
- Audit logging for compliance

Usage:
    from backend.rag.pipeline import answer_question
    from backend.rag.retrieval import create_backend
    
    backend = create_backend()
    result = answer_question("What is an Opportunity?", backend)
    print(result["answer"])
"""

import logging
from typing import Optional, Callable

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.retrieval.base import RetrievalBackend
from backend.rag.config import get_config
from backend.rag.pipeline.constants import LLM_MODEL, ANSWER_MAX_TOKENS, MAX_CONTEXT_TOKENS
from backend.rag.utils import estimate_tokens, preprocess_query, extract_citations
from backend.rag.audit import AuditEntry, log_audit_entry
from backend.rag.prompts import (
    DOCS_QUERY_REWRITE_SYSTEM,
    DOCS_HYDE_SYSTEM,
    format_docs_answer_prompt,
)
from backend.rag.pipeline.base import (
    PipelineProgress,
    apply_lexical_gate,
    apply_per_doc_cap,
    build_context,
)
from backend.common.llm_client import call_llm_safe, call_llm_with_metrics


logger = logging.getLogger(__name__)


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
        system_prompt=DOCS_QUERY_REWRITE_SYSTEM,
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
        system_prompt=DOCS_HYDE_SYSTEM,
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
    prompt = format_docs_answer_prompt(context=context, question=question)
    
    logger.info(f"Generating answer for question: {question[:50]}...")
    result = call_llm_with_metrics(
        prompt=prompt,
        model=LLM_MODEL,
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
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
        company_id: Company ID for audit logging
        user_id: User ID for audit logging
        session_id: Session ID for audit logging
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
    config = get_config()
    progress = PipelineProgress(callback=progress_callback)
    
    # Initialize audit entry
    audit = AuditEntry(
        query=question,
        company_id=company_id,
        user_id=user_id,
        session_id=session_id,
    )
    
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
            audit.rewritten_query = rewritten_question
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
        audit.num_chunks_retrieved = len(scored_chunks)
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
        
        # Update audit entry with success
        audit.num_chunks_used = len(final_chunks)
        audit.answer_length = len(answer_result["answer"])
        audit.latency_ms = int(progress.total_elapsed_ms())
        audit.status = "success"
        audit.sources = used_doc_ids
        
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
        
        # Log audit entry
        log_audit_entry(audit)
        
        return result
        
    except Exception as e:
        # Log error to audit
        audit.status = "error"
        audit.error_message = str(e)
        audit.latency_ms = int(progress.total_elapsed_ms())
        log_audit_entry(audit)
        
        logger.error(f"Pipeline error: {e}")
        raise


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from backend.rag.retrieval import create_backend
    
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    backend = create_backend()
    
    question = "What is an Opportunity in Acme CRM Suite and what fields does it have?"
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = answer_question(question, backend, verbose=True, company_id="TEST")
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\n" + "-" * 60)
    print(f"Doc IDs used: {result['doc_ids_used']}")
    print(f"Cited docs: {result['cited_docs']}")
    print(f"Chunks used: {result['num_chunks_used']}")
    print(f"\nPipeline Steps:")
    for step in result['steps']:
        print(f"  [{step['status']}] {step['label']} ({step['elapsed_ms']:.0f}ms)")
    print(f"\nMetrics: {result['metrics']}")
