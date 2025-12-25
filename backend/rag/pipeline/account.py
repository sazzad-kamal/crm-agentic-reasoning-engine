"""
Account-aware RAG pipeline for private CRM text (MVP2).

Answers questions scoped to a specific company/account using:
- Private CRM text (history, opportunity notes, attachments)
- Optionally, global product docs

Usage:
    from backend.rag.pipeline import answer_account_question
    
    result = answer_account_question(
        "What's the status with Acme?",
        company_name="Acme Manufacturing"
    )
    print(result["answer"])
"""

import re
import logging
import time

import pandas as pd

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.retrieval.private import PrivateRetrievalBackend, create_private_backend
from backend.rag.retrieval.base import create_backend as create_docs_backend
from backend.rag.pipeline.constants import MAX_CONTEXT_TOKENS
from backend.rag.pipeline.utils import estimate_tokens, preprocess_query, extract_citations
from backend.rag.pipeline.base import build_private_context, build_docs_context
from backend.rag.pipeline.prompts import (
    QUERY_REWRITE_SYSTEM,
    HYDE_SYSTEM,
    format_account_answer_prompt,
)
from backend.rag.pipeline.company import load_companies_df
from backend.common.llm_client import call_llm_safe, call_llm_with_metrics


logger = logging.getLogger(__name__)


# =============================================================================
# Cost Configuration
# =============================================================================

# GPT-4.1-mini pricing (per token)
COST_PER_INPUT_TOKEN = 0.40 / 1_000_000
COST_PER_OUTPUT_TOKEN = 1.60 / 1_000_000


# =============================================================================
# Company Resolution
# =============================================================================

# load_companies_df imported from backend.rag.pipeline.company


def resolve_company_id(
    company_id: str | None = None,
    company_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve company_id from either company_id or company_name.
    
    Args:
        company_id: Direct company ID
        company_name: Company name to search for
        
    Returns:
        Tuple of (company_id, company_name)
        
    Raises:
        ValueError: If company cannot be resolved
    """
    if company_id:
        # Validate company_id exists
        df = load_companies_df()
        match = df[df["company_id"] == company_id]
        if match.empty:
            raise ValueError(f"Company ID '{company_id}' not found in companies.csv")
        return company_id, match.iloc[0]["name"]
    
    if company_name:
        df = load_companies_df()
        
        # Case-insensitive contains match
        name_lower = company_name.lower()
        matches = df[df["name"].str.lower().str.contains(name_lower, na=False)]
        
        if matches.empty:
            # Try matching company_id
            matches = df[df["company_id"].str.lower().str.contains(name_lower, na=False)]
        
        if matches.empty:
            raise ValueError(
                f"No company found matching '{company_name}'. "
                f"Available companies: {df['name'].tolist()}"
            )
        
        if len(matches) > 1:
            # Pick best match (shortest name that contains query)
            matches = matches.copy()
            matches["_score"] = matches["name"].str.len()
            matches = matches.sort_values("_score")
            logger.warning(f"Multiple companies match '{company_name}', using: {matches.iloc[0]['name']}")
        
        row = matches.iloc[0]
        return row["company_id"], row["name"]
    
    raise ValueError("Must provide either company_id or company_name")


# =============================================================================
# Query Enhancement
# =============================================================================

def rewrite_query(query: str, company_name: str) -> str:
    """Rewrite query to be more specific."""
    logger.debug(f"Rewriting query for {company_name}: {query[:50]}...")
    return call_llm_safe(
        prompt=f"Question about {company_name}: {query}",
        system_prompt=QUERY_REWRITE_SYSTEM,
        max_tokens=150,
        default=query,
    )


def generate_hyde(query: str, company_name: str) -> str:
    """Generate hypothetical answer for HyDE."""
    logger.debug(f"Generating HyDE for {company_name}: {query[:50]}...")
    return call_llm_safe(
        prompt=f"Question about {company_name}: {query}",
        system_prompt=HYDE_SYSTEM,
        max_tokens=200,
        default="",
    )


# =============================================================================
# Answer Generation
# =============================================================================

def generate_answer(question: str, context: str) -> dict:
    """Generate answer using LLM."""
    prompt = format_account_answer_prompt(context=context, question=question)
    
    result = call_llm_with_metrics(
        prompt=prompt,
        model="gpt-4.1-mini",
        max_tokens=800,
    )
    
    return {
        "answer": result["response"],
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
    }


# =============================================================================
# Main Pipeline
# =============================================================================

# Global backend cache
_private_backend: PrivateRetrievalBackend | None = None
_docs_backend = None


def get_private_backend() -> PrivateRetrievalBackend:
    """Get or create private retrieval backend."""
    global _private_backend
    if _private_backend is None:
        _private_backend = create_private_backend()
    return _private_backend


def get_docs_backend():
    """Get or create docs retrieval backend (optional)."""
    global _docs_backend
    if _docs_backend is None:
        try:
            _docs_backend = create_docs_backend()
        except Exception as e:
            print(f"Warning: Could not load docs backend: {e}")
            _docs_backend = False  # Mark as unavailable
    return _docs_backend if _docs_backend else None


def answer_account_question(
    question: str,
    company_id: str | None = None,
    company_name: str | None = None,
    *,
    config: dict | None = None,
    include_docs: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Answer a question scoped to a specific company/account.
    
    Args:
        question: The question to answer
        company_id: Company ID to scope to
        company_name: Company name to look up
        config: Optional configuration overrides
        include_docs: Whether to also search product docs
        verbose: Print debug info
        
    Returns:
        Dict with answer, sources, raw hits, and metrics
    """
    start_time = time.time()
    config = config or {}
    
    # ---------------------------------------------------------------------------
    # 1. Resolve company
    # ---------------------------------------------------------------------------
    resolved_id, resolved_name = resolve_company_id(company_id, company_name)
    
    if verbose:
        print(f"Resolved company: {resolved_name} ({resolved_id})")
    
    # ---------------------------------------------------------------------------
    # 2. Query enhancement
    # ---------------------------------------------------------------------------
    rewritten = rewrite_query(question, resolved_name)
    hyde_text = generate_hyde(question, resolved_name)
    
    retrieval_query = f"{rewritten} {hyde_text}".strip()
    
    if verbose:
        print(f"Rewritten: {rewritten}")
        print(f"HyDE: {hyde_text[:80]}...")
    
    # ---------------------------------------------------------------------------
    # 3. Retrieve from private index (filtered)
    # ---------------------------------------------------------------------------
    private_backend = get_private_backend()
    
    k = config.get("k", 10)
    private_hits = private_backend.retrieve_candidates(
        query=retrieval_query,
        k_dense=k * 2,
        k_bm25=k * 2,
        top_n=k,
        use_reranker=True,
        company_filter=resolved_id,
    )
    
    if verbose:
        print(f"Private hits: {len(private_hits)}")
    
    # ---------------------------------------------------------------------------
    # 4. Optionally retrieve from docs index (unfiltered)
    # ---------------------------------------------------------------------------
    doc_hits = []
    if include_docs:
        docs_backend = get_docs_backend()
        if docs_backend:
            doc_hits = docs_backend.retrieve_candidates(
                query=retrieval_query,
                k_dense=k,
                k_bm25=k,
                top_n=5,
                use_reranker=True,
            )
            if verbose:
                print(f"Doc hits: {len(doc_hits)}")
    
    # ---------------------------------------------------------------------------
    # 5. Build context
    # ---------------------------------------------------------------------------
    private_context, private_sources = build_private_context(
        private_hits, resolved_id, max_tokens=MAX_CONTEXT_TOKENS
    )
    
    doc_context, doc_sources = "", []
    if doc_hits:
        doc_context, doc_sources = build_docs_context(doc_hits, max_tokens=1000)
    
    full_context = private_context
    if doc_context:
        full_context += "\n\n" + doc_context
    
    if verbose:
        print(f"Context tokens: ~{estimate_tokens(full_context)}")
    
    # ---------------------------------------------------------------------------
    # 6. Generate answer
    # ---------------------------------------------------------------------------
    answer_result = generate_answer(question, full_context)
    
    total_time = (time.time() - start_time) * 1000
    
    # ---------------------------------------------------------------------------
    # 7. Build response
    # ---------------------------------------------------------------------------
    # Extract cited sources from answer
    cited = set(re.findall(r'\[([^\]]+)\]', answer_result["answer"]))
    
    # Estimate cost
    prompt_tokens = answer_result["prompt_tokens"]
    completion_tokens = answer_result["completion_tokens"]
    estimated_cost = (
        prompt_tokens * COST_PER_INPUT_TOKEN +
        completion_tokens * COST_PER_OUTPUT_TOKEN
    )
    
    return {
        "answer": answer_result["answer"],
        "company_id": resolved_id,
        "company_name": resolved_name,
        "sources": private_sources + doc_sources,
        "cited_sources": list(cited),
        "raw_private_hits": [
            {
                "id": sc.chunk.metadata.get("source_id", sc.chunk.doc_id),
                "company_id": sc.chunk.metadata.get("company_id", ""),
                "type": sc.chunk.metadata.get("type", ""),
                "score": sc.rerank_score,
                "text_preview": sc.chunk.text[:100],
            }
            for sc in private_hits
        ],
        "raw_doc_hits": [
            {
                "doc_id": sc.chunk.doc_id,
                "score": sc.rerank_score,
                "text_preview": sc.chunk.text[:100],
            }
            for sc in doc_hits
        ],
        "meta": {
            "latency_ms": total_time,
            "answer_latency_ms": answer_result["latency_ms"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": answer_result["total_tokens"],
            "estimated_cost": estimated_cost,
            "model_used": "gpt-4.1-mini",
            "private_chunks_used": len(private_hits),
            "doc_chunks_used": len(doc_hits),
        },
    }
