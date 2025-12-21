"""
Account-aware RAG pipeline for private CRM text (MVP2).

Answers questions scoped to a specific company/account using:
- Private CRM text (history, opportunity notes, attachments)
- Optionally, global product docs

Usage:
    from backend.rag.account import answer_account_question
    
    result = answer_account_question(
        "What's the status with Acme?",
        company_name="Acme Manufacturing"
    )
    print(result["answer"])
"""

import logging
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.private import (
    PrivateRetrievalBackend,
    create_private_backend,
)
from backend.rag.ingest.text_builder import find_csv_dir
from backend.rag.config import get_config
from backend.rag.utils import estimate_tokens, preprocess_query, extract_citations
from backend.common.llm_client import call_llm_safe, call_llm_with_metrics


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for a CRM system.
Rewrite the user's question to be clearer and more specific for searching CRM records.
Keep it in natural language.
Only output the rewritten query, nothing else."""

HYDE_SYSTEM = """You are a CRM assistant. Given a question about a customer account,
write a short hypothetical answer (2-3 sentences) as if from CRM records.
Include relevant terms like history, notes, opportunities, activities.
Only output the hypothetical answer."""

ANSWER_SYSTEM = """You are an AI assistant answering questions about a specific customer account in a CRM system.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this in the available account data."
3. Cite your sources using [source_id] format, e.g., [history::HIST-ACME-CALL1] or [opp_note::OPP-ACME-UPGRADE].
4. Be concise but complete.
5. Focus on the specific account mentioned.

{context}

Question: {question}

Answer (with citations):"""


# =============================================================================
# Company Resolution
# =============================================================================

def load_companies_df() -> pd.DataFrame:
    """Load companies.csv."""
    csv_dir = find_csv_dir()
    companies_path = csv_dir / "companies.csv"
    if not companies_path.exists():
        raise FileNotFoundError(f"companies.csv not found in {csv_dir}")
    return pd.read_csv(companies_path)


def resolve_company_id(
    company_id: Optional[str] = None,
    company_name: Optional[str] = None,
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
# Context Building
# =============================================================================

def build_private_context(
    chunks: list[ScoredChunk],
    company_id: str,
    max_tokens: Optional[int] = None,
) -> tuple[str, list[dict]]:
    """
    Build context string from private chunks.
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    config = get_config()
    max_tokens = max_tokens or config.max_context_tokens
    max_per_type = config.max_chunks_per_type
    
    # Group by type and apply per-type cap
    by_type = defaultdict(list)
    for sc in chunks:
        t = sc.chunk.metadata.get("type", "unknown")
        if len(by_type[t]) < max_per_type:
            by_type[t].append(sc)
    
    # Flatten back, maintaining score order
    selected = []
    for t in ["history", "opportunity_note", "attachment"]:
        selected.extend(by_type.get(t, []))
    
    # Build context
    parts = []
    sources = []
    total_tokens = 0
    
    header = f"PRIVATE CRM CONTEXT (scoped to company_id={company_id})"
    parts.append(header)
    parts.append("=" * 50)
    
    for sc in selected:
        chunk = sc.chunk
        source_id = chunk.metadata.get("source_id", chunk.doc_id)
        chunk_type = chunk.metadata.get("type", "unknown")
        
        # Format chunk
        chunk_text = f"\n[{source_id}] ({chunk_type})\n"
        if chunk.title:
            chunk_text += f"Title: {chunk.title}\n"
        chunk_text += f"{chunk.text}\n"
        
        chunk_tokens = estimate_tokens(chunk_text)
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        parts.append(chunk_text)
        total_tokens += chunk_tokens
        
        sources.append({
            "type": chunk_type,
            "id": source_id,
            "label": chunk.title or source_id,
            "company_id": chunk.metadata.get("company_id", ""),
        })
    
    return "\n".join(parts), sources


def build_docs_context(
    chunks: list[ScoredChunk],
    max_tokens: int = 1000,
) -> tuple[str, list[dict]]:
    """
    Build context string from product docs chunks.
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    if not chunks:
        return "", []
    
    parts = []
    sources = []
    total_tokens = 0
    
    header = "PRODUCT DOCS CONTEXT"
    parts.append(header)
    parts.append("=" * 50)
    
    for sc in chunks:
        chunk = sc.chunk
        
        chunk_text = f"\n[{chunk.doc_id}]\n{chunk.text}\n"
        chunk_tokens = estimate_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        parts.append(chunk_text)
        total_tokens += chunk_tokens
        
        sources.append({
            "type": "doc",
            "id": chunk.doc_id,
            "label": chunk.title or chunk.doc_id,
        })
    
    return "\n".join(parts), sources


# =============================================================================
# Answer Generation
# =============================================================================

def generate_answer(
    question: str,
    context: str,
) -> dict:
    """Generate answer using LLM."""
    prompt = ANSWER_SYSTEM.format(context=context, question=question)
    
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
_private_backend: Optional[PrivateRetrievalBackend] = None
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
            from backend.rag.retrieval import create_backend
            _docs_backend = create_backend()
        except Exception as e:
            print(f"Warning: Could not load docs backend: {e}")
            _docs_backend = False  # Mark as unavailable
    return _docs_backend if _docs_backend else None


def answer_account_question(
    question: str,
    company_id: Optional[str] = None,
    company_name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
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


# =============================================================================
# CLI
# =============================================================================

def main():
    """Interactive CLI for account RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Account-aware CRM RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--company", "-c", help="Company name or ID")
    parser.add_argument("--include-docs", action="store_true", help="Also search product docs")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if not args.question:
        # Interactive mode
        print("=" * 60)
        print("Account-aware CRM RAG (MVP2)")
        print("=" * 60)
        
        # Show available companies
        df = load_companies_df()
        print("\nAvailable companies:")
        for _, row in df.iterrows():
            print(f"  - {row['name']} ({row['company_id']})")
        
        print("\nEnter questions in format: @CompanyName What is the status?")
        print("Or: What is the status? --company CompanyName")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            
            # Parse @Company prefix
            company = args.company
            question = user_input
            
            if user_input.startswith("@"):
                parts = user_input[1:].split(" ", 1)
                if len(parts) == 2:
                    company = parts[0]
                    question = parts[1]
            
            if not company:
                print("Please specify a company with @CompanyName or --company")
                continue
            
            try:
                result = answer_account_question(
                    question,
                    company_name=company,
                    include_docs=args.include_docs,
                    verbose=args.verbose,
                )
                
                print(f"\n[{result['company_name']}]")
                print(result["answer"])
                print(f"\n(Sources: {len(result['sources'])}, Latency: {result['meta']['latency_ms']:.0f}ms)")
                print()
                
            except Exception as e:
                print(f"Error: {e}\n")
    
    else:
        # Single question mode
        if not args.company:
            print("Error: --company is required")
            return
        
        result = answer_account_question(
            args.question,
            company_name=args.company,
            include_docs=args.include_docs,
            verbose=args.verbose,
        )
        
        print(f"\nCompany: {result['company_name']} ({result['company_id']})")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "-" * 60)
        print(f"Sources: {[s['id'] for s in result['sources'][:5]]}")
        print(f"Latency: {result['meta']['latency_ms']:.0f}ms")
        print(f"Tokens: {result['meta']['total_tokens']}")
        print(f"Est. cost: ${result['meta']['estimated_cost']:.4f}")


if __name__ == "__main__":
    main()
