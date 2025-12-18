"""
Evaluation harness for Account-aware RAG (MVP2).

Evaluates account-scoped RAG using:
- RAG triad metrics (context relevance, answer relevance, groundedness)
- Privacy leakage detection (retrieved chunks from wrong company)
- Latency and cost tracking

Usage:
    python -m project1_rag.eval_account_rag
"""

import json
import time
import random
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import pandas as pd

from project1_rag.private_text_builder import find_csv_dir
from project1_rag.ingest_private_text import (
    ingest_private_texts,
    PRIVATE_COLLECTION_NAME,
    QDRANT_PATH,
)
from project1_rag.private_retrieval import create_private_backend
from project1_rag.account_rag import answer_account_question, load_companies_df
from shared.llm_client import call_llm
from qdrant_client import QdrantClient


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_PATH = Path("data/processed/eval_account_results.json")
NUM_QUESTIONS_PER_COMPANY = 3
NUM_COMPANIES = 4
RANDOM_SEED = 42


# =============================================================================
# Models
# =============================================================================

class JudgeResult(BaseModel):
    """Result from LLM judge."""
    context_relevance: int  # 0 or 1
    answer_relevance: int   # 0 or 1
    groundedness: int       # 0 or 1
    needs_human_review: int # 0 or 1
    explanation: str


class AccountEvalResult(BaseModel):
    """Evaluation result for a single account question."""
    question_id: str
    company_id: str
    company_name: str
    question: str
    question_type: str
    answer: str
    judge_result: JudgeResult
    privacy_leakage: int  # 1 if any retrieved chunk from wrong company
    leaked_company_ids: list[str]
    num_private_hits: int
    latency_ms: float
    total_tokens: int
    estimated_cost: float


# =============================================================================
# Question Generation
# =============================================================================

def generate_eval_questions(seed: int = RANDOM_SEED) -> list[dict]:
    """
    Generate evaluation questions from actual CSV data.
    
    Returns list of dicts with:
        - id: question ID
        - company_id: target company
        - company_name: company name
        - question: the question text
        - question_type: category of question
    """
    random.seed(seed)
    
    # Load companies
    df = load_companies_df()
    
    # Filter to active companies with data
    active = df[df["status"].isin(["Active", "Trial"])]
    
    # Select companies (deterministic)
    selected = active.head(NUM_COMPANIES)
    
    questions = []
    q_id = 1
    
    # Question templates
    templates = [
        {
            "type": "history_summary",
            "template": "Summarize the recent interactions and history with {company_name}. What calls, emails, or meetings have occurred?",
        },
        {
            "type": "opportunity_status",
            "template": "What are the current opportunities for {company_name}? What stages are they in and what are the risks or next steps?",
        },
        {
            "type": "attachments",
            "template": "What documents or attachments are associated with {company_name}'s opportunities? Summarize what they contain.",
        },
    ]
    
    for _, company in selected.iterrows():
        company_id = company["company_id"]
        company_name = company["name"]
        
        for tmpl in templates[:NUM_QUESTIONS_PER_COMPANY]:
            question = tmpl["template"].format(
                company_name=company_name,
                company_id=company_id,
            )
            
            questions.append({
                "id": f"acct_q{q_id}",
                "company_id": company_id,
                "company_name": company_name,
                "question": question,
                "question_type": tmpl["type"],
            })
            q_id += 1
    
    return questions


# =============================================================================
# Judge
# =============================================================================

JUDGE_SYSTEM = """You are an expert evaluator for a CRM RAG system.
Evaluate an answer about a specific customer account.

Score (0 or 1):
1. CONTEXT_RELEVANCE: Does the retrieved context contain relevant account information?
2. ANSWER_RELEVANCE: Does the answer address the question about this account?
3. GROUNDEDNESS: Is the answer grounded in the context (no hallucinations)?
4. NEEDS_HUMAN_REVIEW: Should a human verify this answer?

Respond in JSON:
{
  "context_relevance": 0 or 1,
  "answer_relevance": 0 or 1,
  "groundedness": 0 or 1,
  "needs_human_review": 0 or 1,
  "explanation": "brief explanation"
}"""

JUDGE_PROMPT = """Company: {company_name} ({company_id})

Question: {question}

Retrieved Context (sources: {sources}):
{context}

Generated Answer:
{answer}

Evaluate this account-scoped RAG response:"""


def judge_account_response(
    company_id: str,
    company_name: str,
    question: str,
    context: str,
    answer: str,
    sources: list[str],
) -> JudgeResult:
    """Judge an account RAG response."""
    prompt = JUDGE_PROMPT.format(
        company_id=company_id,
        company_name=company_name,
        question=question,
        sources=", ".join(sources[:5]),
        context=context[:2500],  # Truncate
        answer=answer,
    )
    
    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM,
            model="gpt-4.1-mini",
            max_tokens=300,
        )
        
        # Parse JSON
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
        
        return JudgeResult(
            context_relevance=int(result.get("context_relevance", 0)),
            answer_relevance=int(result.get("answer_relevance", 0)),
            groundedness=int(result.get("groundedness", 0)),
            needs_human_review=int(result.get("needs_human_review", 1)),
            explanation=str(result.get("explanation", "")),
        )
    
    except Exception as e:
        return JudgeResult(
            context_relevance=0,
            answer_relevance=0,
            groundedness=0,
            needs_human_review=1,
            explanation=f"Judge error: {e}",
        )


# =============================================================================
# Privacy Leakage Detection
# =============================================================================

def check_privacy_leakage(
    target_company_id: str,
    raw_hits: list[dict],
) -> tuple[int, list[str]]:
    """
    Check if any retrieved chunks belong to a different company.
    
    Returns:
        Tuple of (leakage_flag, list_of_leaked_company_ids)
    """
    leaked = []
    for hit in raw_hits:
        hit_company = hit.get("company_id", "")
        if hit_company and hit_company != target_company_id:
            leaked.append(hit_company)
    
    leakage = 1 if leaked else 0
    return leakage, list(set(leaked))


# =============================================================================
# Evaluation
# =============================================================================

def ensure_private_collection_exists() -> None:
    """Ensure private Qdrant collection exists, create if not."""
    qdrant = QdrantClient(path=str(QDRANT_PATH))
    
    if not qdrant.collection_exists(PRIVATE_COLLECTION_NAME):
        print(f"Collection '{PRIVATE_COLLECTION_NAME}' not found, creating...")
        ingest_private_texts(recreate=True)
    else:
        info = qdrant.get_collection(PRIVATE_COLLECTION_NAME)
        if info.points_count == 0:
            print(f"Collection '{PRIVATE_COLLECTION_NAME}' is empty, rebuilding...")
            ingest_private_texts(recreate=True)
        else:
            print(f"Using existing collection with {info.points_count} points")


def evaluate_question(
    question_data: dict,
    verbose: bool = False,
) -> AccountEvalResult:
    """Evaluate a single account question."""
    q_id = question_data["id"]
    company_id = question_data["company_id"]
    company_name = question_data["company_name"]
    question = question_data["question"]
    q_type = question_data["question_type"]
    
    if verbose:
        print(f"\n  {q_id}: {question[:50]}...")
    
    # Run RAG
    result = answer_account_question(
        question=question,
        company_id=company_id,
        verbose=False,
    )
    
    # Check privacy leakage
    leakage, leaked_ids = check_privacy_leakage(
        company_id, result["raw_private_hits"]
    )
    
    if verbose and leakage:
        print(f"    WARNING: Privacy leakage! Leaked companies: {leaked_ids}")
    
    # Build context string for judge
    context_parts = []
    for hit in result["raw_private_hits"][:5]:
        context_parts.append(f"[{hit['id']}] {hit['text_preview']}")
    context_str = "\n".join(context_parts)
    
    # Judge
    sources = [s["id"] for s in result["sources"]]
    judge = judge_account_response(
        company_id=company_id,
        company_name=company_name,
        question=question,
        context=context_str,
        answer=result["answer"],
        sources=sources,
    )
    
    if verbose:
        print(f"    ctx={judge.context_relevance} ans={judge.answer_relevance} "
              f"gnd={judge.groundedness} leak={leakage}")
    
    return AccountEvalResult(
        question_id=q_id,
        company_id=company_id,
        company_name=company_name,
        question=question,
        question_type=q_type,
        answer=result["answer"],
        judge_result=judge,
        privacy_leakage=leakage,
        leaked_company_ids=leaked_ids,
        num_private_hits=len(result["raw_private_hits"]),
        latency_ms=result["meta"]["latency_ms"],
        total_tokens=result["meta"]["total_tokens"],
        estimated_cost=result["meta"]["estimated_cost"],
    )


def run_evaluation(verbose: bool = True) -> list[AccountEvalResult]:
    """Run full evaluation."""
    print("=" * 60)
    print("Account RAG Evaluation (MVP2)")
    print("=" * 60)
    
    # Ensure collection exists
    ensure_private_collection_exists()
    
    # Generate questions
    questions = generate_eval_questions()
    print(f"\nEvaluating {len(questions)} questions across "
          f"{len(set(q['company_id'] for q in questions))} companies")
    
    results = []
    for q in questions:
        result = evaluate_question(q, verbose=verbose)
        results.append(result)
    
    return results


def print_summary(results: list[AccountEvalResult]) -> None:
    """Print evaluation summary."""
    n = len(results)
    if n == 0:
        print("No results")
        return
    
    # Triad metrics
    ctx_rel = sum(r.judge_result.context_relevance for r in results) / n
    ans_rel = sum(r.judge_result.answer_relevance for r in results) / n
    grounded = sum(r.judge_result.groundedness for r in results) / n
    
    triad_success = sum(
        1 for r in results
        if r.judge_result.context_relevance == 1
        and r.judge_result.answer_relevance == 1
        and r.judge_result.groundedness == 1
    ) / n
    
    # Privacy leakage
    leakage_count = sum(r.privacy_leakage for r in results)
    leakage_rate = leakage_count / n
    
    # Latency and cost
    avg_latency = sum(r.latency_ms for r in results) / n
    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.estimated_cost for r in results)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Questions evaluated':<25} {n:>10}")
    print(f"{'Context Relevance':<25} {ctx_rel:>10.1%}")
    print(f"{'Answer Relevance':<25} {ans_rel:>10.1%}")
    print(f"{'Groundedness':<25} {grounded:>10.1%}")
    print(f"{'RAG Triad Success':<25} {triad_success:>10.1%}")
    print("-" * 40)
    print(f"{'Privacy Leakage Rate':<25} {leakage_rate:>10.1%}")
    print(f"{'Leaked Questions':<25} {leakage_count:>10}")
    print("-" * 40)
    print(f"{'Avg Latency (ms)':<25} {avg_latency:>10.0f}")
    print(f"{'Total Tokens':<25} {total_tokens:>10,}")
    print(f"{'Total Cost ($)':<25} {total_cost:>10.4f}")
    
    # Per-company breakdown
    print("\n" + "-" * 60)
    print("PER-COMPANY RESULTS")
    print("-" * 60)
    
    by_company = {}
    for r in results:
        cid = r.company_id
        if cid not in by_company:
            by_company[cid] = {"results": [], "name": r.company_name}
        by_company[cid]["results"].append(r)
    
    print(f"{'Company':<25} {'Triad':<8} {'Leak':<8} {'Latency':<10}")
    print("-" * 60)
    
    for cid, data in sorted(by_company.items()):
        company_results = data["results"]
        cn = len(company_results)
        
        company_triad = sum(
            1 for r in company_results
            if r.judge_result.context_relevance == 1
            and r.judge_result.answer_relevance == 1
            and r.judge_result.groundedness == 1
        ) / cn
        
        company_leak = sum(r.privacy_leakage for r in company_results) / cn
        company_latency = sum(r.latency_ms for r in company_results) / cn
        
        print(f"{data['name'][:24]:<25} {company_triad:<8.0%} {company_leak:<8.0%} {company_latency:<10.0f}ms")
    
    # Questions with issues
    issues = [r for r in results if r.judge_result.groundedness == 0 or r.privacy_leakage == 1]
    if issues:
        print("\n" + "-" * 60)
        print("QUESTIONS WITH ISSUES")
        print("-" * 60)
        for r in issues:
            flags = []
            if r.judge_result.groundedness == 0:
                flags.append("NOT_GROUNDED")
            if r.privacy_leakage == 1:
                flags.append(f"LEAKED:{r.leaked_company_ids}")
            print(f"\n{r.question_id} [{', '.join(flags)}]")
            print(f"  Company: {r.company_name}")
            print(f"  Question: {r.question[:60]}...")
            print(f"  Judge: {r.judge_result.explanation[:80]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Account RAG (MVP2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = [r.model_dump() for r in results]
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
