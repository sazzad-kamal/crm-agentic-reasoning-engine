"""
Evaluation harness for Account-aware RAG (MVP2).

Evaluates account-scoped RAG using:
- RAG triad metrics (context relevance, answer relevance, groundedness)
- Privacy leakage detection (retrieved chunks from wrong company)
- Latency and cost tracking

Usage:
    python -m backend.rag.eval.account_eval
    python -m backend.rag.eval.account_eval --verbose
"""

import json
import random
from pathlib import Path
from typing import Optional

import typer
from rich.progress import track
from rich.table import Table
import pandas as pd
from qdrant_client import QdrantClient

from backend.rag.config import PRIVATE_COLLECTION, QDRANT_PATH
from backend.rag.ingest.text_builder import find_csv_dir
from backend.rag.ingest.private_text import ingest_private_texts
from backend.rag.pipeline.account import answer_account_question, load_companies_df
from backend.rag.eval.models import AccountEvalResult
from backend.rag.eval.judge import judge_account_response, check_privacy_leakage
from backend.rag.eval.base import (
    console,
    create_summary_table,
    create_detail_table,
    print_eval_header,
    print_issues_panel,
    add_separator_row,
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_PATH = Path("data/processed/eval_account_results.json")
NUM_QUESTIONS_PER_COMPANY = 3
NUM_COMPANIES = 4
RANDOM_SEED = 42


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
# Evaluation
# =============================================================================

def ensure_private_collection_exists() -> None:
    """Ensure private Qdrant collection exists, create if not."""
    qdrant = QdrantClient(path=str(QDRANT_PATH))
    
    if not qdrant.collection_exists(PRIVATE_COLLECTION):
        print(f"Collection '{PRIVATE_COLLECTION}' not found, creating...")
        ingest_private_texts(recreate=True)
    else:
        info = qdrant.get_collection(PRIVATE_COLLECTION)
        if info.points_count == 0:
            print(f"Collection '{PRIVATE_COLLECTION}' is empty, rebuilding...")
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
    # Ensure collection exists
    ensure_private_collection_exists()
    
    # Generate questions
    questions = generate_eval_questions()
    num_companies = len(set(q['company_id'] for q in questions))
    
    print_eval_header(
        "Account RAG Evaluation (MVP2)",
        f"Evaluating [bold]{len(questions)}[/bold] questions across "
        f"[bold]{num_companies}[/bold] companies",
    )
    
    results = []
    for q in track(questions, description="Evaluating..."):
        result = evaluate_question(q, verbose=verbose)
        results.append(result)
    
    return results


def print_summary(results: list[AccountEvalResult]) -> None:
    """Print evaluation summary using Rich."""
    n = len(results)
    if n == 0:
        console.print("[yellow]No results[/yellow]")
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
    
    # Summary table using shared helper
    summary_table = create_summary_table()
    
    summary_table.add_row("Questions evaluated", str(n))
    summary_table.add_row("Context Relevance", f"{ctx_rel:.1%}")
    summary_table.add_row("Answer Relevance", f"{ans_rel:.1%}")
    summary_table.add_row("Groundedness", f"{grounded:.1%}")
    summary_table.add_row("RAG Triad Success", f"[bold green]{triad_success:.1%}[/bold green]")
    add_separator_row(summary_table)
    leak_style = "[red]" if leakage_rate > 0 else "[green]"
    summary_table.add_row("Privacy Leakage Rate", f"{leak_style}{leakage_rate:.1%}[/]")
    summary_table.add_row("Leaked Questions", f"{leak_style}{leakage_count}[/]")
    add_separator_row(summary_table)
    summary_table.add_row("Avg Latency", f"{avg_latency:.0f}ms")
    summary_table.add_row("Total Tokens", f"{total_tokens:,}")
    summary_table.add_row("Total Cost", f"${total_cost:.4f}")
    
    console.print(summary_table)
    
    # Per-company breakdown using shared helper
    by_company = {}
    for r in results:
        cid = r.company_id
        if cid not in by_company:
            by_company[cid] = {"results": [], "name": r.company_name}
        by_company[cid]["results"].append(r)
    
    company_table = create_detail_table("Per-Company Results", [
        ("Company", "left"),
        ("Triad", "right"),
        ("Leak", "right"),
        ("Latency", "right"),
    ])
    
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
        
        leak_style = "[red]" if company_leak > 0 else "[green]"
        company_table.add_row(
            data["name"][:24],
            f"{company_triad:.0%}",
            f"{leak_style}{company_leak:.0%}[/]",
            f"{company_latency:.0f}ms",
        )
    
    console.print(company_table)
    
    # Questions with issues using shared helper
    issues = [r for r in results if r.judge_result.groundedness == 0 or r.privacy_leakage == 1]
    issue_lines = []
    for r in issues:
        flags = []
        if r.judge_result.groundedness == 0:
            flags.append("[red]NOT_GROUNDED[/red]")
        if r.privacy_leakage == 1:
            flags.append(f"[red]LEAKED:{r.leaked_company_ids}[/red]")
        issue_lines.append(
            f"[bold]{r.question_id}[/bold] [{', '.join(flags)}]\n"
            f"  Company: {r.company_name}\n"
            f"  Question: {r.question[:60]}..."
        )
    
    print_issues_panel("Questions With Issues", issue_lines)


# =============================================================================
# Main
# =============================================================================

app = typer.Typer(help="Account RAG Evaluation Harness (MVP2)")


@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    output: Path = typer.Option(
        OUTPUT_PATH,
        "--output", "-o",
        help="Output file for results",
    ),
):
    """Run evaluation on generated account questions."""
    # Run evaluation
    results = run_evaluation(verbose=verbose)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    results_data = [r.model_dump() for r in results]
    with open(output, "w") as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output}[/dim]")


def main():
    """Main entrypoint."""
    app()


if __name__ == "__main__":
    main()
