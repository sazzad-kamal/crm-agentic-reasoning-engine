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
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.progress import track
from rich.table import Table
import pandas as pd
from qdrant_client import QdrantClient

from backend.rag.retrieval.constants import PRIVATE_COLLECTION, QDRANT_PATH
from backend.rag.utils import find_csv_dir
from backend.rag.ingest.private_text import ingest_private_texts
from backend.rag.pipeline.account import answer_account_question, load_companies_df
from backend.rag.eval.models import (
    AccountEvalResult,
    AccountEvalSummary,
    SLO_CONTEXT_RELEVANCE,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_RAG_TRIAD,
    SLO_PRIVACY_LEAKAGE,
    SLO_LATENCY_P95_MS,
)
from backend.rag.eval.judge import judge_account_response, check_privacy_leakage
from backend.rag.eval.questions import generate_eval_questions, NUM_COMPANIES, NUM_QUESTIONS_PER_COMPANY, RANDOM_SEED
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


# =============================================================================
# Evaluation
# =============================================================================

def ensure_private_collection_exists() -> None:
    """Ensure private Qdrant collection exists, create if not."""
    qdrant = QdrantClient(path=str(QDRANT_PATH))
    
    try:
        if not qdrant.collection_exists(PRIVATE_COLLECTION):
            print(f"Collection '{PRIVATE_COLLECTION}' not found, creating...")
            qdrant.close()  # Close before calling ingest which opens its own client
            ingest_private_texts(recreate=True)
        else:
            info = qdrant.get_collection(PRIVATE_COLLECTION)
            if info.points_count == 0:
                print(f"Collection '{PRIVATE_COLLECTION}' is empty, rebuilding...")
                qdrant.close()  # Close before calling ingest which opens its own client
                ingest_private_texts(recreate=True)
            else:
                print(f"Using existing collection with {info.points_count} points")
                qdrant.close()
    except Exception:
        qdrant.close()
        raise


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


def run_evaluation(verbose: bool = True) -> tuple[list[AccountEvalResult], AccountEvalSummary]:
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
    
    # Compute summary
    summary = compute_account_summary(results)
    
    return results, summary


def compute_account_summary(results: list[AccountEvalResult]) -> AccountEvalSummary:
    """Compute summary statistics and check SLOs."""
    n = len(results)
    
    if n == 0:
        return AccountEvalSummary(
            total_tests=0,
            context_relevance=0.0,
            answer_relevance=0.0,
            groundedness=0.0,
            rag_triad_success=0.0,
            privacy_leakage_rate=0.0,
            leaked_questions=0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_tokens=0,
            total_cost=0.0,
        )
    
    # Compute aggregates
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
    
    # P95 latency
    latencies = sorted([r.latency_ms for r in results])
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_index, len(latencies) - 1)]
    
    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.estimated_cost for r in results)
    
    # Check SLOs
    failed_slos = []
    if ctx_rel < SLO_CONTEXT_RELEVANCE:
        failed_slos.append(f"Context relevance {ctx_rel:.1%} < {SLO_CONTEXT_RELEVANCE:.1%}")
    if ans_rel < SLO_ANSWER_RELEVANCE:
        failed_slos.append(f"Answer relevance {ans_rel:.1%} < {SLO_ANSWER_RELEVANCE:.1%}")
    if grounded < SLO_GROUNDEDNESS:
        failed_slos.append(f"Groundedness {grounded:.1%} < {SLO_GROUNDEDNESS:.1%}")
    if triad_success < SLO_RAG_TRIAD:
        failed_slos.append(f"RAG triad {triad_success:.1%} < {SLO_RAG_TRIAD:.1%}")
    if leakage_rate > SLO_PRIVACY_LEAKAGE:
        failed_slos.append(f"Privacy leakage {leakage_rate:.1%} > {SLO_PRIVACY_LEAKAGE:.1%}")
    if p95_latency > SLO_LATENCY_P95_MS:
        failed_slos.append(f"P95 latency {p95_latency:.0f}ms > {SLO_LATENCY_P95_MS}ms")
    
    return AccountEvalSummary(
        total_tests=n,
        context_relevance=ctx_rel,
        answer_relevance=ans_rel,
        groundedness=grounded,
        rag_triad_success=triad_success,
        privacy_leakage_rate=leakage_rate,
        leaked_questions=leakage_count,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        total_tokens=total_tokens,
        total_cost=total_cost,
        all_slos_passed=len(failed_slos) == 0,
        failed_slos=failed_slos,
    )


def print_summary(results: list[AccountEvalResult], summary: AccountEvalSummary) -> None:
    """Print evaluation summary using Rich."""
    n = len(results)
    if n == 0:
        console.print("[yellow]No results[/yellow]")
        return
    
    # Summary table using shared helper
    summary_table = create_summary_table()
    
    summary_table.add_row("Questions evaluated", str(summary.total_tests))
    
    ctx_style = "[green]" if summary.context_relevance >= SLO_CONTEXT_RELEVANCE else "[red]"
    summary_table.add_row("Context Relevance", f"{ctx_style}{summary.context_relevance:.1%}[/] (SLO: ≥{SLO_CONTEXT_RELEVANCE:.0%})")
    
    ans_style = "[green]" if summary.answer_relevance >= SLO_ANSWER_RELEVANCE else "[red]"
    summary_table.add_row("Answer Relevance", f"{ans_style}{summary.answer_relevance:.1%}[/] (SLO: ≥{SLO_ANSWER_RELEVANCE:.0%})")
    
    gnd_style = "[green]" if summary.groundedness >= SLO_GROUNDEDNESS else "[red]"
    summary_table.add_row("Groundedness", f"{gnd_style}{summary.groundedness:.1%}[/] (SLO: ≥{SLO_GROUNDEDNESS:.0%})")
    
    triad_style = "[green]" if summary.rag_triad_success >= SLO_RAG_TRIAD else "[red]"
    summary_table.add_row("RAG Triad Success", f"[bold {triad_style[1:-1]}]{summary.rag_triad_success:.1%}[/bold {triad_style[1:-1]}] (SLO: ≥{SLO_RAG_TRIAD:.0%})")
    
    add_separator_row(summary_table)
    leak_style = "[red]" if summary.privacy_leakage_rate > SLO_PRIVACY_LEAKAGE else "[green]"
    summary_table.add_row("Privacy Leakage Rate", f"{leak_style}{summary.privacy_leakage_rate:.1%}[/] (SLO: {SLO_PRIVACY_LEAKAGE:.0%})")
    summary_table.add_row("Leaked Questions", f"{leak_style}{summary.leaked_questions}[/]")
    
    add_separator_row(summary_table)
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    
    p95_style = "[green]" if summary.p95_latency_ms <= SLO_LATENCY_P95_MS else "[red]"
    summary_table.add_row("P95 Latency", f"{p95_style}{summary.p95_latency_ms:.0f}ms[/] (SLO: ≤{SLO_LATENCY_P95_MS}ms)")
    
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost:.4f}")
    
    # SLO status
    add_separator_row(summary_table)
    if summary.all_slos_passed:
        summary_table.add_row("SLO Status", "[bold green]✓ ALL PASSED[/bold green]")
    else:
        summary_table.add_row("SLO Status", f"[bold red]✗ {len(summary.failed_slos)} FAILED[/bold red]")
    
    console.print(summary_table)
    
    # Per-company breakdown using shared helper
    by_company: dict = {}
    for r in results:
        by_company.setdefault(r.company_id, {"results": [], "name": r.company_name})["results"].append(r)
    
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
    results, summary = run_evaluation(verbose=verbose)
    
    # Print summary
    print_summary(results, summary)
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "results": [r.model_dump() for r in results],
        "summary": summary.model_dump(),
    }
    with open(output, "w") as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output}[/dim]")
    
    # Exit code based on SLOs
    if not summary.all_slos_passed:
        console.print("\n[red bold]FAIL: One or more SLOs not met[/red bold]")
        for slo in summary.failed_slos:
            console.print(f"  • {slo}")
        raise typer.Exit(code=1)
    else:
        console.print("\n[green bold]✓ PASS: All SLOs met[/green bold]")


def main():
    """Main entrypoint."""
    app()


if __name__ == "__main__":
    main()