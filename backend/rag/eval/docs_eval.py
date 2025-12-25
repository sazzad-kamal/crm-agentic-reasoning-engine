"""
RAG Evaluation Harness for Acme CRM docs.

Evaluates the RAG pipeline using:
- Context relevance: Did we retrieve the right documents?
- Answer relevance: Does the answer address the question?
- Groundedness: Is the answer grounded in the context (no hallucinations)?

Uses LLM-as-judge for relevance and groundedness scoring.

Usage:
    python -m backend.rag.eval.docs_eval
    python -m backend.rag.eval.docs_eval --verbose
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.progress import track

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.models import EvalResult
from backend.rag.eval.judge import judge_response, compute_doc_recall
from backend.rag.eval.base import (
    console,
    create_summary_table,
    create_detail_table,
    print_eval_header,
    print_issues_panel,
    format_check_mark,
    add_separator_row,
)


# =============================================================================
# Configuration
# =============================================================================

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_eval_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_question(
    question_data: dict,
    backend,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a single question through the RAG pipeline.
    
    Args:
        question_data: Dict with id, question, target_doc_ids
        backend: Initialized RetrievalBackend
        verbose: Print progress
        
    Returns:
        EvalResult with all metrics
    """
    question_id = question_data["id"]
    question = question_data["question"]
    target_doc_ids = question_data["target_doc_ids"]
    
    if verbose:
        print(f"\nEvaluating: {question_id}")
        print(f"  Question: {question[:60]}...")
    
    # Run RAG pipeline
    start_time = time.time()
    result = answer_question(question, backend, k=8, verbose=False)
    total_latency = (time.time() - start_time) * 1000
    
    # Build context string for judge
    context = "\n\n".join([
        f"[{c.doc_id}] {c.text[:500]}"
        for c in result["used_chunks"]
    ])
    
    # Compute doc recall
    doc_recall = compute_doc_recall(target_doc_ids, result["doc_ids_used"])
    
    if verbose:
        print(f"  Retrieved: {result['doc_ids_used']}")
        print(f"  Doc recall: {doc_recall:.2f}")
    
    # Judge the response
    judge_result = judge_response(
        question=question,
        context=context,
        answer=result["answer"],
        doc_ids=result["doc_ids_used"],
    )
    
    if verbose:
        print(f"  Judge: context={judge_result.context_relevance}, "
              f"answer={judge_result.answer_relevance}, "
              f"grounded={judge_result.groundedness}")
    
    return EvalResult(
        question_id=question_id,
        question=question,
        target_doc_ids=target_doc_ids,
        retrieved_doc_ids=result["doc_ids_used"],
        answer=result["answer"],
        judge_result=judge_result,
        doc_recall=doc_recall,
        latency_ms=total_latency,
        total_tokens=result["metrics"]["total_tokens"],
    )


def run_evaluation(
    questions: list[dict] | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Run full evaluation over all questions.
    
    Args:
        questions: List of question dicts (or load from file)
        verbose: Print progress
        
    Returns:
        List of EvalResult objects
    """
    if questions is None:
        questions = load_eval_questions()
    
    print_eval_header(
        "RAG Evaluation Harness",
        f"Evaluating [bold]{len(questions)}[/bold] questions",
    )
    
    # Initialize backend
    with console.status("[bold green]Loading backend..."):
        backend = create_backend()
    
    results = []
    for q in track(questions, description="Evaluating..."):
        result = evaluate_question(q, backend, verbose=verbose)
        results.append(result)
    
    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print summary statistics from evaluation results using Rich."""
    n = len(results)
    
    if n == 0:
        console.print("[yellow]No results to summarize[/yellow]")
        return
    
    # Compute aggregates
    context_relevance = sum(r.judge_result.context_relevance for r in results) / n
    answer_relevance = sum(r.judge_result.answer_relevance for r in results) / n
    groundedness = sum(r.judge_result.groundedness for r in results) / n
    needs_review = sum(r.judge_result.needs_human_review for r in results) / n
    
    # RAG triad success (all three = 1)
    triad_success = sum(
        1 for r in results
        if r.judge_result.context_relevance == 1
        and r.judge_result.answer_relevance == 1
        and r.judge_result.groundedness == 1
    ) / n
    
    avg_doc_recall = sum(r.doc_recall for r in results) / n
    avg_latency = sum(r.latency_ms for r in results) / n
    total_tokens = sum(r.total_tokens for r in results)
    
    # Approximate cost (GPT-4.1-mini pricing)
    estimated_cost = (total_tokens * 0.8 * 0.40 + total_tokens * 0.2 * 1.60) / 1_000_000
    
    # Summary table using shared helper
    summary_table = create_summary_table()
    
    summary_table.add_row("Questions evaluated", str(n))
    summary_table.add_row("Context Relevance", f"{context_relevance:.1%}")
    summary_table.add_row("Answer Relevance", f"{answer_relevance:.1%}")
    summary_table.add_row("Groundedness", f"{groundedness:.1%}")
    summary_table.add_row("RAG Triad Success", f"[bold green]{triad_success:.1%}[/bold green]")
    summary_table.add_row("Needs Human Review", f"{needs_review:.1%}")
    summary_table.add_row("Avg Doc Recall", f"{avg_doc_recall:.1%}")
    add_separator_row(summary_table)
    summary_table.add_row("Avg Latency", f"{avg_latency:.0f}ms")
    summary_table.add_row("Total Tokens", f"{total_tokens:,}")
    summary_table.add_row("Est. Cost", f"${estimated_cost:.4f}")
    
    console.print(summary_table)
    
    # Per-question results using shared helper
    detail_table = create_detail_table("Per-Question Results", [
        ("ID", "left"),
        ("Ctx", "center"),
        ("Ans", "center"),
        ("Gnd", "center"),
        ("Recall", "right"),
        ("Latency", "right"),
    ])
    
    for r in results:
        detail_table.add_row(
            r.question_id,
            format_check_mark(r.judge_result.context_relevance == 1),
            format_check_mark(r.judge_result.answer_relevance == 1),
            format_check_mark(r.judge_result.groundedness == 1),
            f"{r.doc_recall:.1%}",
            f"{r.latency_ms:.0f}ms",
        )
    
    console.print(detail_table)
    
    # Failed questions using shared helper
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    print_issues_panel(
        "Questions Needing Attention",
        [f"[bold]{r.question_id}[/bold]: {r.question}\n  [dim]{r.judge_result.explanation}[/dim]" for r in failed],
    )


# =============================================================================
# CLI Entrypoint
# =============================================================================

app = typer.Typer(help="RAG Evaluation Harness")


@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    output: Path = typer.Option(
        Path("data/processed/eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
):
    """Run evaluation on all test questions."""
    results = run_evaluation(verbose=verbose)
    print_summary(results)
    
    # Save results to file
    output.parent.mkdir(parents=True, exist_ok=True)
    results_data = [r.model_dump() for r in results]
    with open(output, "w") as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output}[/dim]")


def main():
    """Main entrypoint for evaluation."""
    app()


if __name__ == "__main__":
    main()
