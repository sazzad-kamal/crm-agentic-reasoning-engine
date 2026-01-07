"""CLI for E2E evaluation."""

from __future__ import annotations

import atexit
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load environment before other imports
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

from backend.agent.rag.client import close_qdrant_client
from backend.eval.base import console, ensure_qdrant_collections

# Register cleanup to prevent shutdown errors
atexit.register(close_qdrant_client)
from backend.eval.e2e.output import print_e2e_eval_results
from backend.eval.e2e.runner import run_e2e_eval
from backend.eval.formatting import print_debug_failures
from backend.eval.shared import save_eval_results

app = typer.Typer()


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of tests to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run tests in parallel"),
    workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing cases"),
    latency: bool = typer.Option(True, "--latency/--no-latency", help="Show per-node latency breakdown from LangSmith"),
) -> None:
    """Run end-to-end agent evaluation."""
    results, summary = run_e2e_eval(
        limit=limit, verbose=verbose, parallel=parallel, max_workers=workers
    )
    print_e2e_eval_results(results, summary)

    # Debug output for failing cases
    if debug:
        low_quality = [r for r in results if r.faithfulness < 0.5 or r.answer_relevance < 0.5]

        def format_failure(i: int, item: dict) -> None:
            console.print(f"\n[bold cyan]--- Case {i + 1}: {item['test_case_id']} ---[/bold cyan]")
            console.print(f"[bold]Question:[/bold] {item['question']}")
            console.print(f"[bold]Category:[/bold] {item['category']}")
            console.print(
                f"[bold]Relevance:[/bold] {item['answer_relevance']:.2f}, "
                f"[bold]Faithfulness:[/bold] {item['faithfulness']:.2f}, "
                f"[bold]CtxPrec:[/bold] {item['context_precision']:.2f}"
            )
            console.print(f"[bold]Sources:[/bold] {item['sources']}")
            console.print(f"[bold]Answer:[/bold]\n{item['answer']}")

        print_debug_failures(
            [
                {
                    "test_case_id": r.test_case_id,
                    "question": r.question,
                    "category": r.category,
                    "answer_relevance": r.answer_relevance,
                    "faithfulness": r.faithfulness,
                    "context_precision": r.context_precision,
                    "sources": r.sources,
                    "answer": r.answer,
                }
                for r in low_quality
            ],
            title="Full details for low quality answers",
            format_item=format_failure,
        )

    # Save results to file if requested
    if output:
        save_eval_results(
            output,
            summary,
            results,
            result_mapper=lambda r: {
                "test_case_id": r.test_case_id,
                "question": r.question,
                "category": r.category,
                "company_correct": r.company_correct,
                "intent_correct": r.intent_correct,
                "answer_relevance": r.answer_relevance,
                "faithfulness": r.faithfulness,
                "context_precision": r.context_precision,
                "latency_ms": r.latency_ms,
                "error": r.error,
            },
        )

    # Show LangSmith latency breakdown if requested
    if latency:
        from backend.eval.langsmith_latency import print_latency_breakdown
        print_latency_breakdown(minutes_ago=60)


if __name__ == "__main__":
    print("Checking Qdrant collections...")
    ensure_qdrant_collections()
    print()
    app()
