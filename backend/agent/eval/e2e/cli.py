"""CLI for E2E evaluation."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

# Load environment before other imports
_project_root = Path(__file__).parent.parent.parent.parent.parent
load_dotenv(_project_root / ".env")

from backend.agent.eval.base import console, ensure_qdrant_collections
from backend.agent.eval.shared import (
    finalize_eval_cli,
    save_eval_results,
    print_debug_failures,
)
from backend.agent.eval.models import (
    SLO_ROUTER_ACCURACY,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
)
from backend.agent.eval.tracking import print_e2e_tracking_report
from backend.agent.eval.e2e.runner import run_e2e_eval
from backend.agent.eval.e2e.output import print_e2e_eval_results

app = typer.Typer()

# Use absolute path for baseline
_BACKEND_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
BASELINE_PATH = _BACKEND_ROOT / "data" / "processed" / "e2e_eval_baseline.json"


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of tests to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run tests in parallel"),
    workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b", help="Path to baseline JSON for regression comparison"
    ),
    set_baseline: bool = typer.Option(
        False, "--set-baseline", help="Save current results as new baseline"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing cases"),
) -> None:
    """Run end-to-end agent evaluation."""
    if no_judge:
        console.print(
            "[yellow]Warning: --no-judge is ignored in e2e_eval "
            "(judge is required for quality metrics)[/yellow]"
        )

    results, summary = run_e2e_eval(
        limit=limit, verbose=verbose, parallel=parallel, max_workers=workers
    )
    print_e2e_eval_results(results, summary)

    # Debug output for failing cases
    if debug:
        ungrounded = [r for r in results if r.answer_grounded == 0 or r.faithfulness == 0]

        def format_ungrounded(i: int, item: dict) -> None:
            console.print(f"\n[bold cyan]--- Case {i + 1}: {item['test_case_id']} ---[/bold cyan]")
            console.print(f"[bold]Question:[/bold] {item['question']}")
            console.print(f"[bold]Category:[/bold] {item['category']}")
            console.print(
                f"[bold]Grounded:[/bold] {item['answer_grounded']}, "
                f"[bold]Faithful:[/bold] {item['faithfulness']}, "
                f"[bold]CtxRel:[/bold] {item['context_relevance']}"
            )
            console.print(f"[bold]Sources:[/bold] {item['sources']}")
            console.print(f"[bold]Answer:[/bold]\n{item['answer']}")
            console.print(f"[bold]Judge Says:[/bold] {item['judge_explanation']}")

        print_debug_failures(
            [
                {
                    "test_case_id": r.test_case_id,
                    "question": r.question,
                    "category": r.category,
                    "answer_grounded": r.answer_grounded,
                    "faithfulness": r.faithfulness,
                    "context_relevance": r.context_relevance,
                    "sources": r.sources,
                    "answer": r.answer,
                    "judge_explanation": r.judge_explanation,
                }
                for r in ungrounded
            ],
            title="Full details for ungrounded answers",
            format_item=format_ungrounded,
        )

    # Print tracking report
    print_e2e_tracking_report(results, summary)

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
                "answer_grounded": r.answer_grounded,
                "context_relevance": r.context_relevance,
                "faithfulness": r.faithfulness,
                "latency_ms": r.latency_ms,
                "error": r.error,
            },
        )

    # Build SLO checks
    slo_checks = [
        ("Intent Classification", summary.intent_accuracy >= SLO_ROUTER_ACCURACY, "", ""),
        ("Answer Relevance", summary.answer_relevance_rate >= SLO_ANSWER_RELEVANCE, "", ""),
        ("Groundedness", summary.groundedness_rate >= SLO_GROUNDEDNESS, "", ""),
    ]

    # Finalize CLI: baseline comparison, SLO panel, exit code
    baseline_path = Path(baseline) if baseline else BASELINE_PATH
    exit_code = finalize_eval_cli(
        primary_score=summary.answer_relevance_rate,
        slo_checks=slo_checks,
        baseline_path=baseline_path,
        score_key="answer_relevance_rate",
        set_baseline=set_baseline,
        baseline_data=summary.model_dump() if set_baseline else None,
    )

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    print("Checking Qdrant collections...")
    ensure_qdrant_collections()
    print()
    app()
