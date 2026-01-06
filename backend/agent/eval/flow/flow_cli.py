"""CLI for flow evaluation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.panel import Panel

# Load environment before other imports
load_dotenv()

from backend.agent.question_tree import get_tree_stats
from backend.agent.eval.base import console, format_percentage, ensure_qdrant_collections
from backend.agent.eval.shared import finalize_eval_cli
from backend.agent.eval.formatting import print_debug_failures
from backend.agent.eval.models import (
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    SLO_FLOW_GROUNDED,
)
from backend.agent.eval.flow.flow_runner import run_flow_eval
from backend.agent.eval.flow.flow_output import print_summary, save_results, check_qdrant_access

app = typer.Typer()

# Use absolute path for baseline
_BACKEND_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
BASELINE_PATH = _BACKEND_ROOT / "agent" / "eval" / "output" / "flow_eval_baseline.json"


async def _run_eval_async(
    limit: int | None,
    verbose: bool,
    parallel: bool,
    workers: int,
    no_judge: bool,
    output: str | None,
    baseline: str | None,
    set_baseline: bool,
    debug: bool,
) -> int:
    """Async implementation of the eval runner."""
    # Check if Qdrant is accessible
    if not check_qdrant_access():
        console.print(
            Panel(
                "[red bold]ERROR: Qdrant storage is locked by another process![/red bold]\n\n"
                "[bold]Solutions:[/bold]\n"
                "  1. Stop the backend server: Ctrl+C in the uvicorn terminal\n"
                "  2. Close any Jupyter notebooks using RAG\n"
                "  3. Run with --mock for testing without real LLM/RAG",
                border_style="red",
            )
        )
        return 1

    # Warmup: trigger model loading
    console.print("\n[dim]Warming up models...[/dim]")
    try:
        from backend.agent.rag.tools import tool_docs_rag

        tool_docs_rag("warmup", top_k=1)
        console.print("[dim]Models loaded.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Model preload failed: {e}[/yellow]")

    # Show tree stats
    stats = get_tree_stats()
    console.print("\n[bold]Question Tree Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    # Run evaluation
    use_judge = not no_judge
    concurrency = workers if parallel else 1
    try:
        results = await run_flow_eval(
            max_paths=limit,
            verbose=verbose,
            use_judge=use_judge,
            concurrency=concurrency,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback

        traceback.print_exc()
        return 1

    # Print summary
    print_summary(results)

    # Debug output for failing paths
    if debug and results.failed_paths:

        def format_failed_path(i: int, item: dict) -> None:
            console.print(f"\n[bold cyan]--- Path {item['path_id']} ---[/bold cyan]")
            for j, step in enumerate(item["steps"]):
                status = "PASS" if step["passed"] else "FAIL"
                console.print(f"[bold]Q{j + 1}:[/bold] {step['question']}")
                console.print(f"    [{status}] R={step['relevance_score']} G={step['grounded_score']}")
                console.print(f"    [bold]Answer:[/bold] {step['answer'][:200]}...")
                if step["judge_explanation"]:
                    console.print(f"    [bold]Judge:[/bold] {step['judge_explanation']}")

        print_debug_failures(
            [
                {
                    "path_id": fp.path_id,
                    "steps": [
                        {
                            "question": s.question,
                            "passed": s.passed,
                            "relevance_score": s.relevance_score,
                            "grounded_score": s.grounded_score,
                            "answer": s.answer,
                            "judge_explanation": s.judge_explanation,
                        }
                        for s in fp.steps
                    ],
                }
                for fp in results.failed_paths
            ],
            title="Full details for failed paths",
            format_item=format_failed_path,
        )

    # Save if requested
    if output:
        save_results(results, Path(output))

    # Cleanup Qdrant client
    try:
        from backend.agent.rag.client import close_qdrant_client

        close_qdrant_client()
    except Exception:
        pass

    # Build SLO checks
    slo_checks = [
        (
            "Path Pass Rate",
            results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE,
            format_percentage(results.path_pass_rate),
            f">={format_percentage(SLO_FLOW_PATH_PASS_RATE)}",
        ),
        (
            "Question Pass Rate",
            results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE,
            format_percentage(results.question_pass_rate),
            f">={format_percentage(SLO_FLOW_QUESTION_PASS_RATE)}",
        ),
        (
            "Relevance",
            results.avg_relevance >= SLO_FLOW_RELEVANCE,
            format_percentage(results.avg_relevance),
            f">={format_percentage(SLO_FLOW_RELEVANCE)}",
        ),
        (
            "Groundedness",
            results.avg_grounded >= SLO_FLOW_GROUNDED,
            format_percentage(results.avg_grounded),
            f">={format_percentage(SLO_FLOW_GROUNDED)}",
        ),
    ]

    # Prepare baseline data
    baseline_data = {
        "path_pass_rate": results.path_pass_rate,
        "question_pass_rate": results.question_pass_rate,
        "avg_relevance": results.avg_relevance,
        "avg_grounded": results.avg_grounded,
    } if set_baseline else None

    # Finalize CLI: baseline comparison, SLO panel, exit code
    baseline_path = Path(baseline) if baseline else BASELINE_PATH
    return finalize_eval_cli(
        primary_score=results.question_pass_rate,
        slo_checks=slo_checks,
        baseline_path=baseline_path,
        score_key="question_pass_rate",
        set_baseline=set_baseline,
        baseline_data=baseline_data,
        extra_failure_check=results.paths_failed > 0,
        extra_failure_reason=f"{results.paths_failed} paths failed",
    )


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output for each question"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", "-p", help="Run flows in parallel"
    ),
    workers: int = typer.Option(5, "--workers", "-w", help="Max parallel workers"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b", help="Path to baseline JSON for regression comparison"
    ),
    set_baseline: bool = typer.Option(
        False, "--set-baseline", help="Save current results as new baseline"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    exit_code = asyncio.run(
        _run_eval_async(
            limit=limit,
            verbose=verbose,
            parallel=parallel,
            workers=workers,
            no_judge=no_judge,
            output=output,
            baseline=baseline,
            set_baseline=set_baseline,
            debug=debug,
        )
    )
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    print("Checking Qdrant collections...")
    ensure_qdrant_collections()
    print()
    app()
