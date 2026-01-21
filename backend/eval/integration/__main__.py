"""CLI for integration evaluation."""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

# Fix Windows asyncio cleanup issues with httpx/RAGAS
# Use SelectorEventLoop instead of ProactorEventLoop to avoid "Event loop is closed" errors
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

import logging
import time

import typer

from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.output import print_summary, save_results
from backend.eval.integration.runner import run_flow_eval
from backend.eval.integration.tree import get_tree_stats
from backend.eval.shared.formatting import console

app = typer.Typer()


def _run_eval(
    limit: int | None,
    verbose: bool,
    no_judge: bool,
    output: str | None,
    debug: bool,
) -> None:
    """Run the flow evaluation."""
    eval_start_time = time.time()

    # Show tree stats
    stats = get_tree_stats()
    console.print("\n[bold]Question Tree Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    # Run evaluation (sequential)
    use_judge = not no_judge
    try:
        results = run_flow_eval(
            max_paths=limit,
            verbose=verbose,
            use_judge=use_judge,
            concurrency=1,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback

        traceback.print_exc()
        return

    # Fetch latency breakdown from LangSmith (optional, informational only)
    elapsed_minutes = int((time.time() - eval_start_time) / 60) + 1
    latency_pcts = get_latency_percentages(minutes_ago=max(elapsed_minutes, 5))

    # Print summary with optional LangSmith info
    print_summary(results, latency_pcts=latency_pcts)

    # Debug output for failing paths
    if debug and results.failed_paths:
        console.print("\n[bold yellow]DEBUG: Failed paths[/bold yellow]")
        for fp in results.failed_paths[:10]:
            console.print(f"\n[bold cyan]--- Path {fp.path_id} ---[/bold cyan]")
            for j, s in enumerate(fp.steps):
                status = "PASS" if s.passed else "FAIL"
                console.print(f"[bold]Q{j + 1}:[/bold] {s.question}")
                console.print(f"    [{status}] R={s.relevance_score} F={s.faithfulness_score}")
                console.print(f"    [bold]Answer:[/bold] {s.answer[:200]}...")
                if s.judge_explanation:
                    console.print(f"    [bold]Judge:[/bold] {s.judge_explanation}")

    # Save if requested
    if output:
        save_results(results, Path(output))


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output for each question"
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(
        limit=limit,
        verbose=verbose,
        no_judge=no_judge,
        output=output,
        debug=debug,
    )


if __name__ == "__main__":
    app()
