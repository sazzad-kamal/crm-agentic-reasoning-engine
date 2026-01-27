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
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]

import logging
import time

import typer

from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.output import print_summary, save_results
from backend.eval.integration.runner import run_flow_eval
from backend.eval.integration.tree import get_tree_stats

app = typer.Typer()


def _run_eval(
    limit: int | None,
    no_judge: bool,
    output: str | None,
    debug: bool,
) -> None:
    """Run the flow evaluation."""
    eval_start_time = time.time()

    # Show tree stats
    stats = get_tree_stats()
    print("\nQuestion Tree Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Run evaluation (sequential)
    use_judge = not no_judge
    try:
        results = run_flow_eval(
            max_paths=limit,
            use_judge=use_judge,
            concurrency=1,
        )
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
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
        print("\nDEBUG: Failed paths")
        for fp in results.failed_paths[:10]:
            print(f"\n--- Path {fp.path_id} ---")
            for j, s in enumerate(fp.steps):
                status = "PASS" if s.passed else "FAIL"
                print(f"Q{j + 1}: {s.question}")
                print(f"    [{status}] R={s.relevance_score} F={s.faithfulness_score}")
                print(f"    Answer: {s.answer[:200]}...")
                if s.judge_explanation:
                    print(f"    Judge: {s.judge_explanation}")

    # Save if requested
    if output:
        save_results(results, Path(output))


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(
        limit=limit,
        no_judge=no_judge,
        output=output,
        debug=debug,
    )


if __name__ == "__main__":
    app()
