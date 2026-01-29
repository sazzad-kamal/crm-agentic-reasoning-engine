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
from backend.eval.integration.runner import run_convo_eval
from backend.eval.integration.tree import get_tree_stats

app = typer.Typer()


def _run_eval(
    limit: int | None,
    output: str | None,
    debug: bool,
) -> None:
    """Run the conversation evaluation."""
    eval_start_time = time.time()

    # Show tree stats
    stats = get_tree_stats()
    print("\nQuestion Tree Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Run evaluation (sequential)
    try:
        results = run_convo_eval(max_paths=limit)
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

    # Debug output for failing cases
    if debug:
        failed = [c for c in results.cases if not c.passed]
        if failed:
            print("\nDEBUG: Failed questions")
            for c in failed[:10]:
                status = "PASS" if c.passed else "FAIL"
                print(f"Q: {c.question}")
                print(f"    [{status}] R={c.relevance_score} A={c.answer_correctness_score}")
                print(f"    Answer: {c.answer[:200]}...")
                if c.errors:
                    print(f"    Errors: {'; '.join(c.errors)}")

    # Save if requested
    if output:
        save_results(results, Path(output))


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing cases"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(
        limit=limit,
        output=output,
        debug=debug,
    )


if __name__ == "__main__":
    app()
