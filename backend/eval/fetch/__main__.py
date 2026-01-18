"""CLI for fetch node evaluation."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

import logging

import typer

from backend.eval.fetch.runner import print_summary, run_sql_eval

app = typer.Typer()


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of questions to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output for each case"),
    difficulty: str | None = typer.Option(
        None,
        "--difficulty",
        "-d",
        help="Filter by difficulty (1-5), e.g. '1,2' or '4,5' (default: all)",
    ),
) -> None:
    """Run fetch node evaluation (tests SQL planner and RAG in isolation)."""
    logging.basicConfig(level=logging.WARNING)

    # Parse difficulty filter
    difficulty_filter: list[int] | None = None
    if difficulty:
        try:
            difficulty_filter = [int(d.strip()) for d in difficulty.split(",")]
        except ValueError as e:
            print(f"Invalid difficulty filter: {difficulty}. Use comma-separated numbers 1-5.")
            raise typer.Exit(1) from e

    results = run_sql_eval(limit=limit, verbose=verbose, difficulty_filter=difficulty_filter)
    print_summary(results)


if __name__ == "__main__":
    app()
